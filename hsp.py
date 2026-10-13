"""
HSP (Hansen Solubility Parameters) predictor using RDKit + LightGBM + Optuna.
Enhanced to better distinguish cyclic ethers/acetals like C1COCOC1 and C1COCCO1.
Outputs R2, MSE, MAE and execution logs to both console and a text file (hsp_execution.log).

Usage (Windows CMD):
    python hsp.py
    python hsp.py --retrain
    python hsp.py --tune
    python hsp.py --evaluate
    python hsp.py --smiles "C1COCOC1"
    python hsp.py --smiles "C1COCCO1"
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import optuna
from lightgbm import LGBMRegressor
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold, train_test_split

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
EXCEL_PATH = BASE_DIR / "HSP_data_p.xlsx"
MODEL_DIR = BASE_DIR / "models_lgbm"
LOG_PATH = BASE_DIR / "hsp_execution.log"

TARGETS = ("δD", "δP", "δH")
REQUIRED_COLUMNS = ("SMILES", *TARGETS)
RANDOM_STATE = 42
MODEL_VERSION = 1  # 評価機能・特徴量定義の更新

# ---------------------------------------------------------------------
# Logging Setup (Console + File)
# ---------------------------------------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.handlers.clear()  # 重複ハンドラーの防止

formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

# コンソール用ハンドラー
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

# テキストファイル用ハンドラー (UTF-8)
file_handler = logging.FileHandler(LOG_PATH, encoding="utf-8")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

optuna.logging.set_verbosity(optuna.logging.WARNING)

# 官能基および環状アセタール・エーテル構造を区別するためのSMARTS
FUNCTIONAL_GROUP_SMARTS = {
    "CH3_count": "[CH3]",
    "CH2_count": "[CH2]",
    "CH_count": "[CH]",
    "OH_count": "[OX2H]",
    "Ether_count": "[#6]-O-[#6]",
    "Carbonyl_count": "[CX3]=[OX1]",
    "Amine_count": "[NX3;H2;!$(NC=O)]",
    "Cyclic_Acetal_O_C_O": "[OX2][CX4][OX2]", # O-C-O結合（アセタール構造）
    "Ring_5_member": "[r5]",                # 5員環
    "Ring_6_member": "[r6]",                # 6員環
}

COMPILED_SMARTS = {
    name: Chem.MolFromSmarts(pattern)
    for name, pattern in FUNCTIONAL_GROUP_SMARTS.items()
}
if any(pattern is None for pattern in COMPILED_SMARTS.values()):
    raise RuntimeError("官能基 SMARTS のコンパイルに失敗しました。")

FEATURE_NAMES = (
    list(FUNCTIONAL_GROUP_SMARTS)
    + [
        "num_atoms",
        "molecular_weight",
        "tpsa",
        "hbd",
        "hba",
        "logp",
        "fraction_csp3",       # sp3炭素の比率
        "hall_kier_alpha",     # 形状・トポロジー指標
    ]
)


class InvalidSmilesError(ValueError):
    """Raised when an input SMILES cannot be parsed by RDKit."""


def clean_smiles(smiles: Any) -> str:
    """Normalize supported aliases and validate the input type."""
    if pd.isna(smiles):
        raise InvalidSmilesError("SMILES が空欄です。")
    value = str(smiles).strip()
    if not value:
        raise InvalidSmilesError("SMILES が空欄です。")
    if value.upper() == "HOH":
        return "O"
    return value


def extract_features(smiles: Any) -> list[float]:
    """Convert one SMILES string into a fixed-order numeric feature vector."""
    value = clean_smiles(smiles)
    mol = Chem.MolFromSmiles(value)
    if mol is None:
        raise InvalidSmilesError(f"SMILES を解析できません: {value!r}")

    counts = [
        len(mol.GetSubstructMatches(pattern))
        for pattern in COMPILED_SMARTS.values()
    ]
    return counts + [
        mol.GetNumAtoms(),
        Descriptors.MolWt(mol),
        rdMolDescriptors.CalcTPSA(mol),
        rdMolDescriptors.CalcNumHBD(mol),
        rdMolDescriptors.CalcNumHBA(mol),
        Descriptors.MolLogP(mol),
        rdMolDescriptors.CalcFractionCSP3(mol),
        Chem.GraphDescriptors.HallKierAlpha(mol),
    ]


def features_frame(smiles_values: list[Any] | pd.Series) -> pd.DataFrame:
    rows = []
    for row_number, smiles in enumerate(smiles_values, start=2):
        try:
            rows.append(extract_features(smiles))
        except InvalidSmilesError as exc:
            raise InvalidSmilesError(f"Excel の {row_number} 行目: {exc}") from exc
    return pd.DataFrame(rows, columns=FEATURE_NAMES, dtype=float)


def load_training_data(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    if not path.is_file():
        raise FileNotFoundError(
            f"学習データが見つかりません: {path}\n"
            "hsp.py と同じフォルダーに HSP_data_p.xlsx を置いてください。"
        )

    df = pd.read_excel(path)
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Excel に必須列がありません: {', '.join(missing)}")

    df = df.copy()
    df["SMILES"] = df["SMILES"].map(
        lambda value: clean_smiles(value) if not pd.isna(value) else ""
    )
    for target in TARGETS:
        df[target] = pd.to_numeric(df[target], errors="coerce")

    before = len(df)
    df = df.dropna(subset=[*TARGETS]).copy()
    df = df[df["SMILES"].str.len() > 0].copy()
    if len(df) != before:
        logger.warning("空欄または数値でない目的変数を含む行を %d 件除外しました。", before - len(df))
    if len(df) < 3:
        raise ValueError("有効な学習データが不足しています（最低3行が必要です）。")

    X = features_frame(df["SMILES"])
    fingerprint_source = df[["SMILES", *TARGETS]].to_csv(index=False).encode("utf-8")
    fingerprint = hashlib.sha256(
        fingerprint_source + json.dumps(FEATURE_NAMES).encode("utf-8")
    ).hexdigest()
    return df.reset_index(drop=True), X, fingerprint


def make_model(params: dict[str, Any] | None = None) -> LGBMRegressor:
    default_params = {
        "objective": "regression",
        "n_estimators": 600,
        "num_leaves": 15,
        "min_child_samples": 5,
        "learning_rate": 0.03,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "random_state": RANDOM_STATE,
        "verbosity": -1,
        "n_jobs": -1,
    }
    if params:
        default_params.update(params)
    return LGBMRegressor(**default_params)


def optimize_hyperparameters(X: pd.DataFrame, y: pd.Series, target_name: str) -> dict[str, Any]:
    logger.info("[%s] Optuna によるハイパーパラメータ最適化を開始します（30トライアル）...", target_name)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 800, step=100),
            "num_leaves": trial.suggest_int("num_leaves", 7, 31),
            "min_child_samples": trial.suggest_int("min_child_samples", 2, max(2, len(X) // 2)),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        }

        n_splits = min(5, len(X))
        if n_splits < 2:
            model = make_model(params)
            model.fit(X, y)
            pred = model.predict(X)
            return float(mean_absolute_error(y, pred))

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
        maes = []
        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            model = make_model(params)
            model.fit(X_train, y_train)
            pred = model.predict(X_val)
            maes.append(mean_absolute_error(y_val, pred))

        return float(np.mean(maes))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=30, show_progress_bar=False)
    logger.info("[%s] 最適化完了: 最良 MAE = %.4f", target_name, study.best_value)
    return study.best_params


def _log_metrics(target: str, evaluation_name: str, actual: np.ndarray, pred: np.ndarray) -> None:
    """共通の回帰評価指標をログに出力する。"""
    r2 = r2_score(actual, pred) if len(actual) >= 2 else float("nan")
    mse = mean_squared_error(actual, pred)
    mae = mean_absolute_error(actual, pred)
    logger.info(
        "%s [%s] -> R² = %.3f, MSE = %.3f, MAE = %.3f (n=%d)",
        target, evaluation_name, r2, mse, mae, len(actual)
    )


def evaluate_holdout(df: pd.DataFrame, X: pd.DataFrame) -> None:
    """通常のランダム分割によるホールドアウト評価。"""
    if len(df) < 10:
        logger.warning("データが10行未満のため、ランダム・ホールドアウト評価を省略します。")
        return

    indices = np.arange(len(df))
    train_idx, test_idx = train_test_split(
        indices, test_size=0.2, random_state=RANDOM_STATE
    )
    logger.info("ランダム・ホールドアウト評価（学習 %d 行 / テスト %d 行）", len(train_idx), len(test_idx))
    for target in TARGETS:
        model = make_model()
        model.fit(X.iloc[train_idx], df[target].iloc[train_idx])
        pred = model.predict(X.iloc[test_idx])
        _log_metrics(target, "Random holdout", df[target].iloc[test_idx].to_numpy(), pred)


def evaluate_cross_validation(df: pd.DataFrame, X: pd.DataFrame, n_splits: int = 5) -> None:
    """ランダムK-fold交差検証。平均値と標準偏差で分割依存性を確認する。"""
    n_splits = min(n_splits, len(df))
    if n_splits < 2:
        logger.warning("交差検証には最低2行必要なため、評価を省略します。")
        return

    logger.info("=== ランダム K-fold 交差検証 (%d-fold) ===", n_splits)
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    for target in TARGETS:
        maes, mses, r2s = [], [], []
        y = df[target].reset_index(drop=True)
        for fold, (train_idx, test_idx) in enumerate(splitter.split(X), start=1):
            model = make_model()
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
            pred = model.predict(X.iloc[test_idx])
            actual = y.iloc[test_idx].to_numpy()
            maes.append(mean_absolute_error(actual, pred))
            mses.append(mean_squared_error(actual, pred))
            if len(actual) >= 2:
                r2s.append(r2_score(actual, pred))
            logger.info(
                "%s [CV fold %d/%d] MAE=%.3f, MSE=%.3f%s",
                target, fold, n_splits, maes[-1], mses[-1],
                f", R²={r2s[-1]:.3f}" if len(actual) >= 2 else ""
            )
        logger.info(
            "%s [CV summary] MAE=%.3f ± %.3f, MSE=%.3f ± %.3f, R²=%.3f ± %.3f",
            target, np.mean(maes), np.std(maes), np.mean(mses), np.std(mses),
            np.mean(r2s) if r2s else float("nan"),
            np.std(r2s) if r2s else float("nan")
        )


def _scaffold_group(smiles: Any) -> str:
    """Bemis-Murcko骨格をグループ化キーにする。非環式分子は正規化SMILESで区別する。"""
    value = clean_smiles(smiles)
    mol = Chem.MolFromSmiles(value)
    if mol is None:
        raise InvalidSmilesError(f"SMILES を解析できません: {value!r}")
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    scaffold_smiles = Chem.MolToSmiles(scaffold, canonical=True)
    if scaffold_smiles:
        return f"SCAFFOLD:{scaffold_smiles}"
    # 非環式分子では空の骨格が全件同一になるのを避ける。
    canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
    return f"ACYCLIC:{canonical_smiles}"


def evaluate_structure_based(df: pd.DataFrame, X: pd.DataFrame, n_splits: int = 5) -> None:
    """Bemis-Murcko骨格を分割単位にしたGroupKFold評価。

    同じ骨格を持つ分子が学習側と評価側に分かれないようにする。
    非環式分子は正規化SMILESごとにグループ化される点に注意。
    """
    groups = np.asarray([_scaffold_group(s) for s in df["SMILES"]])
    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        logger.warning(
            "構造ベース評価を省略: 異なる骨格グループが%d個しかありません。", len(unique_groups)
        )
        return

    n_splits = min(n_splits, len(unique_groups))
    logger.info(
        "=== 構造ベース交差検証 (Bemis-Murcko / GroupKFold %d-fold, グループ数=%d) ===",
        n_splits, len(unique_groups)
    )
    splitter = GroupKFold(n_splits=n_splits)
    for target in TARGETS:
        maes, mses, r2s = [], [], []
        y = df[target].reset_index(drop=True)
        for fold, (train_idx, test_idx) in enumerate(splitter.split(X, y, groups), start=1):
            # 学習データに評価側と同じ骨格が混ざっていないことを確認
            train_groups = set(groups[train_idx])
            test_groups = set(groups[test_idx])
            if train_groups.intersection(test_groups):
                raise RuntimeError("構造ベース分割で骨格グループの重複を検出しました。")

            model = make_model()
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
            pred = model.predict(X.iloc[test_idx])
            actual = y.iloc[test_idx].to_numpy()
            maes.append(mean_absolute_error(actual, pred))
            mses.append(mean_squared_error(actual, pred))
            if len(actual) >= 2:
                r2s.append(r2_score(actual, pred))
            logger.info(
                "%s [Scaffold fold %d/%d] MAE=%.3f, MSE=%.3f%s, test_groups=%d",
                target, fold, n_splits, maes[-1], mses[-1],
                f", R²={r2s[-1]:.3f}" if len(actual) >= 2 else "",
                len(test_groups)
            )
        logger.info(
            "%s [Scaffold summary] MAE=%.3f ± %.3f, MSE=%.3f ± %.3f, R²=%.3f ± %.3f",
            target, np.mean(maes), np.std(maes), np.mean(mses), np.std(mses),
            np.mean(r2s) if r2s else float("nan"),
            np.std(r2s) if r2s else float("nan")
        )


def load_or_train_models(
    df: pd.DataFrame,
    X: pd.DataFrame,
    fingerprint: str,
    force_retrain: bool = False,
    tune_hyperparams: bool = False,
) -> dict[str, LGBMRegressor]:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    models: dict[str, LGBMRegressor] = {}

    for target in TARGETS:
        model_path = MODEL_DIR / f"HSP_LGBM_{target}.joblib"
        model = None
        best_params = None

        if model_path.exists() and not force_retrain and not tune_hyperparams:
            try:
                bundle = joblib.load(model_path)
                if (
                    isinstance(bundle, dict)
                    and bundle.get("model_version") == MODEL_VERSION
                    and bundle.get("fingerprint") == fingerprint
                    and bundle.get("feature_names") == FEATURE_NAMES
                ):
                    model = bundle["model"]
                    logger.info("%s: 保存済みモデルをロード", target)
                else:
                    logger.info("%s: データまたは特徴量定義が変更されたため再学習", target)
            except Exception as exc:
                logger.warning("%s: モデルをロードできないため再学習します (%s)", target, exc)

        if model is None:
            if tune_hyperparams:
                best_params = optimize_hyperparameters(X, df[target], target)
            
            model = make_model(best_params)
            model.fit(X, df[target])
            bundle = {
                "model_version": MODEL_VERSION,
                "fingerprint": fingerprint,
                "feature_names": FEATURE_NAMES,
                "hyperparameters": best_params,
                "model": model,
            }
            joblib.dump(bundle, model_path)
            logger.info("%s: 全データで学習し保存しました -> %s", target, model_path)

        # 全データ（学習データ）に対する評価指標 (R2, MSE, MAE) を計算してログ出力
        preds = model.predict(X)
        actuals = df[target].to_numpy()
        r2 = r2_score(actuals, preds) if len(actuals) >= 2 else float("nan")
        mse = mean_squared_error(actuals, preds)
        mae = mean_absolute_error(actuals, preds)
        logger.info("[%s] 学習データ全体評価 -> R²: %.4f, MSE: %.4f, MAE: %.4f", target, r2, mse, mae)

        models[target] = model

    return models


def calculate_delta_t(delta_d: float, delta_p: float, delta_h: float) -> float:
    return round(float(np.sqrt(delta_d**2 + delta_p**2 + delta_h**2)), 2)


def predict_hsp(smiles: str, models: dict[str, LGBMRegressor]) -> dict[str, float]:
    X_pred = pd.DataFrame([extract_features(smiles)], columns=FEATURE_NAMES)
    result = {}
    for target in TARGETS:
        value = float(models[target].predict(X_pred)[0])
        result[target] = round(max(0.0, value), 2)

    result["δT"] = calculate_delta_t(
        result["δD"], result["δP"], result["δH"]
    )
    return result


def report_training_rows(df: pd.DataFrame, models: dict[str, LGBMRegressor]) -> None:
    print("\n=== 学習データへの予測（参考値）===")
    X = features_frame(df["SMILES"])
    preds = {target: models[target].predict(X) for target in TARGETS}
    for i, row in df.iterrows():
        name = row.get("化合物名", row["SMILES"])
        errors = {
            target: abs(float(row[target]) - float(preds[target][i]))
            for target in TARGETS
        }
        total_error = sum(errors.values())
        predicted = {target: round(max(0.0, float(preds[target][i])), 2) for target in TARGETS}
        predicted["δT"] = calculate_delta_t(
            predicted["δD"], predicted["δP"], predicted["δH"]
        )
        print(f"{name} | 誤差合計={total_error:.2f} | 予測={predicted}")


def main() -> None:
    parser = argparse.ArgumentParser(description="RDKit + LightGBM + Optuna による HSP 予測")
    parser.add_argument("--retrain", action="store_true", help="保存済みモデルを無視して再学習")
    parser.add_argument("--tune", action="store_true", help="Optuna を用いてハイパーパラメータを最適化して学習")
    parser.add_argument(
        "--evaluate", action="store_true",
        help="ランダム・ホールドアウト、交差検証、構造ベース交差検証を実行"
    )
    parser.add_argument(
        "--cv", action="store_true",
        help="ランダムK-fold交差検証を実行"
    )
    parser.add_argument(
        "--structure-evaluate", action="store_true",
        help="Bemis-Murcko骨格による構造ベース交差検証を実行"
    )
    parser.add_argument("--smiles", action="append", help="予測するSMILES（複数回指定可能）")
    args = parser.parse_args()

    df, X, fingerprint = load_training_data(EXCEL_PATH)
    logger.info("学習データ: %d 件、特徴量: %d 個", len(df), X.shape[1])

    if args.evaluate:
        evaluate_cross_validation(df, X)
        evaluate_structure_based(df, X)
    else:
        if args.cv:
            evaluate_cross_validation(df, X)
        if args.structure_evaluate:
            evaluate_structure_based(df, X)

    models = load_or_train_models(
        df, X, fingerprint, force_retrain=args.retrain, tune_hyperparams=args.tune
    )

    if args.smiles:
        test_smiles = args.smiles
    else:
        report_training_rows(df, models)
        test_smiles = ["C1COCOC1", "C1COCCO1"]  # 1,3,5-トリオキサンと 1,3-ジオキソランをテスト

    print("\n=== HSP 予測 ===")
    for smiles in test_smiles:
        try:
            res = predict_hsp(smiles, models)
            msg = f"{smiles} -> {res}"
            print(msg)
            logger.info(msg)
        except InvalidSmilesError as exc:
            logger.error("%s", exc)


if __name__ == "__main__":
    main()