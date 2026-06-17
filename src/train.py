"""Treina o detector XGBoost do KIM e avalia no teste (reprodução).

Hiperparâmetros da Tabela 2 do artigo: objetivo binary:logistic, 1000 árvores, lr 0.05,
max_depth 6, subsample 0.8, colsample_bytree 0.8, lambda (L2) 1.0, gamma 0, min_child_weight 1.

Metas do artigo: PR-AUC 0.93 | ROC-AUC 0.99 | F1 0.97.

Uso:
    python -m src.train                 # 1000 árvores (full)
    python -m src.train --n-trees 200   # mais rápido p/ validar
    python -m src.train --scale-pos-weight   # pesa a classe minoritária (alternativa ao CTGAN)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import xgboost as xgb

from . import data as datamod
from .evaluate import report

ROOT = Path(__file__).resolve().parents[1]

# Tabela 2 do artigo
PARAMS = dict(
    objective="binary:logistic",
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    gamma=0.0,
    min_child_weight=1,
    tree_method="hist",   # eficiente p/ ~7M amostras
    eval_metric=["auc", "aucpr"],
)


def run(n_trees: int = 1000, scale_pos_weight: bool = False, seed: int = 42,
        save_as: str = "xgb_kim.json"):
    print("Carregando dados...")
    Xtr, ytr, Xte, yte = datamod.load(mmap=True)
    # XGBoost lida bem com arrays grandes; materializa em memória (float32, ~340 MB).
    Xtr = np.asarray(Xtr); Xte = np.asarray(Xte)
    print(f"  treino={Xtr.shape}  teste={Xte.shape}  ataque%(treino)={100*ytr.mean():.2f}")

    params = dict(PARAMS, n_estimators=n_trees, random_state=seed, n_jobs=-1)
    if scale_pos_weight:
        spw = (ytr == 0).sum() / max((ytr == 1).sum(), 1)
        params["scale_pos_weight"] = float(spw)
        print(f"  scale_pos_weight = {spw:.2f}")

    print(f"Treinando XGBoost ({n_trees} árvores)...")
    model = xgb.XGBClassifier(**params)
    model.fit(Xtr, ytr, eval_set=[(Xte, yte)], verbose=False)

    print("\nAvaliando no teste:")
    scores = model.predict_proba(Xte)[:, 1]
    metrics = report(yte, scores)

    out = ROOT / "results" / save_as
    out.parent.mkdir(exist_ok=True)
    model.save_model(out)
    (out.with_suffix(".metrics.json")).write_text(json.dumps(metrics, indent=2))
    print("\nmodelo salvo em", out)
    return model, metrics


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n-trees", type=int, default=1000)
    p.add_argument("--scale-pos-weight", action="store_true")
    args = p.parse_args()
    run(n_trees=args.n_trees, scale_pos_weight=args.scale_pos_weight)
