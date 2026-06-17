"""Métricas de avaliação do detector (binário, desbalanceado)."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score, accuracy_score,
    precision_recall_curve, confusion_matrix, precision_score, recall_score,
)


def best_f1_threshold(y_true, scores):
    """Encontra o limiar que maximiza o F1 (classe ataque) a partir da curva PR."""
    prec, rec, thr = precision_recall_curve(y_true, scores)
    f1 = 2 * prec * rec / (prec + rec + 1e-12)
    i = int(np.nanargmax(f1[:-1])) if len(thr) else 0
    return float(thr[i]) if len(thr) else 0.5, float(f1[i])


def report(y_true, scores, threshold: float | None = None) -> dict:
    """Avalia probabilidades de ataque vs. rótulos. Reporta PR-AUC, ROC-AUC e F1.

    Args:
        y_true: rótulos (N,) 0/1.
        scores: probabilidade da classe ataque (N,).
        threshold: limiar fixo; se None, usa o de melhor F1.

    Returns:
        dict com roc_auc, pr_auc, threshold, f1, precision, recall, confusion_matrix.
    """
    roc = roc_auc_score(y_true, scores)
    pr = average_precision_score(y_true, scores)
    if threshold is None:
        threshold, _ = best_f1_threshold(y_true, scores)
    y_pred = (scores >= threshold).astype(int)
    f1 = f1_score(y_true, y_pred, zero_division=0)            # F1 da classe ataque
    f1w = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    f1m = f1_score(y_true, y_pred, average="macro", zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    p = precision_score(y_true, y_pred, zero_division=0)
    r = recall_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    print(f"ROC-AUC: {roc:.4f}  | PR-AUC: {pr:.4f}")
    print(f"Limiar: {threshold:.4f}")
    print(f"  F1 (ataque): {f1:.4f}  | Precision: {p:.4f}  | Recall: {r:.4f}")
    print(f"  F1 ponderado: {f1w:.4f}  | F1 macro: {f1m:.4f}  | Accuracy: {acc:.4f}")
    print("  (o artigo reporta F1 ponderado ~0.97)")
    print("Matriz de confusão (linha=verdadeiro; [normal, ataque]):")
    for row in cm:
        print("   ", row.tolist())
    return {"roc_auc": roc, "pr_auc": pr, "threshold": threshold,
            "f1_attack": f1, "f1_weighted": f1w, "f1_macro": f1m, "accuracy": acc,
            "precision": p, "recall": r, "confusion_matrix": cm.tolist()}
