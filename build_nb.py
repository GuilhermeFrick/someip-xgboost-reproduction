# Gera Kim_Colab.ipynb — reprodução do XGBoost IDS (Kim 2026) no Colab
import json, os

def md(*l): return {"cell_type":"markdown","metadata":{},"source":[x if x.endswith("\n") else x+"\n" for x in l]}
def code(*l): return {"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":[x if x.endswith("\n") else x+"\n" for x in l]}

cells = [
md("# Reprodução — Kim et al. (2026), XGBoost Anomaly Detection para SOME/IP",
   "",
   "Treina o detector **XGBoost** nas 12 *features* comportamentais e avalia.",
   "Metas do artigo: **ROC-AUC 0,99 · PR-AUC 0,93 · F1 0,97**.",
   "",
   "**Não precisa de GPU** (XGBoost roda em CPU; o dataset cabe na RAM do Colab).",
   "",
   "**Achado-chave:** o \"F1 0,97\" do artigo é o **F1 ponderado** (dominado pela classe",
   "normal). O F1 da **classe ataque** é ~0,84 (recall ~0,74) — relevante para deploy real."),

md("## 0. Setup — clonar o repo (dados via Git LFS) e instalar deps"),
code("import os, sys",
    "REPO = 'someip-xgboost-reproduction'",
    "if not os.path.isdir('src'):",
    "    os.system(f'git clone https://github.com/GuilhermeFrick/{REPO}.git')",
    "    os.chdir(REPO)",
    "    os.system('git lfs install && git lfs pull')   # baixa os .npz (~122 MB)",
    "sys.path.insert(0, '.')",
    "!pip -q install xgboost scikit-learn",
    "print('cwd:', os.getcwd(), '| dados:', os.path.exists('data/X_train.npz'))"),

md("## 1. Treinar XGBoost + avaliar",
   "",
   "Hiperparâmetros da Tabela 2 do artigo. `n_trees=300` é um bom equilíbrio para o notebook",
   "(o full do artigo é 1000). `scale_pos_weight` rebalanceia a classe minoritária."),
code("from src.train import run",
    "model, metrics = run(n_trees=300, scale_pos_weight=True)"),

md("## 2. Resultados finais — métricas, matriz de confusão e curva ROC"),
code("import numpy as np, matplotlib.pyplot as plt, pandas as pd",
    "from sklearn.metrics import (confusion_matrix, roc_curve, auc, precision_recall_curve,",
    "                             f1_score, precision_score, recall_score)",
    "from src import data as datamod",
    "",
    "_, _, X_test, y_test = datamod.load()",
    "scores = model.predict_proba(X_test)[:, 1]",
    "",
    "# limiar de melhor F1 (classe ataque)",
    "prec, rec, thr = precision_recall_curve(y_test, scores)",
    "f1c = 2*prec*rec/(prec+rec+1e-12)",
    "th = thr[int(np.nanargmax(f1c[:-1]))]",
    "y_pred = (scores >= th).astype(int)",
    "",
    "# tabela: F1 da classe ataque vs F1 ponderado",
    "tab = pd.DataFrame({",
    "    'Métrica': ['ROC-AUC', 'PR-AUC', 'F1 (classe ATAQUE)', 'F1 PONDERADO', 'Recall ataque', 'Precision ataque'],",
    "    'Valor': [round(auc(*roc_curve(y_test, scores)[:2]), 4),",
    "              round(metrics['pr_auc'], 4),",
    "              round(f1_score(y_test, y_pred), 4),",
    "              round(f1_score(y_test, y_pred, average='weighted'), 4),",
    "              round(recall_score(y_test, y_pred), 4),",
    "              round(precision_score(y_test, y_pred), 4)]})",
    "print('O artigo destaca o F1 PONDERADO (~0,97); o F1 da classe ATAQUE é bem menor.')",
    "tab"),
code("# Matriz de confusão + curva ROC",
    "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))",
    "cm = confusion_matrix(y_test, y_pred)",
    "ax1.imshow(cm, cmap='Blues')",
    "ax1.set_xticks([0,1]); ax1.set_yticks([0,1])",
    "ax1.set_xticklabels(['normal','ataque']); ax1.set_yticklabels(['normal','ataque'])",
    "ax1.set_xlabel('Predito'); ax1.set_ylabel('Verdadeiro'); ax1.set_title('Matriz de Confusão')",
    "for i in range(2):",
    "    for j in range(2):",
    "        ax1.text(j, i, f'{cm[i,j]:,}', ha='center', va='center',",
    "                 color='white' if cm[i,j] > cm.max()/2 else 'black')",
    "fpr, tpr, _ = roc_curve(y_test, scores)",
    "ax2.plot(fpr, tpr, lw=2, label=f'ROC (AUC={auc(fpr,tpr):.3f})')",
    "ax2.plot([0,1],[0,1],'k--',lw=0.7)",
    "ax2.set_xlabel('Taxa de Falso Positivo'); ax2.set_ylabel('Taxa de Verdadeiro Positivo')",
    "ax2.set_title('Curva ROC'); ax2.legend(loc='lower right')",
    "plt.tight_layout(); plt.show()"),

md("---",
   "**Esperado (nossa reprodução):** ROC-AUC ~0,99 e PR-AUC ~0,93 (reproduzem o artigo);",
   "F1 ponderado ~0,97, mas F1 da classe **ataque** ~0,84 (recall ~0,74 → ~26% dos ataques",
   "passam). Ver `docs/resultados.md`."),
]

nb = {"cells":cells,"metadata":{"colab":{"provenance":[]},
      "kernelspec":{"display_name":"Python 3","name":"python3"},
      "language_info":{"name":"python"}},"nbformat":4,"nbformat_minor":0}

with open("Kim_Colab.ipynb","w",encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print("notebook gerado com", len(cells), "celulas")
