# MEMORY — Reprodução Kim et al. (2026)

Quarto trabalho do mestrado (autor distinto dos demais). Reprodução em `C:/Mestrado/KIM-repro/`;
dados originais em `C:/Mestrado/KIM/`.

Kim, Park, You, Kwak, "XGBoost-Based Anomaly Detection Framework for SOME/IP in In-Vehicle
Networks", MDPI Systems 14(2):196, 2026.

## Natureza (o contraponto não-DL)
- **XGBoost** (gradient boosting), NÃO deep learning — único assim entre os 4 trabalhos.
- **Features comportamentais (12), deliberadamente SEM campos de header** (evitar overfitting):
  intervalo de tempo, log-likelihood/entropia de payload (modelo de bytes por posição + Laplace),
  distância de Hamming entre payloads, mudança de tamanho.
- Classificação **binária por pacote** (normal vs ataque). Ataques: DoS, Fuzzy, MITM (single/multi).
- Dataset: simulação vSomeIP 9 ECUs (Docker). Features JÁ extraídas nos .npy (X_train/test
  7.116.674 × 12; ~12% ataque). Sem código nem modelo publicados.

## Reprodução — SUCESSO
Hiperparâmetros Tabela 2: binary:logistic, 1000 árvores, lr 0.05, max_depth 6, subsample 0.8,
colsample 0.8, lambda 1.0. Treino XGBoost direto nos .npy (tree_method='hist').

Resultado (reproduz o artigo): **ROC-AUC 0,987 (≈0,99) · PR-AUC 0,931 (≈0,93) · F1 ponderado
0,966 (≈0,97)**. 

**Nuance crucial:** o "F1 0,97" do artigo é o **F1 PONDERADO** (dominado pela classe Normal).
O F1 da **classe ataque** isolada é só **0,84** (recall 0,74) — ~26% dos ataques se confundem
com normal (provável MITM/injeção com payload/timing válidos). CTGAN do artigo mira essa
fronteira; usamos --scale-pos-weight (efeito modesto). Capacidade discriminativa (AUC) reproduz
integral; detecção da minoria é o limite real.

## Repo (C:/Mestrado/KIM-repro)
- src/data.py (carga .npy, env KIM_DATA), src/train.py (XGBoost), src/evaluate.py
  (ROC-AUC/PR-AUC/F1 ponderado+ataque/limiar). docs/analise-artigo.md. README com resultados.
- Rodar: `python -m src.train` (1000 árvores) ou `--n-trees 200` (rápido) / `--scale-pos-weight`.
- CTGAN e extração de features do pcap NÃO implementados (usa-se os .npy oficiais).

## Posição no estudo comparativo
É o contraponto não-DL + único que descarta o header. Enriquece o estudo em
C:/Mestrado/Estudo-Comparativo/. Relacionar com Alkhatib/Luo/SISSA (todos DL).
