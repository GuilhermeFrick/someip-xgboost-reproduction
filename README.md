# Reprodução — Kim et al. (2026), XGBoost Anomaly Detection para SOME/IP

Reprodução organizada de:

> T. Kim, H. Park, I. You, B. I. Kwak, *XGBoost-Based Anomaly Detection Framework for SOME/IP
> in In-Vehicle Networks*, MDPI Systems, 14(2):196, 2026.

O dataset oficial publica os **vetores de features já extraídos** (`X_train/X_test.npy`,
12 features comportamentais por pacote, rótulo binário), mas **não inclui código nem modelo
treinado**. Este repositório fornece o código de treino/avaliação faltante.

## Diferencial do trabalho
- **XGBoost** (gradient boosting), não deep learning.
- **Features comportamentais** (intervalo de tempo, log-likelihood/entropia de payload,
  distância de Hamming, mudança de tamanho) — **deliberadamente sem campos de header**,
  para evitar overfitting em identificadores específicos.
- Classificação **binária** (normal vs. ataque), por pacote.
- Ataques: **DoS, Fuzzy, MITM** (single/multi-attacker), de simulação vSomeIP com 9 ECUs.

Ver [docs/analise-artigo.md](docs/analise-artigo.md) para a análise completa.

## Estrutura
```
KIM-repro/
├── README.md
├── requirements.txt
├── docs/analise-artigo.md
└── src/
    ├── data.py        # carga dos .npy (caminho via env KIM_DATA)
    ├── train.py       # treina XGBoost (Tabela 2) + avalia
    └── evaluate.py    # ROC-AUC, PR-AUC, F1, matriz de confusão, seleção de limiar
```

## Dados
O dataset (features já extraídas) vem **incluído no repositório** em `data/`, comprimido em
`.npz` (~122 MB no total) e versionado via **Git LFS**. Após clonar:
```bash
git lfs install && git lfs pull     # baixa os .npz
```
O loader (`src/data.py`) lê os `.npz` automaticamente. Para usar os `.npy` originais
(não comprimidos), aponte `KIM_DATA` para a pasta do dataset:
```bash
export KIM_DATA=/caminho/para/os/npy
```
> Os `.npy` originais (~760 MB) e os `.pcap` brutos (~2,2 GB) **não** são versionados.

## Como rodar
```bash
pip install -r requirements.txt

# Treino + avaliação (1000 árvores, Tabela 2)
python -m src.train

# Versão rápida p/ validar
python -m src.train --n-trees 200

# Alternativa ao CTGAN p/ o desbalanceamento (pesa a classe minoritária)
python -m src.train --scale-pos-weight
```

## Hiperparâmetros (Tabela 2 do artigo)
`binary:logistic` · 1000 árvores · lr 0.05 · max_depth 6 · subsample 0.8 ·
colsample_bytree 0.8 · L2 (lambda) 1.0 · gamma 0 · min_child_weight 1.

## Metas do artigo
**PR-AUC 0.93 · ROC-AUC 0.99 · F1 0.97** (dataset desbalanceado).

## Resultados da reprodução — reproduz o artigo ✅

| Métrica | Reprodução (1000 árvores) | Artigo |
|---|---:|---:|
| ROC-AUC | **0,9874** | 0,99 |
| PR-AUC | **0,9313** | 0,93 |
| **F1 (ponderado)** | **0,9661** | **0,97** |
| Accuracy | 0,9682 | — |
| F1 (classe ataque) | 0,844 | — |

**Os três números do artigo são reproduzidos** (ROC-AUC, PR-AUC e F1 ponderado).

### Detalhe importante sobre o F1
O artigo reporta **F1 = 0,97**, que é o **F1 ponderado** (média sobre as duas classes,
dominado pela classe Normal, F1≈0,98). Nós obtemos **0,966**. Já o **F1 da classe ataque**
isoladamente é **0,84** (precision 0,99, recall 0,74) — métrica mais estrita, que o artigo não
destaca. Ou seja: a *capacidade discriminativa* (ROC-AUC/PR-AUC) reproduz integralmente; a
detecção da minoria continua limitada (~26% dos ataques se confundem com normal — provavelmente
injeções/MITM com payload e timing válidos). O **CTGAN** do artigo visa justamente reforçar essa
fronteira; aqui usamos `--scale-pos-weight` (efeito modesto).

## Notas
- O artigo usa **CTGAN** para oversampling sintético da classe minoritária; aqui oferecemos
  `--scale-pos-weight` como alternativa simples (o CTGAN poderia ser adicionado depois).
- A extração de features a partir dos `.pcap` não é reproduzida (usa-se os `.npy` oficiais);
  o cálculo das features está descrito em `docs/analise-artigo.md` (Eqs. 1–7 do artigo).
