# Análise do Artigo — Kim et al. (2026)

**Referência:** T. Kim, H. Park, I. You, B. I. Kwak, *XGBoost-Based Anomaly Detection Framework
for SOME/IP in In-Vehicle Networks*, MDPI Systems, 14(2):196, 2026.

## É ML ou regras?
**Machine learning clássico** — **XGBoost** (gradient boosting de árvores), **não** deep
learning. É a abordagem mais distinta dos outros três trabalhos (Alkhatib/Luo/SISSA usam DL).

## Objetivo de pesquisa
Detectar anomalias em tráfego SOME/IP usando **features comportamentais** (não campos de
header), respondendo a 3 perguntas:
- **RQ1:** features comportamentais (sem header) separam normal/ataque? (visualização, clustering)
- **RQ2:** o XGBoost é robusto em diversos cenários de ataque com baixa taxa de falso alarme?
- **RQ3:** é viável em tempo quase-real em ECU de gateway/domínio?

## Diferencial metodológico — features comportamentais (não-header)
A tese central: **evitar campos de header** (Service/Method ID, portas, etc.) porque são
identificadores específicos de implementação que causam **overfitting** e não distinguem
ataques que mantêm o header válido (ex.: DoS com header legítimo). Em vez disso, usam **12
features comportamentais** agrupadas em 5 classes (Tabela 1):

| Grupo | Features | Captura |
|---|---|---|
| **Intervalo de tempo** | IP time interval | rajadas (DoS), timing anômalo |
| **Verossimilhança de payload** | log-likelihood SOME/IP(-SD) e TCP/UDP | bytes raros (fuzzing/manipulação) |
| **Entropia de payload** | cross-entropy SOME/IP(-SD) e TCP/UDP | conteúdo estatisticamente irregular |
| **Mudança de payload** | distância de Hamming entre payloads consecutivos | mudança abrupta (tamper/injeção MITM) |
| **Mudança de tamanho** | delta de comprimento IP e TCP/UDP | mensagens malformadas/injetadas |

- **Log-likelihood/entropia** usam um modelo de distribuição de bytes por posição aprendido do
  tráfego benigno (com suavização de Laplace).
- Features relacionais (intervalo, mudança) são computadas **dentro do mesmo fluxo** (mesmo
  IP origem/destino e portas).
- O vetor final tem **12 dimensões** por pacote (já fornecido nos `.npy`).

## Ataques cobertos (dataset)
Coletado de uma **simulação vSomeIP com 9 ECUs** (publish-subscribe, containers Docker):
| Cenário | Descrição |
|---|---|
| **Normal** | tráfego benigno |
| **DoS** (Event Notification Flooding) | 3000 notificações o mais rápido possível → congestão |
| **Fuzzy** (SD Offer flooding + ADAS noti aleatórias) | IDs de serviço/evento randomizados, payload não-semântico |
| **MITM single-attacker** | SD "withdraw" (desliga provedor legítimo) + injeção de ADAS forjado |
| **MITM multi-attacker** | relay de evento + SD spoofing + injeção ADAS |

Classificação é **binária** (normal vs ataque), não multi-classe.

## Modelo e treino (Tabela 2)
- **XGBoost**: objetivo `binary:logistic`, **1000 árvores**, lr **0.05**, max_depth **6**,
  subsample **0.8**, colsample_bytree **0.8**, L2 (lambda) **1.0**, gamma 0, min_child_weight 1.
- **CTGAN** (GAN tabular) para **oversampling sintético** da classe minoritária, mitigando o
  forte desbalanceamento (~12% ataque). Embedding 128, 2 camadas de 256, PacGAN(10), 100 épocas.
- Limiar de decisão ajustado no treino.

## Resultados do artigo
- **PR-AUC 0.93 · ROC-AUC 0.99 · F1 0.97** (dataset desbalanceado, "real-world-reflective").
- **0.556 ms/pacote** (extração de features + inferência) → viável em tempo quase-real.

## Dataset (o que está disponível)
- `pcap/` — pcaps brutos (normal + 4 cenários de ataque) + `bening_traffic.csv`.
- **`X_train.npy` (7.116.674, 12)**, `X_test.npy` (7.116.673, 12), `y_train/y_test` binários
  (~6,29 M normal / 831 k ataque por split). **Features já extraídas** → reprodução direta.
- **Não há código** publicado (nem modelo treinado).

## Relação com os outros três trabalhos
| | Alkhatib | Luo | SISSA | **KIM** |
|---|---|---|---|---|
| Abordagem | RNN | regras+GRU | CNN/RNN/LSTM+atenção | **XGBoost** |
| Features | header one-hot | header+payload | one-hot+PMB | **comportamentais (sem header)** |
| Classes | 5 | 3 (payload) | 7 | **2 (binário)** |
| Unidade | sequência (60) | janela (91) | janela (128) | **por pacote** |
| Ataques | processo comm. | fuzzy/spoof/DoS | DDoS/MITM/falha | **DoS/Fuzzy/MITM** |

➡️ KIM é o **contraponto não-DL**: aposta em *feature engineering* comportamental + boosting,
argumentando que features de header causam overfitting. Isso o torna **muito relevante** para
um estudo comparativo — é a única abordagem que **deliberadamente descarta o header**.
