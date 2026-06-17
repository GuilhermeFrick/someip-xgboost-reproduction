# Resultados — Reprodução do KIM (XGBoost)

Treino: `python -m src.train --n-trees 1000 --scale-pos-weight` sobre os `.npy` oficiais
(7.116.674 amostras de treino × 12 features comportamentais; ~12% ataque).

## Métricas — reproduz o artigo ✅

| Métrica | Reprodução | Artigo |
|---|---:|---:|
| ROC-AUC | **0,9874** | 0,99 |
| PR-AUC | **0,9313** | 0,93 |
| **F1 (ponderado)** | **0,9661** | **0,97** |
| Accuracy | 0,9682 | — |

Matriz de confusão (linha = verdadeiro; [normal, ataque]):
```
[6.276.974,     8.540]   # normal
[  217.910,   613.249]   # ataque
```

## ⚠️ O "F1 = 0,97" do artigo é PONDERADO, não da classe ataque
Decompondo o F1 por classe:

| Classe | Precision | Recall | F1 |
|--------|----------:|-------:|----:|
| Normal | 0,966 | 0,999 | **0,982** |
| **Ataque** | 0,986 | **0,738** | **0,844** |
| F1 macro | | | 0,913 |
| **F1 ponderado** | | | **0,966** ← *o "0,97" do paper* |

O F1 ponderado (0,966) é a média ponderada pelas classes, **dominada pela classe Normal**
(majoritária). Por isso ele fica alto mesmo com o ataque em 0,844.

## Isso funciona para deploy em ambiente real? (análise crítica)

**Resposta curta:** o F1 ponderado (0,97) é **mais a título de pesquisa** do que indicador de
prontidão para produção. Para um IDS real, ele **superestima** a capacidade de detecção.

**Por quê:**
- Num dataset com ~88% de tráfego normal, **acertar o normal é "fácil" e infla** qualquer
  métrica ponderada/accuracy. O F1 ponderado de 0,97 reflete sobretudo o acerto do normal.
- O que importa para um IDS é detectar **ataques** (a minoria). A métrica relevante é a da
  **classe ataque**: F1 0,84, **recall 0,74**, precision 0,99.
- **Recall 0,74 ⇒ ~26% dos ataques passam despercebidos** (217.910 falsos negativos). Para
  segurança, deixar passar 1 em cada 4 ataques é uma lacuna séria.
- **Precision 0,99** é o lado bom: pouquíssimos falsos alarmes → baixa fadiga de alerta,
  importante para aceitação operacional.

**Métricas que de fato importam no deploy (e que o paper minimiza):**
1. **Recall por tipo de ataque** — provavelmente DoS/fuzzing são detectados quase 100% (rajada,
   payload aleatório = fáceis), enquanto **MITM/injeção** (replay byte-idêntico, timing válido)
   são o "26% difícil". Isso é uma **lacuna conhecida** a documentar.
2. **PR-AUC (0,93)** — resumo mais honesto que ROC-AUC sob desbalanceamento (ROC-AUC 0,99 é
   otimista demais aqui).
3. **Trade-off de limiar** — dá para aumentar o recall baixando o limiar, mas ao custo de
   precision; o PR-AUC 0,93 **limita** quão bom esse trade-off pode ser.

**Conclusão de deploy:** como **filtro de primeira linha com baixo falso alarme** (precision
0,99) dentro de uma defesa em profundidade, é útil. Como **IDS autônomo**, recall 0,74 é
arriscado — exigiria (i) o CTGAN do paper ou outra técnica para reforçar a minoria, (ii)
avaliação por tipo de ataque, e (iii) validação em **tráfego real** (não simulado).

> **Achado para a dissertação:** métricas globais (F1 ponderado, accuracy, ROC-AUC) podem
> **mascarar** a prontidão real de um IDS em dados desbalanceados. A leitura honesta exige
> **recall/F1 da classe ataque, PR-AUC e desempenho por tipo de ataque**. Vale como crítica
> metodológica transversal aos quatro trabalhos.
