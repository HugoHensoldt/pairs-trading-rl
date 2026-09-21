# Relatório LSTM — experimento `lstm_grid_wide`

> Gerado por `src/report_lstm.py` a partir dos artefatos gravados pelo pipeline (`state.json`, `metrics.json`, `predictions.npz`). Tudo aqui é medido; nenhuma interpretação é gerada automaticamente — a leitura dos resultados fica a cargo de quem analisa.

## 0. Resumo

- **Compra**: AUC no teste = **0.579** em média entre 5 fold(s) (desvio 0.014; mín 0.561, máx 0.598); 5 de 5 folds acima de 0,5. Último fold: 0.598 [0.586; 0.609] (IC95% por bootstrap sobre os pregões). Acurácia no teste menos a do classificador majoritário: +0.044 (média). Referências: AUC de um score que só conhece a combinação (sigma, Re, Ri) = 0.617; AUC do modelo apenas dentro de cada combinação = 0.510.
- **Venda**: AUC no teste = **0.572** em média entre 5 fold(s) (desvio 0.010; mín 0.558, máx 0.579); 5 de 5 folds acima de 0,5. Último fold: 0.578 [0.556; 0.596] (IC95% por bootstrap sobre os pregões). Acurácia no teste menos a do classificador majoritário: +0.032 (média). Referências: AUC de um score que só conhece a combinação (sigma, Re, Ri) = 0.622; AUC do modelo apenas dentro de cada combinação = 0.499.

Uma AUC de 0,5 é o acaso; o IC95% que contém 0,5 significa que, com esses dados, não dá para distinguir o modelo do acaso naquele fold. `AUC só-combo` usa como score a taxa de lucro histórica da combinação (sem olhar o mercado): é o que se ganha só por saber quais parâmetros geraram a oportunidade. `AUC intra-combo` compara apenas oportunidades da mesma combinação, então o efeito do sweep é removido.

---

## 1. Configuração e execução

| Item | Valor |
|---|---|
| Objetivo | classificar cada oportunidade (entrada da heurística) como Lucro (1) ou Prejuízo (0); label = sinal do lucro realizado da negociação (SG se atinge o alvo, −SL se atinge o stop, resultado a mercado se fechada no fim do pregão) |
| Sweep (sigma × Re × Ri) | 120 combinações: sigma [1.0, 1.2, 1.4, 1.6, 1.8, 2.0], Re [0.5, 0.6, 0.75, 0.9], Ri [0.5, 0.75, 1.0, 1.25, 1.5] |
| Features (por tick) | 14: bid, ask, Wbjusto, Wajusto, volume, bvolume, SL, spread, volatilidade, SG, day_sin, day_cos, time_sin, time_cos |
| Janela | 120 ticks anteriores à entrada |
| Rede | 2× LSTM(50) + Dense(1, sigmoid) |
| Treino | Adam, binary_crossentropy, batch 256, até 50 épocas, early stopping (paciência 8, melhor val_loss); class_weight |
| Validação | TimeSeriesSplit expanding window sobre os pregões (split por dia); scaler e balanceamento ajustados só no treino do fold |

**Janelas** (pregões, com o nº de dias entre parênteses; o teste é o mesmo em todos os folds):

| fold | treino | validação | teste |
|---|---|---|---|
| 1 | 1–69 (69) | 70–138 (69) | 416–488 (72) |
| 2 | 1–138 (138) | 139–207 (69) | 416–488 (72) |
| 3 | 1–207 (207) | 208–276 (69) | 416–488 (72) |
| 4 | 1–276 (276) | 277–346 (69) | 416–488 (72) |
| 5 | 1–346 (345) | 347–415 (69) | 416–488 (72) |

---

## 2. Balanceamento dos labels por combinação

Taxa de lucro de cada combinação (pregões de desenvolvimento). Laranja = maioria de prejuízos, azul = maioria de lucros; o cinza é 50%.

![Balanceamento buy](img/lstm_lstm_grid_wide_fig1_balanco_buy.png)

![Balanceamento sell](img/lstm_lstm_grid_wide_fig2_balanco_sell.png)

| lado | n | lucros | prejuizos | fechados_forcado | taxa de lucro |
|---|---|---|---|---|---|
| Compra | 853282 | 458996 | 394286 | 17852 | 0.538 |
| Venda | 832599 | 447208 | 385391 | 21085 | 0.537 |

---

## 3. Curvas de treino e validação

Linha tracejada = melhor época (menor `val_loss`, a que o early stopping restaura).

![Loss](img/lstm_lstm_grid_wide_fig3_curva_loss.png)

> A loss de **treino** do Keras já incorpora o `class_weight` (ponderada), enquanto a de validação não; por isso as duas não são diretamente comparáveis em nível — o que importa é a forma (quando a validação para de melhorar e sobe).

![Acurácia](img/lstm_lstm_grid_wide_fig4_curva_acuracia.png)

---

## 4. Resultados: validação e teste

| lado | fold | épocas | melhor época | n treino | n val | n teste | AUC val | AUC teste | IC95% teste | AUC só-combo | AUC intra-combo | acc teste | acc majoritária | bal_acc teste | prec teste | rec teste | min treinando |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Compra | 1 | 9 | 1 | 122458 | 166680 | 100283 | 0.525 | 0.586 | [0.565; 0.607] | 0.617 | 0.536 | 0.565 | 0.514 | 0.564 | 0.576 | 0.586 | 1.602 |
| Compra | 2 | 9 | 1 | 289138 | 126134 | 100283 | 0.566 | 0.561 | [0.542; 0.580] | 0.617 | 0.497 | 0.546 | 0.514 | 0.547 | 0.566 | 0.506 | 2.868 |
| Compra | 3 | 9 | 1 | 415272 | 138305 | 100283 | 0.582 | 0.574 | [0.556; 0.591] | 0.617 | 0.503 | 0.554 | 0.514 | 0.555 | 0.573 | 0.522 | 3.938 |
| Compra | 4 | 9 | 1 | 553577 | 170283 | 100283 | 0.596 | 0.577 | [0.559; 0.594] | 0.617 | 0.510 | 0.555 | 0.514 | 0.557 | 0.577 | 0.509 | 5.017 |
| Compra | 5 | 9 | 1 | 723860 | 129422 | 100283 | 0.625 | 0.598 | [0.586; 0.609] | 0.617 | 0.502 | 0.574 | 0.514 | 0.574 | 0.590 | 0.564 | 7.008 |
| Venda | 1 | 9 | 1 | 116435 | 155658 | 90127 | 0.557 | 0.579 | [0.560; 0.596] | 0.622 | 0.520 | 0.559 | 0.521 | 0.561 | 0.587 | 0.517 | 1.547 |
| Venda | 2 | 9 | 1 | 272093 | 130471 | 90127 | 0.589 | 0.564 | [0.543; 0.584] | 0.622 | 0.493 | 0.550 | 0.521 | 0.549 | 0.568 | 0.569 | 2.823 |
| Venda | 3 | 9 | 1 | 402564 | 146967 | 90127 | 0.581 | 0.579 | [0.557; 0.600] | 0.622 | 0.506 | 0.559 | 0.521 | 0.559 | 0.581 | 0.551 | 3.872 |
| Venda | 4 | 9 | 1 | 549531 | 155467 | 90127 | 0.593 | 0.558 | [0.534; 0.576] | 0.622 | 0.481 | 0.541 | 0.521 | 0.542 | 0.565 | 0.518 | 4.923 |
| Venda | 5 | 9 | 1 | 704998 | 127601 | 90127 | 0.612 | 0.578 | [0.556; 0.596] | 0.622 | 0.494 | 0.556 | 0.521 | 0.559 | 0.589 | 0.487 | 6.963 |

`acc majoritária` é a acurácia de um classificador que sempre prevê a classe mais comum no teste — o piso que a acurácia do modelo precisa superar. `IC95%` = bootstrap sobre os pregões do teste.

![AUC por fold](img/lstm_lstm_grid_wide_fig5_auc_por_fold.png)

### Curva ROC

![ROC por fold](img/lstm_lstm_grid_wide_fig6_roc_por_fold.png)

![ROC treino/val/teste](img/lstm_lstm_grid_wide_fig7_roc_treino_val_teste.png)

A distância entre a curva de treino e as de validação/teste (último fold) mede o sobreajuste.

### Matriz de confusão (teste, limiar 0,5)

Cada célula: contagem e % da linha (classe real).

![Matriz de confusão](img/lstm_lstm_grid_wide_fig8_matriz_confusao_teste.png)

### Utilidade para operar: aceitar só as melhores oportunidades

Ordena as oportunidades do teste pela probabilidade prevista de lucro e mostra a taxa de lucro das aceitas conforme se aceita mais ou menos delas. Se o modelo discrimina, a curva começa acima da taxa base e decai até ela. A linha tracejada laranja é a referência **só-combo**: aceitar as oportunidades na ordem da taxa de lucro histórica de cada combinação (sigma, Re, Ri), sem olhar o mercado. Só o que fica acima dela é informação além do sweep.

![Lift](img/lstm_lstm_grid_wide_fig9_lift_teste.png)

### Calibração

![Calibração](img/lstm_lstm_grid_wide_fig10_calibracao_teste.png)

> Com `class_weight` as probabilidades não são calibradas para a taxa real; o que vale é o ranking (AUC/lift), não o limiar 0,5.

---

## 5. Onde o modelo discrimina: AUC por combinação (sigma, Re, Ri)

AUC no teste de cada combinação (média entre folds). Azul > 0,5; laranja < 0,5; células com menos de uma centena de amostras são ruidosas.

![AUC por combinação buy](img/lstm_lstm_grid_wide_fig11_auc_combo_buy.png)

![AUC por combinação sell](img/lstm_lstm_grid_wide_fig12_auc_combo_sell.png)

---

## 7. Limitações e ressalvas

- **É classificação, não P/L.** O label vem da regra fixa SG/SL da heurística, sem custos de execução (meio-spread, taxas, atraso). Um bom AUC aqui não implica lucro operável; o passo seguinte seria medir o P/L (com custos) de operar só as oportunidades aceitas pelo modelo.
- **Amostras correlacionadas.** Combinações diferentes da grade no mesmo pregão compartilham quase as mesmas entradas, e janelas de 120 ticks de entradas próximas se sobrepõem; o número efetivo de amostras independentes é bem menor que `n`. Por isso os ICs são por pregão.
- **Um único período de teste** (os últimos pregões, fixos), e poucos folds de validação: a variação entre folds mostra a instabilidade temporal, mas não substitui mais dados.
- **Balanceamento por construção.** A taxa de lucro depende do sweep (sobretudo de Ri); um modelo pode aprender a taxa por combinação a partir das features SL/SG. A seção 5 mostra se há discriminação dentro de cada combinação.

## Apêndice: reprodução

```
python src/report_lstm.py --run-dir sdumont_backup_lstm/lstm_runs --tag lstm_grid_wide
```
