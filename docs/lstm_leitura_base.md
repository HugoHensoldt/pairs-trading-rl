## Leitura dos resultados (análise manual, não gerada pelo script)

Números vindos das tabelas e figuras deste relatório (5 folds, teste = 72 pregões, 416–488) e da comparação da seção 6.

**1. O modelo aprendeu o efeito do sweep, não a oportunidade.** O AUC de 0,59 (compra) e 0,62 (venda) é real e estável entre folds, mas as três referências abaixo mostram de onde ele vem:

- Um score que só conhece a combinação (sigma, Re, Ri), sem olhar o mercado, tem AUC **0,634 (compra) e 0,640 (venda)**, igual ou maior que o do modelo.
- Dentro de cada combinação o AUC do modelo é **0,491 (compra) e 0,510 (venda)**, isto é, acaso. No fold 5 as 27 combinações ficam entre 0,46 e 0,53 (compra) e entre 0,50 e 0,56 (venda); nas figuras de AUC por combinação (média dos folds) nenhuma passa de 0,55.
- No fold 5, a probabilidade média prevista por combinação tem correlação **0,998 (compra) e 0,995 (venda)** com a taxa de lucro histórica da combinação. Trocar cada previsão pela média da sua combinação dá AUC 0,633 (compra) e 0,640 (venda), contra 0,601 e 0,631 das previsões reais: na compra, a variação dentro da combinação só acrescenta ruído.

**2. Por que isso acontece (hipótese forte, ainda não testada).** No tick da entrada, `SL = Ri·|Saída−Entrada|` e `SG = Re·|Saída−Entrada|` estão nas features (o `SL` já fazia parte da Tabela 1; o `SG`, o take profit, foi acrescentado nesta versão). A razão SL/SG é exatamente Ri/Re, e a taxa de lucro depende sobretudo de Ri (35–48% com Ri = 0,5, 51–64% com Ri = 1,0 e 60–72% com Ri = 1,5, seção 2). A rede pode ler a combinação nessas duas colunas e devolver a taxa base dela. O teste direto é retreinar sem `SL`/`SG` (ou só com quantidades que não revelem Re/Ri) e comparar o AUC intra-combinação.

**3. O treino para cedo.** Em 9 dos 10 treinos a melhor época é a 1 ou a 2, e depois a `val_loss` só sobe enquanto a de treino cai (fig. 3): a rede sobreajusta quase de imediato. Isso é coerente com o item 1, já que a taxa por combinação se aprende em poucas passadas. A queda da loss de treino não é comparável em nível com a de validação por causa do `class_weight`, mas a divergência de forma é clara.

**4. Nenhuma variação de experimento se distingue do ruído.** Nas 27 combinações comuns, a AUC de teste (compra/venda) foi: `base` 0,591/0,621; `seed2` 0,603/0,623; `net_big` (LSTM 128 + dropout 0,2) 0,603/0,618; `grid_wide` (120 combinações) 0,597/0,595. Só trocar a semente já move a AUC por fold em média 0,014 (compra, máx 0,043) e 0,010 (venda, máx 0,017), do tamanho das diferenças entre experimentos. Rede maior e grade mais densa não ajudaram de forma mensurável; o AUC intra-combinação continua entre 0,49 e 0,51 em todos.

**5. O lift no topo é indício fraco.** Aceitando só as ~1–10% oportunidades de maior probabilidade, a taxa de lucro fica entre 0,60 e 0,78, mas a linha só-combo já está em ~0,65–0,69 (compra) e ~0,70–0,72 (venda) nessa região. Os folds ficam acima da linha em alguns casos e abaixo em outros, e o topo de 1% tem ~200 amostras por fold. Não dá para dizer que há informação além do sweep.

**6. Próximos passos (hipóteses, a discutir).**
- Ablação sem `SL`/`SG`, com o AUC intra-combinação como métrica principal (e o AUC global só como referência).
- Controle "só combinação": um modelo que recebe apenas a combinação, para ver quanto o AUC global melhora sem nenhuma leitura do mercado.
- Para medir utilidade operacional é preciso o **lucro em pontos** de cada oportunidade (hoje só o sinal do lucro é gravado) e calcular o P/L, com custos, de operar só as oportunidades aceitas pelo modelo. Isso exige regerar as amostras guardando o lucro por trade.
