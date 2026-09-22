## Leitura dos resultados (análise manual, não gerada pelo script)

Campanha v2: sigma 2,2/2,4/2,6 (antes 1,4/1,6/1,8), teste dobrado para os últimos 30% dos pregões (343–488, 146 pregões; antes 15%, 72 pregões), pregões 168/304/336/338/371/463 excluídos por cotação impossível. Números das tabelas e figuras deste relatório (5 folds) e da comparação `lstm_v2_base` × `lstm_v2_seed2`.

**1. Sigma maior não resolveu o problema identificado na campanha anterior.** O AUC de teste é 0,61 (compra) e 0,60 (venda), parecido com o da campanha com sigma 1,4–1,8 (0,59/0,62). E a causa é a mesma:

- Um score que só conhece a combinação (sigma, Re, Ri), sem olhar o mercado, tem AUC **0,642 (compra) e 0,631 (venda)** — igual ou maior que o do modelo.
- Dentro de cada combinação (mesma sigma, Re, Ri, comparando só as oportunidades entre si) o AUC do modelo é **0,513 (compra) e 0,507 (venda)** — acaso. No mapa por combinação (fig. 11–12), nenhuma das 27 passa de 0,54.
- Na curva de lift (fig. 9), aceitar as oportunidades de maior probabilidade segue de perto a linha "só a combinação" (aceitar pela taxa histórica da combinação, sem olhar o mercado); no fold 1 fica abaixo dela.

**2. O balanceamento de labels (pregões de desenvolvimento, 1–342) segue perto do equilíbrio mesmo com sigma maior.** Taxa de equilíbrio = Ri/(Re+Ri); a taxa observada fica em média 1,7 pontos percentuais ABAIXO do equilíbrio (mediana igual). Só 4 das 54 linhas (27 combinações × compra/venda) ficam acima do equilíbrio, e a folga nelas é de no máximo 0,4 pp. Ou seja, a heurística com sigma 2,2–2,6 continua sem edge bruto detectável nesta amostra, antes de qualquer custo.

**3. O ruído entre execuções continua do tamanho das diferenças que importariam.** Trocando só a semente (`lstm_v2_base` × `lstm_v2_seed2`, 27 combinações comuns), a AUC por fold varia em média 0,024 (compra, máx 0,034) e 0,012 (venda, máx 0,030). A diferença observada entre os dois experimentos é 0,011 (compra) e 0,008 (venda) — menor que o ruído entre sementes, portanto não é uma melhora real.

**4. Conclusão preliminar.** As duas mudanças pedidas (sigma maior, teste maior) não alteraram o quadro da campanha anterior: o modelo aprende a taxa de lucro por combinação (sigma, Re, Ri), não a distinguir oportunidades dentro de uma combinação. Isso é consistente com a hipótese já registrada em `docs/lstm_leitura_base.md`: as features `SL` e `SG` revelam Ri/Re, e a rede pode estar devolvendo a taxa base da combinação em vez de ler o mercado.

**5. O que esta campanha NÃO testou.**
- Não temos aqui o P/L em pontos das oportunidades aceitas pela LSTM com a grade sigma 2,2–2,6 (isso exigiria rodar `--stage pl` para essa grade; a análise de P/L anterior foi feita com a grade antiga, sigma 1,4–1,8).
- Não foi feita a ablação sem `SL`/`SG` sugerida na campanha anterior — continua sendo o teste mais direto da hipótese do item 4.
- Sigma maior que 2,6 não foi varrido aqui; a análise `src/analyze_spread_vs_edge.py` (heurística pura, sem LSTM) indicava, num recorte menor de dados, que o edge bruto (antes do spread) crescia até sigma≈3–4 no desenvolvimento mas piorava no teste — sinal de instabilidade, não de tendência confiável.
