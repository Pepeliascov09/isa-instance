# Caracterização de dados em nível de instância — resumo técnico

Iniciação Científica (ITA). Ambiente: Python 3.11.9, pyhard 2.2.4 (pandas 1.5.3,
numpy 1.23.5), openml 0.15.1, panel 0.14.4/bokeh 2.4.3. Datasets OpenML: iris (61),
diabetes (37), blood-transfusion-service-center (1464), hill-valley (1479).
Todos os números abaixo vêm dos CSVs em `resultados/` ou da saída dos scripts citados.

## 1. Pipeline implementado

OpenML → frame numérico → medidas por instância → desempenho out-of-fold →
tabela unificada → projeção 2D → interface interativa.

| Etapa | Módulo / função |
|---|---|
| Carga OpenML | `isaspace/intake.py` — `load_openml_dataset` |
| Conversão para o formato PyHard | `isaspace/intake.py` — `to_pyhard_frame` |
| Medidas por instância (19) | `isaspace/measures.py` — `instance_measures` (envolve `pyhard.measures.ClassificationMeasures`) |
| Desempenho do portfólio | `isaspace/performance.py` — `algo_performance` (kNN, árvore, NB, regressão logística, SVM RBF, random forest; StratifiedKFold k=5, `StandardScaler` dentro de `Pipeline`, seed 42) |
| Tabela unificada por instância | `isaspace/pipeline.py` — `build_instance_table` (fixa `PYHARD_SEED`; colunas `feature_*`, `algo_*`, `proba_*`, `class`, `n_wrong`, `ih`) |
| Projeção 2D | `isaspace/projection.py` — `instance_space` (winsorização p1/p99 + z-score + PCA) |
| Footprints aproximadas | `isaspace/footprint.py` — `grid_footprints` |
| Interface | `isaspace/app.py` (Panel/Bokeh; scatter ligado a painel de detalhes por instância) |

Scripts de experimento: `run_demo.py`, `run_contraste.py`, `run_hard.py`,
`run_multi.py`, `run_diag.py`, `run_perf.py`, `run_table.py`, `run_space.py`,
`run_footprint.py`, `run_transfer.py`, `run_transfer_geom.py`.

## 2. Resultados

**Medida geométrica prevê falha do portfólio.** Spearman entre `feature_kDN` e
`n_wrong` (nº de algoritmos que erram a instância): 0.562 no iris e 0.761 no
diabetes (`run_perf.py`). O rho do iris é limitado por falta de variância no
alvo: 93.3% das instâncias têm `n_wrong = 0`, o que gera empates em massa.

**Transferência entre datasets** (`run_transfer.py`): RandomForestRegressor
treinado em três datasets prevendo `n_wrong` no quarto, com medidas brutas (i)
e z-scoreadas por dataset (ii); baseline = prever a média de `n_wrong` do treino.

| dataset de teste | rho (i) | MAE (i) | rho (ii) | MAE (ii) | MAE baseline |
|---|---|---|---|---|---|
| iris | 0.439 | 0.468 | 0.430 | 0.659 | 2.040 |
| diabetes | 0.825 | 0.618 | 0.809 | 0.601 | 1.945 |
| blood-transfusion | 0.814 | 0.622 | 0.811 | 0.566 | 2.042 |
| hill-valley | 0.512 | 1.246 | 0.713 | 1.298 | 1.734 |

Rho médio: 0.648 (bruto) vs 0.691 (z-score). O ganho do z-score concentra-se no
hill-valley (0.512 → 0.713), o dataset com escalas mais deslocadas, o que apoia a
hipótese de que medidas normalizadas por tamanho de classe não são comparáveis
entre datasets sem padronização.

**Origem das medidas** (`run_transfer_geom.py`): as 18 medidas foram classificadas
por inspeção do código do pyhard em 13 geométricas (distâncias, MST, local sets,
contagens de rótulo) e 5 derivadas de modelo (CL/CLD do Naive Bayes calibrado;
DS/DCP/TD_U de árvores de decisão). Leave-one-dataset-out com z-score:

| dataset de teste | rho todas | MAE todas | rho geom. | MAE geom. | rho modelo | MAE modelo | MAE baseline |
|---|---|---|---|---|---|---|---|
| iris | 0.430 | 0.659 | 0.405 | 1.702 | 0.442 | 0.528 | 2.040 |
| diabetes | 0.809 | 0.601 | 0.646 | 1.543 | 0.745 | 0.761 | 1.945 |
| blood-transfusion | 0.811 | 0.566 | 0.644 | 1.723 | 0.780 | 0.680 | 2.042 |
| hill-valley | 0.713 | 1.298 | 0.388 | 1.825 | 0.587 | 1.416 | 1.734 |

Rho médio: todas 0.691; geométricas 0.521; derivadas de modelo 0.639.
Observações: (a) 5 medidas derivadas de modelo superam 13 geométricas, o que
descarta explicação por número de preditoras — há sobreposição de família de
modelo entre as medidas e o portfólio (que contém um GaussianNB e uma árvore);
(b) na condição geométrica, rho e MAE divergem: o ranqueamento transfere
(rho 0.39–0.65) mas a calibração não — no hill-valley o MAE geométrico (1.825)
é pior que o baseline (1.734); (c) dentro do grupo geométrico, o kDN concentra
0.486 da importância total, mais que o dobro da soma das duas seguintes.

**Bimodalidade da dificuldade** (`run_table.py`): a distribuição de `n_wrong` é
bimodal em todos os datasets — a maioria das instâncias não engana nenhum
algoritmo e existe um núcleo que engana todos os seis.

| dataset | n_wrong = 0 | n_wrong = 6 |
|---|---|---|
| iris | 93.3% | 2.0% |
| diabetes | 51.0% | 9.1% |
| blood-transfusion | 59.2% | 12.3% |
| hill-valley | 11.5% | 6.4% |

**Estrutura espacial** (`run_space.py`): mesmo com a projeção PCA calculada só
sobre as medidas (sem acesso ao desempenho), o `ih` organiza-se no mapa: no iris
as instâncias difíceis formam um grupo destacado do corpo principal; no diabetes
há gradiente contínuo de dificuldade ao longo de z1. O PC1 tem loadings positivos
de magnitude comparável em quase todas as medidas, comportando-se como um fator
geral de dificuldade (variância explicada em 2D: 65.6% iris, 47.5% diabetes).

## 3. Limitações de método encontradas

- **Saturação do LSC.** O pyhard normaliza |LS| pelo tamanho da classe. Nos
  datasets maiores o |LS| mediano é 1–4 pontos contra classes de 178–606
  (`run_diag.py`), comprimindo o LSC mediano para ≥ 0.992. Não é só escala:
  mesmo após z-score, a importância do LSC na transferência é 0.009 — a
  informação foi destruída pela saturação.
- **Redundância LSC × N2.** |rho de Spearman| > 0.7 em 3 dos 4 datasets
  (0.83 iris, 0.85 diabetes, 0.85 hill-valley; 0.39 blood-transfusion).
  Agregações por média simples somam medidas parcialmente redundantes.
- **Colapso das correlações no blood-transfusion.** Entre kDN/N1/N2/LSC, a maior
  correlação lá é 0.56 (`run_diag.py`) — com 4 atributos discretos, os empates
  de distância de Gower degradam as vizinhanças.
- **Footprint dependente da resolução; empates dominam.** Na grade 50×50 do
  iris, 17 das 17 células têm empate entre os 6 algoritmos; no diabetes, 56 de
  60. A área da footprint muda com a resolução (svm_rbf: 0.800 em 50×50 vs
  0.614 em 15×15; `resultados/footprint_*.csv` registra as duas). Sem teste de
  significância, "melhor algoritmo local" é quase sempre indecidível aqui.
- **TD_P indefinido com poda fixa.** Com `ccp_alpha = 0.01`, a árvore podada do
  hill-valley colapsa na raiz e TD_P vira NaN (divisão 0/0 em
  `pyhard/measures.py:332`); a medida foi descartada da transferência.
- **Conversão de dados.** `to_pyhard_frame` codifica categóricas como códigos
  inteiros (ordem artificial onde não existe) e imputa faltantes pela mediana —
  decisões que afetam as distâncias de Gower a jusante.
- **Projeção é PCA, não PILOT.** A projeção atual não otimiza a relação entre
  coordenadas, medidas e desempenho, como o PILOT do ISA faz; o gradiente
  observado é evidência favorável, mas a projeção não é a canônica.
- **Medidas in-sample vs alvo out-of-fold.** Todas as medidas do pyhard são
  calculadas sobre o dataset inteiro, enquanto `n_wrong` é out-of-fold. No caso
  de CL/CLD é vazamento direto: o Naive Bayes calibrado é treinado incluindo a
  própria instância que descreve. Parte da vantagem das medidas derivadas de
  modelo na transferência (0.639 vs 0.521) pode vir desse vazamento, além da
  sobreposição de família de modelo.

## 4. Implicação: requisito de formato

As limitações acima têm uma causa comum: um valor de medida por instância não é
interpretável sozinho. O LSC de 0.99 significa coisas diferentes em classes de 50
ou de 606; a área de footprint de 0.80 só vale para uma tripla
(resolução, k, limiar); o TD_P depende de um `ccp_alpha`; o kDN, de um k e de uma
métrica de distância; CL/CLD, de um protocolo de treino (in-sample ou não); e a
comparabilidade entre datasets exige saber qual padronização foi aplicada.
Isso aponta para um requisito de formato de armazenamento: valores em nível de
instância devem ser gravados junto com (i) os parâmetros de cálculo de cada
medida, (ii) a semente e as versões de biblioteca, e (iii) metadados de
comparabilidade entre datasets (tamanhos de classe, normalizações, escalas).
Os CSVs atuais em `resultados/` guardam apenas os valores — o restante está
implícito nos scripts, o que não sobrevive à circulação dos arquivos.

## 5. Próximos passos

- Recalcular as medidas em protocolo out-of-fold (ao menos CL/CLD e as de
  árvore), eliminando o vazamento de rótulo, e refazer a comparação
  geométricas × derivadas de modelo.
- Substituir o PCA pelo PILOT (projeção que usa medidas e desempenho) e a grade
  de footprint por fronteiras contínuas com teste de significância (TRACE).
- Definir e prototipar o formato de armazenamento do item 4.
- Ampliar o conjunto de datasets para dar suporte estatístico à transferência
  (quatro datasets não sustentam conclusão geral).
- Inspecionar qualitativamente as instâncias com `n_wrong = 6` (candidatas a
  ruído de rótulo, sobretudo no blood-transfusion, onde são 12.3%).
- Reavaliar a codificação de categóricas e a imputação em `to_pyhard_frame`.
