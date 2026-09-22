# Formato da pasta de saída do engine

Contrato entre quem escreve, `isaspace.engine.run_instancespace` (Python 3.12,
`.venv-isa`, instancespace 0.3.0), e quem lê, `isaspace.ui.loader_is.load_is_output`.
Uma pasta corresponde a uma execução sobre um `metadata.csv`. As pastas
versionadas estão em `resultados/is/<nome>/` e são geradas por
`scripts/run_is_all.py`.

A antiga `resultados/isa/<nome>/` (pyispace, lida por `isaspace/ui/loader.py`)
tem outro formato e não segue este contrato.

## Convenções gerais

- **Pasta completa.** `run_info.json` é gravado por último. Pasta sem ele é
  execução incompleta, e o loader a recusa. Numa reexecução, o engine apaga
  apenas os arquivos listados aqui, antes de gravar; recusa pasta que tenha
  esses nomes sem um `run_info.json` dele próprio.
- **CSV.** Vírgula como separador, cabeçalho na primeira linha, UTF-8, ponto
  decimal, sem linhas de comentário. Floats na representação completa do
  pandas, salvo onde há arredondamento indicado.
- **`Row` nos arquivos por instância.** É o **rótulo da instância** (coluna
  `instances` do metadata), não um contador. Deve ser lido como texto
  (`dtype={"Row": str}`): o rótulo `"1"` não é o inteiro 1. Todos os arquivos
  por instância têm as mesmas linhas, na mesma ordem (a do metadata). Com as
  opções padrão, todas as instâncias do metadata aparecem; se
  `selvars.small_scale_flag` ou `selvars.density_flag` forem ligados, é um
  subconjunto (`run_info.n_instancias` < `n_instancias_entrada`).
- **Booleanos** são gravados como `True` / `False`.
- **Campo vazio** significa ausente (NaN, ou `None` para nomes).
- **Nomes** de algoritmos e features vêm sem os prefixos `algo_` / `feature_`.
- **Ordem dos algoritmos:** é a das colunas de `algorithm_raw.csv`, igual a
  `run_info.algoritmos`. Os índices de portfólio referem-se a essa ordem.
- **Arquivos cuja ausência tem significado:** `coordinates_trace.csv`
  (sem jitter) e os `footprint_*.csv` (footprint vazia).
- **Origem de cada arquivo:** *sc* = gravado por `Model.save_to_csv` do
  instancespace; *eng* = gravado pelo engine.

## Índice

| arquivo | origem | uma linha por | presença |
|---|---|---|---|
| `run_info.json` | eng | — | sempre |
| `run_options.json` | eng | — | sempre |
| `metadata.csv` | eng (cópia) | instância de entrada | sempre |
| `annotations.json` | eng (cópia) | — | se existir ao lado do metadata de entrada |
| `degenerate_report.csv` | eng (cópia) | feature descartada antes do engine | se existir ao lado do metadata de entrada |
| `feature_info.csv` | eng (cópia) | feature recebida | se existir ao lado do metadata de entrada |
| `coordinates.csv` | eng | instância | sempre |
| `coordinates_trace.csv` | eng | instância | só com jitter |
| `projection_matrix.csv` | sc | eixo (z_1, z_2) | sempre |
| `pilot_r2.csv` | eng | feature ou algoritmo | sempre |
| `feature_raw.csv`, `feature_process.csv` | sc | instância | sempre |
| `algorithm_raw.csv`, `algorithm_process.csv` | sc | instância | sempre |
| `algorithm_bin.csv` | sc | instância | sempre |
| `good_algos.csv`, `beta_easy.csv`, `portfolio.csv` | sc | instância | sempre |
| `sifted_report.csv` | eng | feature de entrada | sempre |
| `sifted_correlations.csv` | eng | par feature x algoritmo | sempre (pode ter só cabeçalho) |
| `sifted_silhouette.csv` | eng | k testado | sempre (pode ter só cabeçalho) |
| `algorithm_svm.csv`, `portfolio_svm.csv` | sc | instância | sempre |
| `pythia_proba.csv`, `pythia_selection.csv` | eng | instância | sempre |
| `pythia_confusion.csv` | eng | algoritmo | sempre |
| `svm_table.csv` | sc | algoritmo, mais `Oracle` e `Selector` | sempre |
| `bounds.csv`, `bounds_prunned.csv` | sc | vértice da fronteira | sempre |
| `footprint_<algo>_<good\|best>.csv` | sc | vértice | só se não vazia |
| `footprint_space.csv`, `footprint_hard.csv` | eng | vértice | só se não vazia |
| `footprint_performance.csv` | sc | algoritmo | sempre |

---

## Execução

### `run_info.json` (eng)

Objeto JSON.

| chave | tipo | conteúdo |
|---|---|---|
| `gerado_por` | str | `"isaspace.engine.run_instancespace"`; marca a pasta como do engine |
| `gerado_em` | str | data e hora ISO 8601, local, em segundos |
| `instancespace_version`, `python` | str | versões usadas |
| `metadata_entrada` | str | caminho absoluto do metadata lido |
| `n_instancias_entrada`, `n_instancias` | int | linhas do metadata e linhas nos arquivos por instância |
| `n_features_entrada`, `n_features_selecionadas` | int | features do metadata e escolhidas pelo SIFTED |
| `algoritmos` | list[str] | ordem canônica dos algoritmos |
| `tem_source` | bool | se o metadata tinha coluna `source` |
| `tipos_anotacao` | dict | `{"arquivo": "annotations.json" ou null, "declarados": {anotação: tipo}}` |
| `arquivos_auxiliares` | list[str] | quais de `annotations.json`, `degenerate_report.csv` e `feature_info.csv` foram copiados |
| `regra_bom` | str | regra de `algorithm_bin.csv`, por exemplo `"bom = algo_* >= 0.5"` |
| `tempos_s` | dict | segundos por estágio (`PREPROCESSING` … `TRACE`), mais `deteccao_duplicatas`, `build_total` e `escrita` |
| `trace_robustez` | dict | ver abaixo |
| `arquivos_footprint` | dict | `{algo: {"good": arquivo ou null, "best": arquivo ou null}}`; ver TRACE |
| `footprints_especiais` | dict | `{"space": {...}, "hard": {...}}`; ver TRACE |
| `pythia` | dict | contagens de diagnóstico do PYTHIA; ver PYTHIA |
| `arquivos` | list[str] | arquivos desta pasta gravados pelo engine |
| `avisos` | list[str] | inconsistências detectadas pelo engine ao montar os arquivos; vazio é o esperado |
| `avisos_instancespace` | list[str] | mensagens de nível WARNING ou acima do log do instancespace, sem repetição |
| `warnings_python` | list[dict] | `{categoria, mensagem, n}`: warnings Python capturados durante a execução, com contagem |

`trace_robustez` registra a correção de pontos quase coincidentes da projeção
antes do TRACE. O alpha shape do TRACE legado devolve polígono vazio quando há
pontos **distintos** a ~1e-14 um do outro.

- Sempre presentes:
  - `limiar` (1e-6);
  - `pares_quase_duplicados`: todos os pares a menos do limiar;
  - `pares_identicos`: os pares com distância 0; o TRACE já os funde com `np.unique`, então não atrapalham;
  - `pares_distintos_proximos`: pares entre posições distintas;
  - `menor_distancia_distintos_antes`;
  - `regra`;
  - `jitter_aplicado`.
- Quando `jitter_aplicado` é `true`:
  - `jitter_escala`, `jitter_semente` e `jitter_distribuicao`;
  - `posicoes_perturbadas` e `pontos_perturbados`;
  - `rotulos_perturbados`: os `Row` alterados;
  - `deslocamento_maximo`;
  - `menor_distancia_distintos_depois` e `pares_distintos_proximos_depois`;
  - `aplicado_em`.
- `motivo_sem_jitter` aparece quando a correção foi desligada (`fix_near_duplicates=False`).

### `run_options.json` (eng)

É `dataclasses.asdict(InstanceSpaceOptions)`: todas as opções efetivas, agrupadas
(`parallel`, `perf`, `auto`, `bound`, `norm`, `selvars`, `sifted`, `pilot`,
`cloister`, `pythia`, `trace`, `outputs`, `general`, `prelim`), com os nomes de
campo dos dataclasses (`max_perf`, `use_sim`, `purity`…).
`InstanceSpaceOptions.from_dict(json.load(...))` reconstrói as mesmas opções.

O engine altera só estes padrões da biblioteca: `perf.max_perf=true`,
`perf.abs_perf=true`, `perf.epsilon=0.5` e `trace.use_sim=false`. O loader usa
`trace.purity` como limiar de footprint "suspeita".

### `metadata.csv` (eng)

Cópia byte a byte do metadata de entrada. Colunas, com prefixos e nomes
comparados sem diferenciar maiúsculas:

- `instances`: rótulo da instância; vira o `Row` dos demais arquivos;
- `source` (opcional): origem da instância;
- `feature_<f>`: features (numéricas);
- `algo_<a>`: desempenho de cada algoritmo (numérico);
- qualquer outra coluna é **anotação**: o instancespace a ignora, e o loader a
  carrega em `IsResult.instances`, mantendo o nome ou usando `ann_<nome>` se
  colidir com uma coluna derivada.

Tipo das anotações no loader, por ordem de prioridade:

1. `load_is_output(..., annotation_types={coluna: tipo})`: origem `"forcado"`;
   é o que a interface usa para trocar, na sessão, o tipo de uma anotação
   inferida;
2. o declarado em `annotations.json`: origem `"declarado"`;
3. a heurística, só quando não há declaração: origem `"inferido"`. Texto ou
   bool → `"categorica"`; número → `"numerica"`, exceto número inteiro com no
   máximo 2 valores distintos (código binário), que vira `"categorica"`.

A origem de cada tipo fica em `IsResult.annotation_origins`. O CSV não guarda
tipo: o rótulo de texto `"1"` volta como número, com ou sem aspas. Por isso a
declaração existe.

### `annotations.json` (eng, cópia opcional)

Objeto JSON `{anotação: tipo}`, com `tipo` ∈ `"categorica"`, `"numerica"` e
`"numerica_inteira"` (numérica cujos valores são inteiros, como contagens).
As chaves usam o nome da coluna no `metadata.csv`. Nem toda anotação precisa
estar declarada.

O engine valida o arquivo **antes** de rodar e recusa três casos:

- tipo desconhecido;
- chave que não é coluna de anotação (é `instances`, `source`, `feature_*` ou
  `algo_*`, ou não existe);
- tipo numérico com valor não numérico, ou `numerica_inteira` com valor não
  inteiro.

Os tipos declarados vão também para `run_info.tipos_anotacao`.

### `degenerate_report.csv` (eng, cópia opcional)

Medidas descartadas **antes** do engine pelo gerador do metadata (no IC7,
`isaspace.isa.to_isa_metadata`, por variância nula depois do recorte de
outliers e do z-score). Colunas: `feature` (str, sem o prefixo), `var_bruta`
(float), `iqr` (float) e `motivo` (str). Uma linha por medida descartada. Ter
só o cabeçalho significa "verificado, nenhuma caiu". A ausência do arquivo
significa que não há informação sobre descartes anteriores ao engine.

### `feature_info.csv` (eng, cópia opcional)

`feature` (str, sem o prefixo) e `family` (str, livre). Uma linha por feature
recebida, incluindo as degeneradas. A interface mostra a família como coluna
na aba Features, e a ordem das linhas define a ordem dessa tabela. No IC7,
`family` é `model_derived` (CL, CLD, DS, DCP, TD_U, TD_P) ou `geometric`.

A tabela da aba Features (`IsResult.features_table()`) junta
`degenerate_report.csv` (status `dropped_degenerate`) e `sifted_report.csv`,
com `r2_pilot` de `pilot_r2.csv` para as mantidas.

---

## Projeção (PILOT)

### `coordinates.csv` (eng)

`Row`, `z_1`, `z_2` (float). O **z do PILOT, sem correção**. É o que a
interface desenha. O engine regrava este arquivo depois do `save_to_csv`,
que gravaria o z do TRACE.

### `coordinates_trace.csv` (eng, opcional)

Mesmas colunas. O z que o TRACE usou, **só quando `trace_robustez.jitter_aplicado`**.
Difere de `coordinates.csv` apenas nas linhas de `rotulos_perturbados`, e no
máximo por `deslocamento_maximo`. As footprints foram calculadas sobre este z.
O loader exige coerência: o arquivo existe se, e só se, o jitter foi aplicado.
Expõe o conteúdo em `IsResult.coordinates_trace` (ou `None`).

### `projection_matrix.csv` (sc)

`Row` ∈ {`Z_{1}`, `Z_{2}`} (o loader renomeia para `z_1`, `z_2`), mais uma
coluna por feature selecionada. É a matriz A do PILOT, **arredondada a 4 casas**.
Vale `z ≈ A · x`, com `x` a linha de `feature_process.csv`. A precisão completa
está só no `Model`.

### `pilot_r2.csv` (eng)

| coluna | tipo | conteúdo |
|---|---|---|
| `variable` | str | nome da feature ou do algoritmo |
| `kind` | str | `feature` ou `algorithm` |
| `r2` | float | quadrado da correlação entre a variável processada e sua reconstrução a partir de z (`x̂ = z Bᵀ`, pilot.py:478) |

As linhas vêm primeiro com as features selecionadas, na ordem de
`feature_raw.csv`, e depois com os algoritmos, na ordem canônica. O `r2` diz
quanto do plano 2D explica cada variável.

---

## Dados e PRELIM

Todos por instância (`Row`), com uma coluna por feature ou algoritmo.

- **`feature_raw.csv`**: valores de entrada das features **selecionadas pelo
  SIFTED**. As demais continuam em `metadata.csv`.
- **`feature_process.csv`**: as mesmas features depois do PRELIM: recorte de
  outliers em mediana ± `prelim.iqr_multiplier`·IQR (`bound.flag`),
  deslocamento para valores positivos, Box-Cox e z-score (`norm.flag`). Ambos
  exigem `auto.preproc`. Com o padrão, cada coluna tem média 0 e desvio 1.
- **`algorithm_raw.csv`**: o `algo_*` de entrada.
- **`algorithm_process.csv`**: o desempenho que o PILOT e o SIFTED usam.
  - Com `abs_perf`, é o bruto; com desempenho relativo, é `1 − algo/melhor`
    (ou `algo/melhor − 1` sem `max_perf`).
  - Em seguida é deslocado para valores positivos e passa por Box-Cox e
    z-score (`auto.preproc` e `norm.flag`; prelim.py:922).
- **`algorithm_bin.csv`**: bool, "bom" segundo `run_info.regra_bom` (o `y_bin`
  do instancespace).
- **`good_algos.csv`**: `NumGoodAlgos` (int), o número de algoritmos bons na instância.
- **`beta_easy.csv`**: `IsBetaEasy` (bool), igual a `NumGoodAlgos > perf.beta_threshold × n_algoritmos`.
- **`portfolio.csv`**: `Best_Algorithm` (int), índice **1-based** do melhor
  algoritmo por `algo_*` (argmax com `max_perf`, argmin sem). Empates são
  sorteados com `general.seed`. Sempre vale de 1 a n. O loader converte para
  nome em `instances["best_algo"]`.

---

## SIFTED

### `sifted_report.csv` (eng)

Uma linha por feature do metadata, na ordem do metadata.

| coluna | tipo | conteúdo |
|---|---|---|
| `feature` | str | nome |
| `status` | str | `kept`, `dropped_correlation`, `dropped_redundancy`; raramente `dropped_preprocessing` (removida antes do SIFTED) ou `undetermined` (reconstrução inconsistente, também registrada em `run_info.avisos`) |
| `rho` | float | a correlação de maior valor absoluto entre a feature e os algoritmos, com sinal |
| `rho_algo` | str | algoritmo dessa correlação |
| `pval` | float | p-valor dessa mesma correlação |
| `n_algos_sig` | int | algoritmos com \|rho\| ≥ `sifted.rho` e p ≤ `sifted.pval` |
| `cluster` | int ou vazio | cluster (1..k) das features que passaram pela correlação, quando houve clusterização |
| `kept_instead` | str ou vazio | só em `dropped_redundancy`: a feature mantida no mesmo cluster |

Regras, reconstruídas de `Model.sifted` (sifted.py:774-808 e 1056-1078):

- **Passa na correlação** a feature que:
  - é a mais correlacionada com algum algoritmo, **ou**
  - tem \|rho\| ≥ `sifted.rho` com p ≤ `sifted.pval` para algum algoritmo.
- As que não passam ficam como `dropped_correlation`.
- **Clusterização:** se sobram mais de 3 features e mais que `sifted.k`, elas
  são agrupadas em `sifted.k` clusters, e um algoritmo genético mantém
  **exatamente uma por cluster**. As demais ficam como `dropped_redundancy`.
- Se não há clusterização, todas as que passaram na correlação ficam como `kept`.

### `sifted_correlations.csv` (eng)

Formato longo, com a matriz completa: uma linha por par feature x algoritmo,
nas colunas `feature`, `algorithm`, `rho` e `pval` (float). É a correlação de
Pearson entre a feature processada e o desempenho processado
(`algorithm_process.csv`), sobre todas as features que entraram no SIFTED.
Tem só o cabeçalho se o SIFTED não calculou correlações. O loader expõe o
formato longo e as matrizes `IsResult.sifted_rho` e `IsResult.sifted_pval`
(feature x algoritmo).

### `sifted_silhouette.csv` (eng)

| coluna | tipo | conteúdo |
|---|---|---|
| `k` | int | número de clusters testado: 3 .. (features que passaram na correlação) − 1 |
| `silhouette` | float | silhueta média com distância de correlação |
| `used` | bool | `k == sifted.k`; é o k de fato usado, fixo nas opções |
| `best` | bool | o k de maior silhueta; só é sugerido no log, não é usado |

Tem só o cabeçalho quando não houve clusterização.

---

## PYTHIA

O PYTHIA treina um classificador por algoritmo (SVM por padrão,
`pythia.classifier`) que prevê "bom" (`algorithm_bin.csv`) a partir de z. Os
hiperparâmetros são ajustados por validação cruzada estratificada com
`pythia.cv_folds` partes.

- **`algorithm_svm.csv`** (sc): bool, `y_hat`. É a previsão de `predict()` do
  classificador final, **dentro da amostra**. O loader a coloca em `algo_<a>_svm`.
- **`portfolio_svm.csv`** (sc): `Best_Algorithm` (int), igual a `selection0`,
  índice **0-based**, com **-1 = nenhum**. O loader converte para nome em
  `instances["best_algo_svm"]`.

### `pythia_proba.csv` (eng)

`Row`, depois uma coluna `<algo>` por algoritmo, depois uma `<algo>_hat` por
algoritmo, na ordem canônica. Todos os valores são floats em [0, 1] e significam
**P(ruim)**: a probabilidade (`predict_proba`, com escala de Platt no SVM) de o
algoritmo **não** ser bom na instância. P(bom) = 1 − valor.

- **`<algo>` = `pr0_sub`, o padrão:** probabilidade **fora da amostra**, de
  `cross_val_predict` com o classificador ajustado. Cada instância é avaliada
  por um modelo que não a viu. É a estimativa honesta.
- **`<algo>_hat` = `pr0_hat`:** o modelo final, treinado em todas as
  instâncias e avaliado nelas mesmas, **dentro da amostra**. É mais otimista.
- `y_hat` (`algorithm_svm.csv`) pode discordar de `pr0_hat < 0.5`, porque
  `predict` e `predict_proba` do SVC não são equivalentes.
  `run_info.pythia.y_hat_discorda_de_pr0_hat` conta os pares em desacordo.

O engine recusa, antes de rodar, algoritmos cujo nome colida com `<outro>_hat`.
O loader expõe `IsResult.pythia_proba` (pr0_sub) e `IsResult.pythia_proba_hat`
(pr0_hat), ambos com colunas = algoritmos.

### `pythia_confusion.csv` (eng)

`Algorithm` (str), `tn`, `fp`, `fn`, `tp` (int). É a matriz de confusão por
algoritmo: a verdade é "bom" (`algorithm_bin.csv`), e a previsão é a da
validação cruzada (`y_sub`, fora da amostra).

- positivo = bom;
- `tn + fp` = instâncias ruins;
- `fn + tp` = instâncias boas;
- a soma de cada linha é `n_instancias`.

Acurácia, precisão e recall calculados daqui batem com as colunas `CV_model_*`
de `svm_table.csv`. O engine confere isso, porque o instancespace usa outra
ordem de colunas no caminho de avaliação.

### `pythia_selection.csv` (eng)

`Row`, `selection0`, `selection1` (nome de algoritmo; vazio = nenhum).

- **`selection0`:** entre os algoritmos com `y_hat` verdadeiro, o de maior
  precisão de validação cruzada. Fica vazio quando nenhum é previsto bom. É o
  mesmo que `portfolio_svm.csv`.
- **`selection1`:** igual a `selection0`, mas, quando não há recomendação, usa
  o algoritmo com maior fração de instâncias boas. Nunca fica vazio.
- `run_info.pythia` traz as contagens `selection0_nenhum` e
  `selection1_difere_de_selection0`.

### `svm_table.csv` (sc)

`Row` é um algoritmo, `Oracle` ou `Selector`. Valores arredondados a 3 casas;
percentuais a 1 casa; célula vazia onde não se aplica.

| coluna | algoritmo | `Oracle` | `Selector` |
|---|---|---|---|
| `Avg_Perf_all_instances`, `Std_Perf_all_instances` | média e desvio de `algo_*` em todas as instâncias | do melhor desempenho por instância | do algoritmo de `selection1` |
| `Probability_of_good` | fração de instâncias boas | 1 | fração em que o algoritmo de `selection1` é bom |
| `Avg_Perf_selected_instances`, `Std_Perf_selected_instances` | de `algo_*` onde `y_hat` é verdadeiro | vazio | do algoritmo de `selection0` (sem as instâncias "nenhum") |
| `CV_model_accuracy` | acurácia de CV (%) | vazio | vazio |
| `CV_model_precision`, `CV_model_recall` | precisão e recall de CV (%) | vazio | precisão e recall do seletor, definição do MATLAB (pythia.py:2006-2016) |
| `BoxConstraint`, `KernelScale` | hiperparâmetros do SVM | vazio | vazio |

---

## CLOISTER

### `bounds.csv` e `bounds_prunned.csv` (sc)

`Row` (`bnd_pnt_1` … `bnd_pnt_m`), `z_1`, `z_2`. São os vértices, em ordem e
sem repetir o primeiro, de um polígono convexo em z: a fronteira estimada da
região onde podem existir instâncias.

- A fronteira é obtida projetando com A as combinações de mínimo e máximo de
  cada feature processada e tomando o fecho convexo.
- `bounds_prunned.csv` usa só as combinações compatíveis com as correlações
  significativas entre features (`cloister.p_val`, `cloister.c_thres`). Pode
  ser igual a `bounds.csv`.
- Com mais de `cloister.max_features` features (padrão 20), o CLOISTER não
  enumera as combinações: usa o fecho convexo das instâncias projetadas, e os
  dois arquivos saem iguais (cloister.py:190-203).

O loader expõe `IsResult.bounds` e `IsResult.bounds_pruned`, como `Poligono`.

---

## TRACE

### Esquema das footprints

Vale para `footprint_<algo>_good.csv`, `footprint_<algo>_best.csv`,
`footprint_space.csv` e `footprint_hard.csv`.

| coluna | tipo | conteúdo |
|---|---|---|
| `Row` | int | contador de vértices, 1..k; **não** é rótulo de instância |
| `Part` | int | parte do (multi)polígono, 1..p |
| `Ring` | str | `exterior` ou `hole_<j>` (furo j da parte) |
| `Vertex` | int | ordem do vértice no anel, 1..; o primeiro não se repete no fim |
| `z_1`, `z_2` | float | coordenadas |

- Cada `Part` tem um anel `exterior` e zero ou mais furos.
- O loader devolve um `Poligono` por `Part` (`exterior` e `furos`), com área
  pela fórmula do laço.
- Ausência de arquivo = footprint **vazia**. O loader marca o status como
  `vazia`, nunca `ok`; como `suspeita`, quando a pureza fica abaixo de
  `trace.purity`; e `ok` nos demais casos.
- As footprints são calculadas sobre o z de `coordinates_trace.csv`, quando
  ele existe.

### Arquivos

- **`footprint_<algo>_good.csv`** (sc): região onde o algoritmo é bom
  (`algorithm_bin.csv`), mantida só onde a pureza é ≥ `trace.purity`.
  - Com `trace.use_sim=false` (padrão do engine) usa o desempenho observado;
    com `true`, as previsões do PYTHIA.
  - O nome do arquivo usa uma versão sanitizada do nome do algoritmo
    (`_portable_stems` do instancespace). O nome exato está em
    `run_info.arquivos_footprint`, com `null` para as vazias.
- **`footprint_<algo>_best.csv`** (sc): o mesmo, para a região onde o
  algoritmo é o melhor (`portfolio.csv`). Com `trace.contra` (padrão),
  sobreposições entre footprints best de algoritmos diferentes são resolvidas
  e removidas (trace.py:646-648). As footprints good não passam por essa etapa.
- **`footprint_space.csv`** (eng): região de todas as instâncias. A área e a
  densidade dela são o denominador das colunas `*_Normalized`.
- **`footprint_hard.csv`** (eng): região das instâncias **não** beta-fáceis
  (`IsBetaEasy` falso), o "beta-footprint" do TRACE.
- As métricas das duas últimas ficam em `run_info.footprints_especiais.<space|hard>`:
  `arquivo`, `area`, `densidade`, `pureza`, `elementos` (instâncias
  cobertas), `elementos_bons` e, só no `hard`, `area_normalizada` e
  `densidade_normalizada`.

### `footprint_performance.csv` (sc)

`Row` (algoritmo), `Area_Good_Normalized`, `Density_Good_Normalized`,
`Purity_Good`, `Area_Best_Normalized`, `Density_Best_Normalized` e
`Purity_Best` (float, **3 casas**).

- Área e densidade são normalizadas pelas do espaço (`footprint_space`).
- A pureza é a fração de instâncias cobertas que são boas (ou melhores).
- Vale 0 quando a footprint é vazia.
- As áreas e densidades absolutas não são gravadas.
