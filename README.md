# isa-instance

Interface interativa para **Instance Space Analysis (ISA) em nível de observação
individual**: cada ponto do espaço é uma instância de um dataset, não um dataset
inteiro. Projeto de Iniciação Científica (IC7) no ITA, sob orientação da
Profa. Ana Carolina Lorena.

## O que é

Um pipeline que parte de datasets do OpenML, calcula medidas de dificuldade por
instância (PyHard) e o desempenho *out-of-fold* de um portfólio de seis
classificadores, monta uma tabela por instância, projeta essa tabela num plano
com PILOT e delimita *footprints* com TRACE (pyispace), e uma interface
Panel/Bokeh que lê a pasta de saída no formato MATILDA e permite explorar o
resultado. Os resultados para quatro datasets já estão versionados em
`resultados/`, então o app abre logo depois do clone.

## O que a interface mostra

- **Instance Space**: espaço de dados (PCA dos atributos originais) e espaço de
  instâncias (PILOT) lado a lado sobre o mesmo DataFrame; a seleção por *lasso*
  ou caixa num destaca os mesmos pontos no outro e alimenta as demais abas.
- **Footprint Performance**: mapa z_1 x z_2 com as *footprints* (good ou best,
  de um algoritmo ou de todos) e a tabela `footprint_performance.csv`, com as
  *footprints* vazias ou de pureza abaixo do limiar sinalizadas.
- **Distributions**: histograma ou boxplot de até seis variáveis (medidas,
  desempenhos, `ih`, `n_wrong`) comparando todas as instâncias com as
  selecionadas.
- **Data Explorer**: scatter x-y com cor configurável, filtro `pandas.query`
  e tabela das linhas filtradas.

Na barra lateral ficam fixos o seletor de dataset, o botão de recarregar e os
números do resultado (instâncias, features no PILOT, features descartadas,
algoritmos); o bloco de baixo troca conforme a aba ativa.

## Limitações (leia antes de interpretar)

- O pyispace 0.3.7 implementa **PRELIM parcial** (recorte de outliers,
  normalização Yeo-Johnson + z-score e binarização do desempenho), **PILOT** e
  **TRACE**. Ele **não implementa SIFTED, CLOISTER nem PYTHIA**.
- Consequências: **não há recomendador de algoritmo**, e a seleção de
  features **não é SIFTED**: é apenas o descarte, feito em `isaspace/isa.py`,
  das medidas que o pré-processamento do PRELIM tornaria constantes (IQR zero
  antes do recorte de outliers). Toda medida que sobrevive a esse descarte
  entra no PILOT.
- As *footprints* do tipo **best são degeneradas nos quatro datasets**: em cada
  um, dois ou três dos seis algoritmos ficam com polígono vazio e quase todos
  os demais com pureza abaixo do limiar (a exceção recorrente é a árvore de
  decisão). Isso é um **resultado observado** do TRACE com este portfólio e
  estes dados, não um defeito da interface, que marca esses casos como "vazia"
  ou "suspeita" na aba Footprint Performance.
- O espaço de dados (PCA dos atributos padronizados) é só referência visual;
  a análise está no espaço de instâncias do PILOT.

## Instalação

1. **Python 3.11**, não 3.12 nem 3.13. O `pyhard` 2.2.4 fixa `pandas~=1.5.0`,
   e o pandas 1.5.x não tem *wheels* para Python 3.12 ou 3.13. Este projeto
   foi desenvolvido com o 3.11.9.

   ```powershell
   # Windows
   py -3.11 -m venv .venv
   .venv\Scripts\activate
   ```

   ```bash
   # Linux / macOS
   python3.11 -m venv .venv
   source .venv/bin/activate
   ```

2. Dependências com versões fixas (congeladas do ambiente que gerou os
   resultados):

   ```
   pip install -r requirements.txt
   ```

3. **Passo obrigatório**: corrigir o pyispace para o Python 3.11.

   ```
   python scripts/apply_pyispace_patch.py
   ```

   Motivo: `pyispace/train.py` (versão 0.3.7) declara o dataclass `Data` com
   dois campos cujo default é um array NumPy
   (`field(init=False, default=np.array([]))`). Até o Python 3.10 o módulo
   `dataclasses` só recusava defaults do tipo `list`, `dict` e `set`; a partir
   do 3.11 ele recusa qualquer default não hasheável, e `np.ndarray` não é
   hasheável. Resultado: o simples `import pyispace` morre com
   `ValueError: mutable default <class 'numpy.ndarray'> for field Yraw is not
   allowed: use default_factory`, e com ele os módulos `pyhard.integrator` e
   `pyhard.cli`. O script troca os dois defaults por `default_factory`
   (semântica idêntica), guarda um backup em `train.py.orig`, valida o import
   num subprocesso e é idempotente. Qualquer reinstalação do pyispace desfaz
   o patch, e o script precisa ser executado de novo. Detalhes em
   `patches/README.md`.

## Como gerar os dados

Os resultados dos quatro datasets (iris, diabetes,
blood-transfusion-service-center e hill-valley) já estão em `resultados/`.
Para regenerar, na raiz do projeto e com o `.venv` ativo:

1. `python run_table.py`: baixa os quatro datasets do OpenML (ids 61, 37, 1464
   e 1479), calcula as medidas por instância e o desempenho *out-of-fold* do
   portfólio (kNN, árvore, Naive Bayes, regressão logística, SVM RBF e random
   forest; 5 folds, semente 42) e grava `resultados/table_<nome>.csv`.
2. `python scripts/run_isa_all.py [nome ...]`: converte cada tabela para o
   formato do pyispace, roda PILOT + TRACE e grava
   `resultados/isa/<nome>/` no layout MATILDA (padrão: iris e diabetes; passe
   os nomes para os outros).
3. `python scripts/build_data_space.py [nome ...]`: gera
   `resultados/isa/<nome>/data_space.csv` (PCA 2D dos atributos originais) para
   o scatter da esquerda da aba Instance Space (padrão: os quatro).

## Como abrir o app

```
python -m isaspace.ui.app
```

Abre o navegador em <http://localhost:5006>. Opções: `--no-show` (não abre o
navegador), `--port N` e `--root PASTA` (outra pasta no formato de
`resultados/isa/`). Se o servidor for reiniciado ou a conexão cair, a página
mostra uma faixa vermelha pedindo para recarregar; até lá ela parece viva mas
não responde a nada além da troca de abas.

## Testes

`scripts/verify_browser.py` é o teste de regressão da interface. Ele controla
um Edge ou Chrome headless pelo DevTools Protocol e, contra um app já servido,
troca datasets, percorre as abas nos dois sentidos, mexe nos controles de cada
aba e repete o caminho "abas antes do dataset", conferindo título, cabeçalho,
sidebar e painel a cada passo:

```
python -m isaspace.ui.app --no-show
python scripts/verify_browser.py --url http://localhost:5006/ --desconexao 5012
```

## Estrutura de pastas

- `isaspace/intake.py`: carga do OpenML e conversão para o frame numérico do PyHard.
- `isaspace/measures.py`: as medidas de dificuldade por instância (`ClassificationMeasures` do pyhard).
- `isaspace/performance.py`: desempenho *out-of-fold* do portfólio de seis classificadores.
- `isaspace/pipeline.py`: tabela unificada por instância (`feature_*`, `algo_*`, `proba_*`, `class`, `n_wrong`, `ih`).
- `isaspace/projection.py`: projeção 2D simples (winsorização, z-score, PCA) usada nos experimentos iniciais.
- `isaspace/footprint.py`: *footprints* aproximadas por grade, dos experimentos iniciais.
- `isaspace/isa.py`: ponte com o pyispace: `to_isa_metadata` (descarte de degeneradas) e `run_isa` (PILOT + TRACE, gravação MATILDA e guarda-corpo do sentido de "bom").
- `isaspace/app.py` e `isaspace/app_v2.py`: primeira interface (scatter ligado a um painel de detalhes), superada por `isaspace/ui/`; os dois arquivos são idênticos e ficam como histórico.
- `isaspace/ui/loader.py`: leitor da pasta MATILDA que devolve um único `IsaResult`.
- `isaspace/ui/app.py`: a interface das quatro abas.
- `scripts/apply_pyispace_patch.py`: patch do pyispace para Python 3.11.
- `scripts/run_isa_all.py`: PILOT + TRACE para cada dataset.
- `scripts/build_data_space.py`: espaço de dados (PCA) para a interface.
- `scripts/verify_browser.py`: teste de regressão da interface no navegador.
- `patches/README.md`: descrição do bug do pyispace e do patch.
- `resultados/`: `table_<nome>.csv`, `isa/<nome>/` (saídas MATILDA) e PNGs dos experimentos iniciais.
- `run_*.py`: scripts de experimento da primeira fase (medidas, contraste com meta-features, transferência entre datasets); os números estão em `resumo.md`.
- `resumo.md`: resumo técnico dos resultados da primeira fase.
- `requirements.txt`: dependências com versões fixas.

`isaspace/ui/` é deliberadamente independente do backend de cálculo: lê uma
pasta no formato MATILDA (`resultados/isa/<nome>/`) e a tabela por instância, e
não importa pyispace, pyhard nem scikit-learn; o que faltar na pasta é gerado
pelos scripts, nunca pela interface.

## Referências

- Smith-Miles, K.; Muñoz, M. A. *Instance Space Analysis for Algorithm
  Testing: Methodology and Software Tools*. ACM Computing Surveys, 55(12),
  2023.
- PyHard: Paiva, P. Y. A.; Moreno, C. C.; Smith-Miles, K.; Valeriano, M. G.;
  Lorena, A. C. *Relating instance hardness to classification performance in a
  dataset: a visual approach*. Machine Learning, 111, 2022. Código:
  <https://gitlab.com/ita-ml/pyhard>.
- pyispace: implementação em Python de partes do MATILDA (PRELIM parcial,
  PILOT e TRACE). <https://gitlab.com/ita-ml/pyispace>.
- MATILDA / InstanceSpace (referência em MATLAB):
  <https://github.com/andremun/InstanceSpace>.
