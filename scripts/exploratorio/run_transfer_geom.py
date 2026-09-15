"""Variante do run_transfer: separa medidas geometricas das derivadas de modelo.

Teste de circularidade: CL/CLD/DS/DCP/TD_U dependem de treinar classificadores
DENTRO do pyhard (Naive Bayes calibrado e arvores de decisao), entao usa-las
para prever falha de modelos e parcialmente circular. Compara a transferencia
leave-one-dataset-out (z-score por dataset, RandomForestRegressor
random_state=42 n_jobs=1) em tres condicoes: (a) todas as medidas,
(b) so geometricas, (c) so derivadas de modelo.
"""

import pandas as pd

from scripts.exploratorio.run_transfer import NOMES, avaliar, carregar

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

# Classificacao confirmada lendo pyhard/measures.py (linhas citadas).
# GEOMETRICAS: apenas distancias (Gower), vizinhanca, MST e contagens de rotulo.
GEOMETRICAS = {
    "feature_kDN": "vizinhos por Gower + rotulos (measures.py:250-261)",
    "feature_MV": "contagem de instancias por classe (measures.py:400-404)",
    "feature_CB": "proporcao da classe no dataset (measures.py:421-426)",
    "feature_N1": "MST sobre a matriz de Gower (measures.py:447-458)",
    "feature_N2": "razao de distancias intra/extra-classe (measures.py:494-511)",
    "feature_LSC": "local set via Gower ate o inimigo mais proximo (measures.py:537-550)",
    "feature_LSR": "raio do local set via Gower (measures.py:571-586)",
    "feature_Harmfulness": "contagem de 'sou o inimigo mais proximo de quem' (measures.py:608-621)",
    "feature_Usefulness": "pertencimento aos local sets alheios (measures.py:643-658)",
    "feature_F1": "sobreposicao por atributo, maxmin/minmax (measures.py:660-726)",
    "feature_F2": "sobreposicao por atributo (measures.py:728-731)",
    "feature_F3": "sobreposicao por atributo (measures.py:733-736)",
    "feature_F4": "sobreposicao por atributo (measures.py:738-741)",
}
# DERIVADAS DE MODELO: exigem classificador treinado no proprio dataset.
MODELO = {
    "feature_DS": "folhas da arvore NAO podada self.dtc (measures.py:263-271)",
    "feature_DCP": "folhas da arvore podada self.dtc_pruned (measures.py:287-295)",
    "feature_TD_U": "profundidade na arvore nao podada self.dtc (measures.py:313-314)",
    "feature_CL": "predict_proba do Naive Bayes calibrado (measures.py:353-359)",
    "feature_CLD": "predict_proba do Naive Bayes calibrado (measures.py:378-394)",
}
# feature_TD_P tambem seria DERIVADA DE MODELO (arvore podada, measures.py:332),
# mas e descartada por conter NaN no hill-valley.


def main():
    dados = carregar()

    feature_cols = sorted(
        set.intersection(
            *[
                {c for c in t.columns if c.startswith("feature_")}
                for t in dados.values()
            ]
        )
    )
    com_nan = sorted(
        {c for t in dados.values() for c in feature_cols if t[c].isna().any()}
    )
    if com_nan:
        print(f"descartadas por NaN: {com_nan}")
        feature_cols = [c for c in feature_cols if c not in com_nan]

    print("\nclassificacao das medidas (confirmada em pyhard/measures.py):")
    for c in feature_cols:
        if c in GEOMETRICAS:
            print(f"  GEOMETRICA        {c:22s} — {GEOMETRICAS[c]}")
        elif c in MODELO:
            print(f"  DERIVADA DE MODELO {c:21s} — {MODELO[c]}")
        else:
            raise ValueError(f"medida sem classificacao: {c}")

    geo = [c for c in feature_cols if c in GEOMETRICAS]
    mod = [c for c in feature_cols if c in MODELO]
    print(f"\n{len(geo)} geometricas, {len(mod)} derivadas de modelo")

    condicoes = {
        "todas": feature_cols,
        "geometricas": geo,
        "modelo": mod,
    }
    resultados = {}
    importancias = {}
    for nome_c, cols in condicoes.items():
        res, imp = avaliar(dados, cols, zscore=True)
        resultados[nome_c] = res
        importancias[nome_c] = pd.Series(imp, index=cols)

    tabela = resultados["todas"][["dataset_teste"]].copy()
    for nome_c in condicoes:
        tabela[f"rho_{nome_c}"] = resultados[nome_c]["rho"].values
        tabela[f"mae_{nome_c}"] = resultados[nome_c]["mae"].values
    tabela["mae_baseline"] = resultados["todas"]["mae_baseline"].values

    print("\nleave-one-dataset-out com z-score (alvo: n_wrong):")
    print(tabela.round(3).to_string(index=False))

    print("\nrho medio por condicao:")
    for nome_c in condicoes:
        print(f"  {nome_c:12s} {resultados[nome_c]['rho'].mean():.3f}")

    print("\nimportancia media das features na condicao (b) so geometricas:")
    print(
        importancias["geometricas"]
        .sort_values(ascending=False)
        .round(4)
        .to_string()
    )


if __name__ == "__main__":
    main()
