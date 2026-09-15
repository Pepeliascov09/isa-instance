"""Calculo de medidas de dificuldade em nivel de instancia via PyHard."""

from pyhard.measures import ClassificationMeasures


def instance_measures(df, target_col="target", measures_list=None, ccp_alpha=None):
    """Calcula medidas de dificuldade por instancia.

    df: DataFrame numerico com atributos e a coluna de rotulo juntos.
    measures_list: lista de nomes (ex.: ["kDN", "N1"]) ou None para todas.
    ccp_alpha: poda da arvore usada por DCP/TD_P. Se None, o pyhard tuna via
        GridSearchCV(n_jobs=-1) — evite None dentro de processos filhos, pois
        o paralelismo aninhado do loky trava no Windows.
    Retorna um DataFrame com uma coluna por medida (prefixo "feature_").
    """
    cm = ClassificationMeasures(df, target_col=target_col, ccp_alpha=ccp_alpha)
    return cm.calculate_all(measures_list=measures_list)
