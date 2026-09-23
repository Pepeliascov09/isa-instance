"""Instance-level hardness measures via PyHard."""

from pyhard.measures import ClassificationMeasures


def instance_measures(df, target_col="target", measures_list=None, ccp_alpha=None):
    """Compute hardness measures per instance.

    df: numeric DataFrame with the attributes and the label column together.
    measures_list: list of names (e.g. ["kDN", "N1"]) or None for all of them.
    ccp_alpha: pruning of the tree used by DCP/TD_P. If None, pyhard tunes it
        with GridSearchCV(n_jobs=-1); avoid None inside child processes, since
        loky's nested parallelism hangs on Windows.
    Returns a DataFrame with one column per measure ("feature_" prefix).
    """
    cm = ClassificationMeasures(df, target_col=target_col, ccp_alpha=ccp_alpha)
    return cm.calculate_all(measures_list=measures_list)
