"""Testes de isaspace.ui.loader_is.

Os testes sobre dados reais leem resultados/is/<nome>/ (gere com
scripts/run_is_all.py no .venv-isa) e cobrem os erros que o loader antigo
(isaspace/ui/loader.py) cometeria em silencio com a saida do instancespace. Os
testes sinteticos montam uma pasta minima em tmp_path para o que os quatro
datasets nao exercitam (rotulos nao numericos, furos, colisao de nomes, source).

Uso, da raiz, com o .venv-isa:  .venv-isa/bin/python -m pytest tests/
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from isaspace.ui import loader_is  # noqa: E402
from isaspace.ui.loader import _split_polygons  # noqa: E402  (loader antigo)
from isaspace.ui.loader_is import (  # noqa: E402
    CATEGORICA, INTEIRA, NUMERICA, OK, VAZIA, Poligono, load_is_output,
)

PASTA_IS = RAIZ / "resultados" / "is"
DATASETS = ["iris", "diabetes", "blood-transfusion-service-center", "hill-valley"]


def _pasta(nome):
    path = PASTA_IS / nome
    if not (path / "run_info.json").is_file():
        pytest.skip(f"{path} nao gerado (rode scripts/run_is_all.py no .venv-isa)")
    return path


@pytest.fixture(scope="module", params=DATASETS)
def resultado(request):
    return load_is_output(_pasta(request.param))


# --------------------------------------------------------------------------- #
# dados reais: os erros silenciosos do loader antigo
# --------------------------------------------------------------------------- #
def test_footprint_de_duas_partes_do_iris_vira_dois_poligonos():
    path = _pasta("iris")
    r = load_is_output(path)
    duas_partes = 0
    for (algo, tipo), fp in r.footprints.items():
        if fp.arquivo is None:
            continue
        bruto = pd.read_csv(path / fp.arquivo)
        partes = sorted(bruto["Part"].unique())
        assert len(fp.poligonos) == len(partes), (algo, tipo)
        esperado = [int(((bruto["Part"] == p) & (bruto["Ring"] == "exterior")).sum()) for p in partes]
        assert [p.n_vertices for p in fp.poligonos] == esperado, (algo, tipo)
        if len(partes) == 2:
            duas_partes += 1
            # o loader antigo so separa em linhas NaN: juntaria as partes num so anel
            antigo = _split_polygons(pd.read_csv(path / fp.arquivo, index_col="Row"))
            assert len(antigo) == 1 and len(antigo[0]) == sum(esperado)
    assert duas_partes > 0, "iris deveria ter footprint de 2 partes"


def test_footprint_sem_arquivo_aparece_como_vazia(resultado):
    mapa = resultado.run_info["arquivos_footprint"]
    for (algo, tipo), fp in resultado.footprints.items():
        sem_arquivo = mapa[algo][tipo] is None
        assert sem_arquivo == (fp.status == VAZIA), (algo, tipo, fp.status)
        if sem_arquivo:
            assert fp.poligonos == [] and fp.status != OK and fp.area == 0
    assert len(resultado.footprints) == 2 * len(resultado.algos)


def test_iris_tem_footprints_sem_arquivo():
    r = load_is_output(_pasta("iris"))
    vazias = r.empty_footprints
    assert vazias, "iris deveria ter footprints best vazias"
    for chave in vazias:
        assert not (r.path / f"footprint_{chave[0]}_{chave[1]}.csv").exists()


def test_portfolios_1_e_0_based_dao_o_mesmo_nome_quando_concordam(resultado):
    p = pd.read_csv(resultado.path / "portfolio.csv", dtype={"Row": str}).set_index("Row")["Best_Algorithm"]
    s = pd.read_csv(resultado.path / "portfolio_svm.csv", dtype={"Row": str}).set_index("Row")["Best_Algorithm"]
    inst = resultado.instances
    concordam = (p - 1 == s).to_numpy()
    assert concordam.any()
    assert (inst["best_algo"][concordam] == inst["best_algo_svm"][concordam]).all()
    # ler portfolio_svm com a regra 1-based do loader antigo erraria por um
    ingenuo = [resultado.algos[i - 1] if 1 <= i <= len(resultado.algos) else None for i in s[concordam]]
    assert all(a != b for a, b in zip(ingenuo, inst["best_algo_svm"][concordam]))
    nenhum = (s == -1).to_numpy()
    assert inst["best_algo_svm"][nenhum].isna().all()
    assert inst["best_algo"].notna().all()   # PRELIM sempre escolhe um


def test_anotacoes_batem_com_a_tabela_via_row_original(resultado):
    inst = resultado.instances
    tabela = pd.read_csv(RAIZ / "resultados" / f"table_{resultado.name}.csv", index_col=0)
    t = tabela.loc[inst["row_original"].to_numpy()]
    assert (inst["class"].to_numpy() == t["class"].astype(str).to_numpy()).all()
    np.testing.assert_allclose(inst["ih"].to_numpy(dtype=float), t["ih"].to_numpy(), rtol=1e-12, atol=0)
    assert (inst["n_wrong"].to_numpy() == t["n_wrong"].to_numpy()).all()
    assert resultado.annotations["class"] == CATEGORICA
    assert resultado.annotations["ih"] == NUMERICA
    assert resultado.annotations["n_wrong"] == INTEIRA          # declarada em annotations.json


def test_linhas_de_coordinates_igual_metadata(resultado):
    coords = pd.read_csv(resultado.path / "coordinates.csv", dtype={"Row": str})
    meta = pd.read_csv(resultado.path / "metadata.csv", dtype={"instances": str})
    assert len(coords) == len(meta) == resultado.n
    assert list(coords["Row"]) == list(meta["instances"]) == list(resultado.instances.index)
    assert resultado.run_info["n_instancias"] == resultado.run_info["n_instancias_entrada"] == resultado.n


def test_binarios_viram_bool(resultado):
    inst = resultado.instances
    for a in resultado.algos:
        assert inst[f"algo_{a}_bin"].dtype == bool
        assert inst[f"algo_{a}_svm"].dtype == bool
    assert inst["IsBetaEasy"].dtype == bool
    # algo_*_bin e o limiar do perf sobre algo_* (max_perf, abs_perf, epsilon=0.5)
    eps = resultado.run_options["perf"]["epsilon"]
    for a in resultado.algos:
        assert (inst[f"algo_{a}_bin"] == (inst[f"algo_{a}"] >= eps)).all()


def test_sifted_report_coerente(resultado):
    rep = resultado.sifted_report
    assert set(rep["status"]) <= {"kept", "dropped_correlation", "dropped_redundancy"}
    assert list(rep.loc[rep["status"] == "kept", "feature"]) == resultado.features
    mantidas = rep[rep["status"] == "kept"].set_index("feature")["cluster"]
    for linha in rep[rep["status"] == "dropped_redundancy"].itertuples():
        assert linha.kept_instead in mantidas.index
        assert mantidas[linha.kept_instead] == linha.cluster
    assert resultado.run_info["avisos"] == []


def test_pythia_proba_fora_e_dentro_da_amostra(resultado):
    sub, hat = resultado.pythia_proba, resultado.pythia_proba_hat
    for proba in (sub, hat):
        assert proba.shape == (resultado.n, len(resultado.algos))
        assert list(proba.index) == list(resultado.instances.index)
        assert list(proba.columns) == resultado.algos
        assert ((proba.to_numpy() >= 0) & (proba.to_numpy() <= 1)).all()
    assert not np.allclose(sub.to_numpy(), hat.to_numpy())
    # algorithm_svm.csv (y_hat) contra pr0_hat < 0.5: a contagem do run_info
    y_hat = resultado.instances[[f"algo_{a}_svm" for a in resultado.algos]].to_numpy()
    discorda = int(np.sum(y_hat != (hat.to_numpy() < 0.5)))
    assert discorda == resultado.run_info["pythia"]["y_hat_discorda_de_pr0_hat"]


def test_coordinates_e_o_z_do_pilot_e_trace_so_com_jitter(resultado):
    rob = resultado.run_info["trace_robustez"]
    z = resultado.instances[["z_1", "z_2"]]
    bruto = pd.read_csv(resultado.path / "coordinates.csv", dtype={"Row": str}).set_index("Row")
    assert z.equals(bruto[["z_1", "z_2"]])
    # pares entre posicoes distintas a menos do limiar, no z do PILOT
    pos = np.unique(z.to_numpy(), axis=0)
    d = np.sqrt(((pos[:, None, :] - pos[None, :, :]) ** 2).sum(-1))
    proximos = int(np.sum(np.triu(d < rob["limiar"], k=1)))
    assert proximos == rob["pares_distintos_proximos"]
    if not rob["jitter_aplicado"]:
        assert resultado.coordinates_trace is None
        assert not (resultado.path / "coordinates_trace.csv").exists()
        return
    tr = resultado.coordinates_trace
    mexidos = set(rob["rotulos_perturbados"])
    difere = (tr != z).any(axis=1)
    assert set(difere[difere].index) == mexidos
    assert np.abs((tr - z).to_numpy()).max() <= rob["deslocamento_maximo"] + 1e-12


def test_pythia_confusion_reproduz_svm_table(resultado):
    conf = resultado.pythia_confusion
    assert (conf.sum(axis=1) == resultado.n).all()
    tab = resultado.svm_table.loc[resultado.algos]
    acc = 100 * (conf.tp + conf.tn) / resultado.n
    prec = 100 * conf.tp / (conf.tp + conf.fp)
    rec = 100 * conf.tp / (conf.tp + conf.fn)
    # a svm_table arredonda os percentuais para uma casa
    np.testing.assert_allclose(acc.round(1), tab["CV_model_accuracy"], atol=0.051)
    np.testing.assert_allclose(prec.round(1), tab["CV_model_precision"], atol=0.051)
    np.testing.assert_allclose(rec.round(1), tab["CV_model_recall"], atol=0.051)
    # tn + fp = instancias ruins no desempenho observado
    ruins = (~resultado.instances[[f"algo_{a}_bin" for a in resultado.algos]]).sum().to_numpy()
    assert ((conf.tn + conf.fp).to_numpy() == ruins).all()


def test_pythia_selection(resultado):
    sel = resultado.pythia_selection
    inst = resultado.instances
    assert list(sel.index) == list(inst.index)
    assert sel["selection0"].equals(inst["best_algo_svm"])
    com = sel["selection0"].notna()
    assert (sel.loc[com, "selection1"] == sel.loc[com, "selection0"]).all()
    assert sel["selection1"].notna().all()
    # sem recomendacao, selection1 cai no algoritmo com mais instancias boas
    padrao = inst[[f"algo_{a}_bin" for a in resultado.algos]].mean().idxmax()[len("algo_"):-len("_bin")]
    assert (sel.loc[~com, "selection1"] == padrao).all()


def test_features_fora_do_pilot_vem_do_metadata(resultado):
    meta = pd.read_csv(resultado.path / "metadata.csv", dtype={"instances": str}).set_index("instances")
    todas = [c[len("feature_"):] for c in meta.columns if c.startswith("feature_")]
    assert resultado.features_all == todas
    assert set(resultado.features_fora_pilot) == set(todas) - set(resultado.features)
    fora = resultado.sifted_report.set_index("feature").loc[resultado.features_fora_pilot, "status"]
    assert (fora != "kept").all()
    for f in todas:
        assert (resultado.instances[f"feature_{f}"].to_numpy() == meta[f"feature_{f}"].to_numpy()).all()


def test_pilot_r2(resultado):
    r2 = resultado.pilot_r2
    assert list(r2["variable"]) == resultado.features + resultado.algos
    assert list(r2["kind"]) == ["feature"] * len(resultado.features) + ["algorithm"] * len(resultado.algos)
    assert ((r2["r2"] >= 0) & (r2["r2"] <= 1)).all()


def test_sifted_correlations_e_silhueta(resultado):
    c = resultado.sifted_correlations
    feats = list(resultado.sifted_report["feature"])
    assert len(c) == len(feats) * len(resultado.algos)
    assert list(resultado.sifted_rho.index) == feats
    rho, pval = resultado.sifted_rho, resultado.sifted_pval
    rep = resultado.sifted_report.set_index("feature")
    for f in feats:
        j = rho.loc[f].abs().idxmax()
        assert rho.loc[f, j] == pytest.approx(rep.loc[f, "rho"])
        assert pval.loc[f, j] == pytest.approx(rep.loc[f, "pval"])
    sil = resultado.sifted_silhouette
    k = resultado.run_options["sifted"]["k"]
    assert list(sil["k"]) == list(range(3, 3 + len(sil)))
    assert list(sil.loc[sil["used"], "k"]) == [k]
    assert sil["best"].sum() == 1 and sil.loc[sil["best"], "silhouette"].iloc[0] == sil["silhouette"].max()


def test_footprints_space_e_hard(resultado):
    space, hard = resultado.footprint_space, resultado.footprint_hard
    assert space.tipo == "space" and space.status == OK and space.area > 0
    for fp in (space, hard):
        assert (fp.arquivo is None) == (fp.status == VAZIA) == (fp.poligonos == [])
    # as areas normalizadas do footprint_performance sao area / area do espaco
    for fp in resultado.footprints.values():
        if fp.area_normalizada >= 0.1:
            assert fp.area / fp.area_normalizada == pytest.approx(space.area, rel=0.01)
    if hard.status != VAZIA:
        assert hard.area_normalizada == pytest.approx(hard.area / space.area, rel=1e-6)


def test_cloister_como_poligono(resultado):
    for pol in (resultado.bounds, resultado.bounds_pruned):
        assert pol is not None and pol.n_vertices >= 3 and pol.area > 0
    # a caixa envolvente da fronteira cobre a nuvem de instancias
    z = resultado.instances[["z_1", "z_2"]].to_numpy()
    ext = resultado.bounds.exterior
    assert (z.min(axis=0) >= ext.min(axis=0) - 1e-9).all()
    assert (z.max(axis=0) <= ext.max(axis=0) + 1e-9).all()


def test_loader_novo_recusa_a_pasta_do_pyispace():
    antiga = RAIZ / "resultados" / "isa" / "iris"
    if not antiga.is_dir():
        pytest.skip("resultados/isa/iris ausente")
    with pytest.raises(FileNotFoundError, match="run_info.json"):
        load_is_output(antiga)




# --------------------------------------------------------------------------- #
# tipos declarados, degenerate_report e aba Features
# --------------------------------------------------------------------------- #
def test_tipos_declarados_vem_do_annotations_json(resultado):
    declarados = json.loads((resultado.path / "annotations.json").read_text())
    assert declarados == {"class": CATEGORICA, "ih": NUMERICA, "n_wrong": INTEIRA}
    assert resultado.run_info["tipos_anotacao"]["declarados"] == declarados
    for col, tipo in declarados.items():
        assert resultado.annotations[col] == tipo
        assert resultado.annotation_origins[col] == "declarado"
    # row_original nao esta declarada: tipo pela heuristica, marcado como inferido
    assert resultado.annotation_origins["row_original"] == "inferido"
    assert resultado.anotacoes_inferidas == ["row_original"]
    assert pd.api.types.is_numeric_dtype(resultado.instances["n_wrong"])
    assert resultado.instances["class"].map(type).eq(str).all()


def test_tipo_forcado_sobrepoe_declaracao_e_heuristica():
    path = _pasta("hill-valley")
    r = load_is_output(path, annotation_types={"row_original": CATEGORICA, "class": NUMERICA})
    assert r.annotations["row_original"] == CATEGORICA and r.annotation_origins["row_original"] == "forcado"
    assert r.annotations["class"] == NUMERICA and r.annotation_origins["class"] == "forcado"
    assert pd.api.types.is_numeric_dtype(r.instances["class"])       # "0"/"1" -> 0/1


def _features_recebidas(nome):
    tabela = pd.read_csv(RAIZ / "resultados" / f"table_{nome}.csv", index_col=0, nrows=1)
    return [c[len("feature_"):] for c in tabela.columns if c.startswith("feature_")]


def test_degenerate_report_cobre_o_que_nao_chegou_ao_engine(resultado):
    deg = resultado.degenerate_report
    assert deg is not None and list(deg.columns) == ["feature", "var_bruta", "iqr", "motivo"]
    recebidas = _features_recebidas(resultado.name)
    assert set(deg["feature"]) == set(recebidas) - set(resultado.features_all)
    assert not set(deg["feature"]) & set(resultado.features_all)
    assert deg["motivo"].str.len().gt(0).all()


def test_degenerate_report_do_iris():
    r = load_is_output(_pasta("iris"))
    assert list(r.degenerate_report["feature"]) == ["kDN", "MV", "CB", "N1", "Harmfulness"]


def test_features_table(resultado):
    t = resultado.features_table()
    recebidas = _features_recebidas(resultado.name)
    assert list(t["feature"]) == recebidas                  # ordem do feature_info.csv
    assert set(t["status"]) <= {"kept", "dropped_degenerate", "dropped_correlation", "dropped_redundancy"}
    assert (t["status"] == "dropped_degenerate").sum() == len(resultado.degenerate_report)
    mantidas = t[t["status"] == "kept"]
    assert list(mantidas["feature"]) == resultado.features
    assert mantidas["r2_pilot"].notna().all() and t.loc[t["status"] != "kept", "r2_pilot"].isna().all()
    red = t[t["status"] == "dropped_redundancy"]
    assert red["substituida_por"].isin(resultado.features).all()
    assert t.loc[t["status"] != "dropped_degenerate", "max_abs_rho"].ge(0).all()
    derivadas = {"CL", "CLD", "DS", "DCP", "TD_U", "TD_P"}
    assert set(t.loc[t["family"] == "model_derived", "feature"]) == derivadas & set(recebidas)
    assert set(t["family"]) == {"model_derived", "geometric"}


def test_pertinencia_na_footprint_bate_com_o_trace(resultado):
    """A pertinencia do loader reproduz os elementos que o TRACE contou."""
    for nome, fp in (("space", resultado.footprint_space), ("hard", resultado.footprint_hard)):
        esperado = resultado.run_info["footprints_especiais"][nome]["elementos"]
        obtido = len(resultado.instancias_na_footprint(fp)) if fp.poligonos else 0
        assert obtido == esperado, nome


def test_poligono_contem_com_furo_e_bordas():
    quadrado = Poligono(np.array([[0, 0], [4, 0], [4, 4], [0, 4]], float),
                        [np.array([[1, 1], [2, 1], [2, 2], [1, 2]], float)])
    pts = np.array([[3, 3], [1.5, 1.5], [0, 2], [4, 4], [1, 1.5], [5, 5], [-0.1, 2]])
    assert quadrado.contem(pts).tolist() == [True, False, True, True, True, False, False]


# --------------------------------------------------------------------------- #
# pasta sintetica
# --------------------------------------------------------------------------- #
def _csv(path, texto):
    path.write_text(texto.strip() + "\n")


@pytest.fixture
def pasta_sintetica(tmp_path):
    """Quatro instancias com rotulos de texto, dois algoritmos, uma footprint
    com furo, uma sem arquivo e uma anotacao chamada z_1."""
    p = tmp_path / "sint"
    p.mkdir()
    rot = ["alfa", "beta", "gama", "delta"]
    _csv(p / "coordinates.csv", "Row,z_1,z_2\n" + "\n".join(
        f"{r},{x},{y}" for r, x, y in zip(rot, [0, 4, 4, 0], [0, 0, 4, 4])))
    _csv(p / "algorithm_raw.csv", "Row,A,B\n" + "\n".join(
        f"{r},{a},{b}" for r, a, b in zip(rot, [0.9, 0.2, 0.7, 0.6], [0.1, 0.8, 0.3, 0.4])))
    _csv(p / "algorithm_bin.csv", "Row,A,B\n" + "\n".join(
        f"{r},{a},{b}" for r, a, b in zip(rot, ["True", "False", "True", "True"],
                                          ["False", "True", "False", "False"])))
    _csv(p / "algorithm_svm.csv", "Row,A,B\n" + "\n".join(f"{r},True,False" for r in rot))
    _csv(p / "feature_raw.csv", "Row,f1,f2\n" + "\n".join(f"{r},{i},{2 * i}" for i, r in enumerate(rot)))
    _csv(p / "feature_process.csv", "Row,f1,f2\n" + "\n".join(f"{r},{i / 3},{-i / 3}" for i, r in enumerate(rot)))
    _csv(p / "good_algos.csv", "Row,NumGoodAlgos\n" + "\n".join(f"{r},1" for r in rot))
    _csv(p / "beta_easy.csv", "Row,IsBetaEasy\n" + "\n".join(f"{r},False" for r in rot))
    _csv(p / "portfolio.csv", "Row,Best_Algorithm\n" + "\n".join(
        f"{r},{v}" for r, v in zip(rot, [1, 2, 1, 1])))
    _csv(p / "portfolio_svm.csv", "Row,Best_Algorithm\n" + "\n".join(
        f"{r},{v}" for r, v in zip(rot, [0, 1, -1, 0])))
    _csv(p / "metadata.csv", "instances,Source,grupo,z_1,peso,feature_f1,feature_f2,algo_A,algo_B\n" + "\n".join(
        f"{r},{s},{g},{k},{w},0,0,0,0" for r, s, g, k, w in
        zip(rot, ["S1", "S1", "S2", "S2"], ["x", "y", "x", ""], [1, 0, 1, 0], [0.5, 1.5, 2.5, 3.5])))
    _csv(p / "footprint_A_good.csv", """
Row,Part,Ring,Vertex,z_1,z_2
1,1,exterior,1,0,0
2,1,exterior,2,4,0
3,1,exterior,3,4,4
4,1,exterior,4,0,4
5,1,hole_1,1,1,1
6,1,hole_1,2,2,1
7,1,hole_1,3,2,2
8,1,hole_1,4,1,2
9,2,exterior,1,10,10
10,2,exterior,2,11,10
11,2,exterior,3,10,11
""")
    _csv(p / "footprint_performance.csv", """
Row,Area_Good_Normalized,Density_Good_Normalized,Purity_Good,Area_Best_Normalized,Density_Best_Normalized,Purity_Best
A,0.9,1.0,0.8,0.0,0.0,0.0
B,0.0,0.0,0.0,0.0,0.0,0.0
""")
    _csv(p / "projection_matrix.csv", "Row,f1,f2\nZ_{1},0.5,0.5\nZ_{2},-0.5,0.5")
    _csv(p / "svm_table.csv", "Row,Avg_Perf_all_instances\nA,0.6\nB,0.4\nOracle,0.9\nSelector,0.6")
    _csv(p / "sifted_report.csv", "feature,status,rho,rho_algo,pval,n_algos_sig,cluster,kept_instead\n"
         "f1,kept,0.5,A,0.01,1,,\nf2,kept,-0.4,B,0.02,1,,")
    _csv(p / "pythia_proba.csv", "Row,A,B,A_hat,B_hat\n" + "\n".join(
        f"{r},0.1,0.9,0.2,0.8" for r in rot))
    _csv(p / "pythia_confusion.csv", "Algorithm,tn,fp,fn,tp\nA,1,0,0,3\nB,3,0,1,0")
    _csv(p / "pythia_selection.csv", "Row,selection0,selection1\n" + "\n".join(
        f"{r},{a},{b}" for r, a, b in zip(rot, ["A", "B", "", "A"], ["A", "B", "A", "A"])))
    _csv(p / "pilot_r2.csv", "variable,kind,r2\nf1,feature,0.9\nf2,feature,0.5\n"
         "A,algorithm,0.3\nB,algorithm,0.2")
    _csv(p / "sifted_correlations.csv", "feature,algorithm,rho,pval\nf1,A,0.5,0.01\n"
         "f1,B,-0.1,0.5\nf2,A,0.2,0.3\nf2,B,-0.4,0.02")
    _csv(p / "sifted_silhouette.csv", "k,silhouette,used,best")
    _csv(p / "footprint_space.csv", "Row,Part,Ring,Vertex,z_1,z_2\n1,1,exterior,1,0,0\n"
         "2,1,exterior,2,4,0\n3,1,exterior,3,4,4\n4,1,exterior,4,0,4")
    (p / "bounds.csv").write_text("Row,z_1,z_2\nbnd_pnt_1,-1,-1\nbnd_pnt_2,5,-1\nbnd_pnt_3,5,5\nbnd_pnt_4,-1,5\n")
    (p / "run_options.json").write_text(json.dumps({"trace": {"purity": 0.55}, "perf": {"epsilon": 0.5}}))
    (p / "run_info.json").write_text(json.dumps({
        "algoritmos": ["A", "B"],
        "arquivos_footprint": {"A": {"good": "footprint_A_good.csv", "best": None},
                               "B": {"good": None, "best": None}},
        "trace_robustez": {"jitter_aplicado": False},
        "footprints_especiais": {
            "space": {"arquivo": "footprint_space.csv", "area": 16.0, "pureza": 1.0},
            "hard": {"arquivo": None, "area": 0.0, "pureza": 0.0,
                     "area_normalizada": 0.0, "densidade_normalizada": 0.0},
        },
    }))
    return p


def test_sintetico_rotulos_texto_furos_e_vazias(pasta_sintetica):
    r = load_is_output(pasta_sintetica)
    assert list(r.instances.index) == ["alfa", "beta", "gama", "delta"]
    good = r.footprints[("A", "good")]
    assert good.status == OK and len(good.poligonos) == 2
    externo = good.poligonos[0]
    assert externo.n_vertices == 4 and len(externo.furos) == 1
    assert externo.area == pytest.approx(16 - 1)            # quadrado 4x4 menos furo 1x1
    assert good.area == pytest.approx(15 + 0.5)
    for chave in [("A", "best"), ("B", "good"), ("B", "best")]:
        assert r.footprints[chave].status == VAZIA
    assert sorted(r.empty_footprints) == [("A", "best"), ("B", "best"), ("B", "good")]


def test_sintetico_portfolios_bool_e_anotacoes(pasta_sintetica):
    r = load_is_output(pasta_sintetica)
    inst = r.instances
    assert list(inst["best_algo"]) == ["A", "B", "A", "A"]
    assert list(inst["best_algo_svm"]) == ["A", "B", None, "A"]
    assert list(inst["algo_A_bin"]) == [True, False, True, True]
    assert inst["IsBetaEasy"].dtype == bool
    assert r.source_column == "source" and list(inst["source"]) == ["S1", "S1", "S2", "S2"]
    # anotacao chamada z_1 nao sobrescreve a coordenada
    assert r.annotation_renames == {"z_1": "ann_z_1"}
    assert list(inst["z_1"]) == [0, 4, 4, 0]
    assert r.annotations == {"grupo": CATEGORICA, "ann_z_1": CATEGORICA, "peso": NUMERICA}
    assert pd.isna(inst.loc["delta", "grupo"])
    assert list(inst["ann_z_1"]) == ["1", "0", "1", "0"]
    assert list(r.projection_matrix.index) == ["z_1", "z_2"]
    assert r.bounds.area == pytest.approx(36)
    forcado = load_is_output(pasta_sintetica, annotation_types={"z_1": NUMERICA})
    assert forcado.annotations["ann_z_1"] == NUMERICA


def test_sintetico_arquivos_novos(pasta_sintetica):
    r = load_is_output(pasta_sintetica)
    assert list(r.pythia_proba.columns) == ["A", "B"] and list(r.pythia_proba_hat.columns) == ["A", "B"]
    assert r.pythia_proba.loc["alfa", "A"] == 0.1 and r.pythia_proba_hat.loc["alfa", "A"] == 0.2
    assert list(r.pythia_selection["selection0"]) == ["A", "B", None, "A"]
    assert r.pythia_confusion.loc["B", "fn"] == 1
    assert r.sifted_rho.loc["f2", "B"] == -0.4 and r.sifted_pval.loc["f1", "A"] == 0.01
    assert r.sifted_silhouette.empty
    assert r.coordinates_trace is None
    assert r.footprint_space.status == OK and r.footprint_space.area == pytest.approx(16)
    assert r.footprint_hard.status == VAZIA


def test_sintetico_coordinates_trace_sem_jitter_e_erro(pasta_sintetica):
    (pasta_sintetica / "coordinates_trace.csv").write_text(
        (pasta_sintetica / "coordinates.csv").read_text())
    with pytest.raises(ValueError, match="coordinates_trace.csv"):
        load_is_output(pasta_sintetica)


def test_sintetico_selecao_com_algoritmo_desconhecido_e_erro(pasta_sintetica):
    f = pasta_sintetica / "pythia_selection.csv"
    f.write_text(f.read_text().replace("delta,A,A", "delta,Z,A"))
    with pytest.raises(ValueError, match="desconhecidos"):
        load_is_output(pasta_sintetica)




def test_sintetico_sem_arquivos_auxiliares(pasta_sintetica):
    r = load_is_output(pasta_sintetica)
    assert set(r.annotation_origins.values()) == {"inferido"}
    assert r.degenerate_report is None and r.feature_info is None
    t = r.features_table()
    assert list(t["feature"]) == ["f1", "f2"] and "family" not in t.columns


@pytest.mark.parametrize("declaracao, erro", [
    ({"grupo": NUMERICA}, "nao numericos"),          # 'x', 'y' nao sao numeros
    ({"peso": "texto"}, "invalido"),
    ({"feature_f1": NUMERICA}, "nao sao anotacoes"),
])
def test_sintetico_declaracao_invalida_e_erro(pasta_sintetica, declaracao, erro):
    (pasta_sintetica / "annotations.json").write_text(json.dumps(declaracao))
    with pytest.raises(ValueError, match=erro):
        load_is_output(pasta_sintetica)


def test_sintetico_declaracao_valida(pasta_sintetica):
    (pasta_sintetica / "annotations.json").write_text(json.dumps({"peso": INTEIRA, "z_1": NUMERICA}))
    r = load_is_output(pasta_sintetica)
    assert r.annotations["peso"] == INTEIRA and r.annotation_origins["peso"] == "declarado"
    assert r.annotations["ann_z_1"] == NUMERICA and r.annotation_origins["ann_z_1"] == "declarado"
    assert r.annotation_origins["grupo"] == "inferido"


def test_sintetico_erros_explicitos(pasta_sintetica):
    # Row desalinhado entre arquivos vira erro, nao join silencioso
    f = pasta_sintetica / "algorithm_bin.csv"
    f.write_text(f.read_text().replace("delta", "epsilon"))
    with pytest.raises(ValueError, match="algorithm_bin.csv"):
        load_is_output(pasta_sintetica)


def test_sintetico_footprint_no_formato_antigo_e_recusada(pasta_sintetica):
    (pasta_sintetica / "footprint_A_good.csv").write_text("Row,z_1,z_2\n0,0,0\n1,1,0\n2,0,1\n")
    with pytest.raises(ValueError, match="formato do instancespace"):
        load_is_output(pasta_sintetica)


@pytest.mark.parametrize("modulo", ["loader_is.py", "app.py"])
def test_interface_nao_importa_backend(modulo):
    fonte = (Path(loader_is.__file__).parent / modulo).read_text()
    for proibido in ("instancespace", "sklearn", "pyispace", "pyhard"):
        assert f"import {proibido}" not in fonte and f"from {proibido}" not in fonte
