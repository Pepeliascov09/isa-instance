"""Testes rapidos da validacao de upload (isaspace.ui.upload) e do disparo do
engine em subprocesso (isaspace.ui.execucao).

Uso, da raiz, no .venv-isa:
    .venv-isa/bin/python -m pytest tests/test_upload.py
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from isaspace.ui import execucao
from isaspace.ui.upload import (
    FRACAO_MIN, TEMPOS_ALGOS, TEMPOS_MEDIDOS, fora_da_faixa, fracao_boas,
    linhas_do_prelim, tempo_estimado, validar_anotacoes, validar_feature_info, validar_metadata,
)

RAIZ = Path(__file__).resolve().parents[1]
PASTA_IS = RAIZ / "resultados" / "is"
DATASETS = ["iris", "diabetes", "blood-transfusion-service-center", "hill-valley"]


def _csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def _valido(n=30, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"instances": [f"i{k}" for k in range(n)]})
    for f in ("a", "b", "c"):
        df[f"feature_{f}"] = rng.normal(size=n)
    for a in ("x", "y"):
        df[f"algo_{a}"] = rng.uniform(size=n)
    df["grupo"] = rng.choice(["p", "q"], size=n)
    return df


# ------------------------------------------------------------------ metadata
def test_metadata_valido():
    v = validar_metadata(_csv(_valido()))
    assert v.ok and not v.erros and not v.avisos
    assert (v.n, v.features, v.algos, v.anotacoes) == (
        30, ["feature_a", "feature_b", "feature_c"], ["algo_x", "algo_y"], ["grupo"])


def test_sem_coluna_instances():
    v = validar_metadata(_csv(_valido().rename(columns={"instances": "id"})))
    assert not v.ok
    assert any("Falta a coluna **instances**" in e for e in v.erros)


def test_rotulo_repetido():
    df = _valido()
    df.loc[5, "instances"] = df.loc[3, "instances"]
    df.loc[9, "instances"] = df.loc[3, "instances"]
    v = validar_metadata(_csv(df))
    assert not v.ok
    [erro] = [e for e in v.erros if "repetidos" in e]
    assert "'i3'" in erro and "(1 distintos)" in erro


def test_rotulo_vazio():
    df = _valido()
    df.loc[4, "instances"] = ""
    v = validar_metadata(_csv(df))
    assert not v.ok
    assert any("sem rótulo" in e and "6" in e for e in v.erros)   # linha 6 do arquivo


def test_coluna_nao_numerica():
    df = _valido()
    df[["feature_b", "algo_y"]] = df[["feature_b", "algo_y"]].astype(object)
    df.loc[2, "feature_b"] = "abc"
    df.loc[7, "algo_y"] = "n/d"
    v = validar_metadata(_csv(df))
    assert not v.ok
    assert any("**feature_b** tem valores não numéricos: 'abc'" in e for e in v.erros)
    assert any("**algo_y** tem valores não numéricos: 'n/d'" in e for e in v.erros)


def test_so_duas_features():
    v = validar_metadata(_csv(_valido().drop(columns="feature_c")))
    assert not v.ok
    assert any("pelo menos 3 colunas **feature_" in e and "tem 2" in e for e in v.erros)


def test_um_algoritmo_so():
    v = validar_metadata(_csv(_valido().drop(columns="algo_y")))
    assert not v.ok
    assert any("pelo menos 2 colunas **algo_" in e and "tem 1" in e for e in v.erros)


def test_nan_reportado_por_coluna_sem_bloquear():
    df = _valido()
    df.loc[[1, 2, 3], "feature_a"] = np.nan
    df.loc[[4], "algo_x"] = np.nan
    v = validar_metadata(_csv(df))
    assert v.ok
    assert v.nan == {"feature_a": 3, "algo_x": 1}
    assert any("feature_a: 3" in a and "algo_x: 1" in a for a in v.avisos)


def test_algoritmo_todo_nan_bloqueia():
    df = _valido()
    df["algo_y"] = np.nan
    v = validar_metadata(_csv(df))
    assert not v.ok and any("Algoritmos sem nenhum valor" in e for e in v.erros)


def test_arquivo_vazio_e_binario():
    assert validar_metadata(b"").erros == ["O arquivo está vazio."]
    assert "UTF-8" in validar_metadata(b"\xff\xfe\x00instances").erros[0]


def test_colunas_com_mesmo_nome():
    v = validar_metadata(b"instances,feature_a,feature_a,feature_b,algo_x,algo_y\n1,1,1,1,1,1\n")
    assert any("mesmo nome" in e for e in v.erros)


@pytest.mark.parametrize("dataset", DATASETS)
def test_metadata_dos_datasets_do_projeto_e_valido(dataset):
    v = validar_metadata((PASTA_IS / dataset / "metadata.csv").read_bytes())
    assert v.ok, v.erros


# ---------------------------------------------------------- auxiliares
def test_annotations_invalido_e_valido():
    v = validar_metadata(_csv(_valido()))
    _, erros = validar_anotacoes(b"{nao e json", v)
    assert erros and "não é um JSON válido" in erros[0]
    _, erros = validar_anotacoes(json.dumps({"grupo": "texto"}).encode(), v)
    assert erros and "não é um de" in erros[0]
    _, erros = validar_anotacoes(json.dumps({"feature_a": "numerica"}).encode(), v)
    assert erros and "não é uma coluna de anotação" in erros[0]
    tipos, erros = validar_anotacoes(json.dumps({"grupo": "categorica"}).encode(), v)
    assert (tipos, erros) == ({"grupo": "categorica"}, [])


def test_feature_info_sem_colunas():
    v = validar_metadata(_csv(_valido()))
    assert validar_feature_info(b"feature,familia\na,x\n", v) == [
        "feature_info.csv sem as colunas family."]
    assert validar_feature_info(b"feature,family\na,x\n", v) == []


# ------------------------------------------------------- regra do PRELIM
@pytest.mark.parametrize("maior", [True, False])
@pytest.mark.parametrize("absoluto", [True, False])
def test_fracao_boas_igual_a_do_instancespace(maior, absoluto):
    """Mesma regra de compute_binary_performance do instancespace, com NaN,
    zeros e empates no melhor."""
    prelim = pytest.importorskip("instancespace.stages.prelim")
    from instancespace.data.options import GeneralOptions, PerformanceOptions

    rng = np.random.default_rng(1)
    y = rng.uniform(0, 1, size=(300, 4))
    y[rng.random(y.shape) < 0.05] = np.nan
    y[rng.random(y.shape) < 0.05] = 0.0
    y[:10, 1] = y[:10, 0]
    eps = 0.3 if absoluto else 0.15
    df = pd.DataFrame(y, columns=[f"algo_{k}" for k in "abcd"])
    esperado = prelim.compute_binary_performance(
        y, PerformanceOptions(max_perf=maior, abs_perf=absoluto, epsilon=eps,
                              beta_threshold=0.55),
        GeneralOptions.default()).y_bin.mean(axis=0)
    np.testing.assert_array_equal(fracao_boas(df, maior, absoluto, eps).to_numpy(), esperado)


@pytest.mark.parametrize("dataset", DATASETS)
def test_fracao_boas_igual_ao_algorithm_bin_gravado(dataset):
    pasta = PASTA_IS / dataset
    v = validar_metadata((pasta / "metadata.csv").read_bytes())
    perf = json.loads((pasta / "run_options.json").read_text())["perf"]
    fr = fracao_boas(linhas_do_prelim(v), perf["max_perf"], perf["abs_perf"], perf["epsilon"])
    gravado = pd.read_csv(pasta / "algorithm_bin.csv", index_col=0).mean()
    gravado.index = [c.removeprefix("algo_") for c in gravado.index]
    pd.testing.assert_series_equal(fr, gravado.reindex(fr.index), check_names=False)


def test_direcao_invertida_no_iris_sai_da_faixa():
    """Com limiar absoluto, inverter a direcao troca f por 1 - f (a menos dos
    empates em epsilon): no iris, quase tudo bom vira quase tudo ruim."""
    v = validar_metadata((PASTA_IS / "iris" / "metadata.csv").read_bytes())
    certo = fracao_boas(linhas_do_prelim(v), True, True, 0.5)
    invertido = fracao_boas(linhas_do_prelim(v), False, True, 0.5)
    assert (invertido < FRACAO_MIN).any()
    assert set(fora_da_faixa(invertido)) == set(fora_da_faixa(certo))
    np.testing.assert_allclose(certo + invertido, 1.0)


# ---------------------------------------------------------------- tempo
def test_tempo_estimado_reproduz_as_medidas_e_cresce():
    for n, (total, _) in TEMPOS_MEDIDOS.items():
        assert tempo_estimado(n, TEMPOS_ALGOS) == pytest.approx(total, rel=0.1)
    assert tempo_estimado(1000, 10) > tempo_estimado(1000, 6) > tempo_estimado(500, 6)
    assert tempo_estimado(0, 6) is None


# ------------------------------------------------------------ execucao
def test_nome_seguro_e_pasta_unica(tmp_path):
    assert execucao.nome_seguro("meu metadata (v2).csv") == "meu_metadata_v2_csv"
    assert execucao.nome_seguro("///") == "metadata"
    from datetime import datetime
    agora = datetime(2026, 9, 22, 12, 0, 0)
    a = execucao.nova_pasta("x", tmp_path, agora)
    b = execucao.nova_pasta("x", tmp_path, agora)
    assert a.name == "x_20260922-120000" and b.name == "x_20260922-120000_2"
    assert (a / execucao.ENTRADA).is_dir()


def test_subprocesso_com_erro_devolve_mensagem_sem_traceback(tmp_path):
    """Metadata que o instancespace recusa: o erro vem numa linha, o log fica em disco."""
    pasta = execucao.nova_pasta("quebrado", tmp_path)
    meta = execucao.gravar_entrada(
        pasta, b"instances,feature_a,feature_b,algo_x,algo_y\n1,1,2,0.1,0.2\n2,3,4,0.5,0.6\n")
    estagios = []
    exe = execucao.iniciar(meta, pasta, {}, ao_estagio=lambda _e, s: estagios.append(s))
    fim = time.time() + 120
    while not exe.terminou and time.time() < fim:
        time.sleep(0.2)
    assert exe.terminou and exe.ok is False
    assert "three features" in exe.erro and "\n" not in exe.erro
    assert "Traceback" in exe.log.read_text()        # o traceback so vai para o log
    assert not (pasta / "run_info.json").exists()


# --------------------------------------------------- bloco da interface
def test_bloco_novo_so_habilita_depois_da_escolha(tmp_path):
    from isaspace.ui.novo import NovoInstanceSpace

    bloco = NovoInstanceSpace(tmp_path, 260, ao_concluir=lambda _p: None)
    assert bloco.w_rodar.disabled
    bloco.w_meta.filename = "meu metadata.csv"
    bloco.w_meta.value = (PASTA_IS / "diabetes" / "metadata.csv").read_bytes()
    assert bloco.w_nome.value == "meu_metadata"
    assert bloco.w_rodar.disabled and "direção do desempenho" in bloco.falta.object
    bloco.w_direcao.value = "max"
    bloco.w_limiar.value = "abs"
    assert bloco.w_rodar.disabled and "ε" in bloco.falta.object
    bloco.w_eps.value = 0.5
    assert not bloco.w_rodar.disabled and bloco.falta.object == ""
    assert "Prévia" in bloco.previa.object and not bloco.previa_aviso.objects   # 70-78%
    bloco.w_k.value, bloco.w_usesim.value = 4, True
    assert bloco.opcoes() == {"perf": {"max_perf": True, "abs_perf": True, "epsilon": 0.5},
                              "sifted": {"k": 4}, "trace": {"use_sim": True}}
    bloco.w_ann.value = b'{"class": "texto"}'                    # tipo invalido bloqueia
    assert bloco.w_rodar.disabled
    assert "não é um de" in bloco.msg_validacao.objects[0].object
    bloco.w_ann.value = b'{"class": "categorica", "row_original": "identifier"}'
    assert not bloco.w_rodar.disabled
    assert any("Tempo estimado" in o.object for o in bloco.tempo.objects)
