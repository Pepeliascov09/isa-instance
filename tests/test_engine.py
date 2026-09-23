"""Validacao dos arquivos auxiliares pelo engine (sem rodar o pipeline).

run_instancespace le e valida annotations.json, degenerate_report.csv e
feature_info.csv antes do build; um erro ali tem de aparecer em segundos, nao
depois do PYTHIA. Precisa do .venv-isa (importa instancespace).
"""

import json
import sys
import time
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

engine = pytest.importorskip("isaspace.engine")

METADATA = (
    "instances,classe,peso,feature_a,feature_b,feature_c,algo_x,algo_y\n"
    + "\n".join(f"i{k},{'ab'[k % 2]},{k},{k * 0.1},{(k * 7) % 5},{k % 3},{0.1 * (k % 9)},{0.9 - 0.1 * (k % 9)}"
                for k in range(30))
    + "\n"
)


@pytest.fixture
def entrada(tmp_path):
    pasta = tmp_path / "entrada"
    pasta.mkdir()
    (pasta / "metadata.csv").write_text(METADATA)
    return pasta


def test_sem_auxiliares(entrada):
    presentes, tipos = engine._ler_auxiliares(entrada / "metadata.csv")
    assert presentes == {} and tipos == {}


def test_auxiliares_validos(entrada):
    (entrada / "annotations.json").write_text(json.dumps({"classe": "categorica", "peso": "identifier"}))
    (entrada / "degenerate_report.csv").write_text("feature,var_bruta,iqr,motivo\nd,0.0,0.0,constante\n")
    (entrada / "feature_info.csv").write_text("feature,family\na,f1\nb,f1\nc,f2\nd,f2\n")
    presentes, tipos = engine._ler_auxiliares(entrada / "metadata.csv")
    assert sorted(presentes) == ["annotations.json", "degenerate_report.csv", "feature_info.csv"]
    assert tipos == {"classe": "categorica", "peso": "identifier"}


@pytest.mark.parametrize("conteudo, erro", [
    ({"classe": "numerica"}, "não numéricos"),
    ({"peso": "texto"}, "não é um de"),
    ({"feature_a": "numerica"}, "não é uma coluna de anotação"),
    ({"inexistente": "numerica"}, "não é uma coluna de anotação"),
    (["classe"], "objeto"),
])
def test_annotations_json_invalido_falha_antes_de_rodar(entrada, tmp_path, conteudo, erro):
    (entrada / "annotations.json").write_text(json.dumps(conteudo))
    t0 = time.perf_counter()
    with pytest.raises(ValueError, match=erro):
        engine.run_instancespace(entrada / "metadata.csv", tmp_path / "saida")
    assert time.perf_counter() - t0 < 5
    assert not (tmp_path / "saida").exists()


def test_inteira_com_valores_fracionarios_e_erro(entrada):
    (entrada / "metadata.csv").write_text(METADATA.replace(",3,0.30", ",3.5,0.30"))
    (entrada / "annotations.json").write_text(json.dumps({"peso": "numerica_inteira"}))
    with pytest.raises(ValueError, match="não inteiros"):
        engine._ler_auxiliares(entrada / "metadata.csv")


def test_relatorio_sem_colunas_obrigatorias_e_erro(entrada):
    (entrada / "degenerate_report.csv").write_text("feature,motivo\nd,constante\n")
    with pytest.raises(ValueError, match="faltam as colunas"):
        engine._ler_auxiliares(entrada / "metadata.csv")
