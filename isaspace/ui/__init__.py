"""Interface do espaco de instancias.

REGRA ARQUITETURAL: nenhum modulo deste pacote importa pyispace, pyhard ou
sklearn. A interface le uma pasta de saida ISA no formato MATILDA
(resultados/isa/<nome>/) e a tabela por instancia (resultados/table_<nome>.csv)
e nada mais. O que faltar na pasta e gerado por scripts/ (run_isa_all.py,
build_data_space.py), nunca calculado aqui.
"""
