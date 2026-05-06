import pandas as pd
from pathlib import Path

pasta = Path("resultados_mc2_2026")

arquivos = [
    "identificacao_mc2_2026.csv",
    "resumo_qualidade_mc2_2026.csv",
    "estrutura_mc2_2026.csv",
    "resumo_temporal_mc2_2026.csv",
    "frequencia_short_name_mc2_2026.csv",
    "frequencia_party_type_mc2_2026.csv",
    "frequencia_party_mc2_2026.csv",
    "frequencia_org_node_type_mc2_2026.csv",
    "frequencia_org_relation_mc2_2026.csv",
    "resumo_textual_mc2_2026.csv",
    "frequencia_palavras_mc2_2026.csv",
    "frequencia_bigramas_mc2_2026.csv",
    "frequencia_trigramas_mc2_2026.csv",
    "resumo_outliers_qtd_parties_mc2_2026.csv",
    "resumo_outliers_texto_mc2_2026.csv",
    "resumo_outliers_eventos_hora_mc2_2026.csv",
    "correlacao_mc2_2026.csv"
]

for arquivo in arquivos:
    caminho = pasta / arquivo

    print("\n" + "=" * 100)
    print(arquivo)
    print("=" * 100)

    if caminho.exists():
        df = pd.read_csv(caminho, encoding="utf-8-sig")
        print(df.head(20).to_string(index=False))
    else:
        print("Arquivo não encontrado.")