import pandas as pd
from pathlib import Path

pasta = Path("resultados_mc1")

arquivos = [
    "identificacao_mc1.csv",
    "estrutura_mc1_nodes.csv",
    "estrutura_mc1_links.csv",
    "resumo_qualidade_mc1.csv",
    "faltantes_mc1_nodes.csv",
    "faltantes_mc1_links.csv",
    "frequencia_node_type_mc1.csv",
    "frequencia_edge_type_mc1.csv",
    "frequencia_genre_mc1.csv",
    "resumo_temporal_mc1.csv",
    "top_20_nos_por_grau_mc1.csv",
    "resumo_outliers_grau_mc1.csv",
    "correlacao_mc1.csv"
]

for arquivo in arquivos:
    caminho = pasta / arquivo

    print("\n" + "=" * 100)
    print(arquivo)
    print("=" * 100)

    if caminho.exists():
        df = pd.read_csv(caminho)
        print(df.head(20).to_string(index=False))
    else:
        print("Arquivo não encontrado.")