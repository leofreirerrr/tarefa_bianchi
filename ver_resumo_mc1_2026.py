import pandas as pd
from pathlib import Path

pasta = Path("resultados_mc1_2026")

arquivos = [
    "resumo_qualidade_mc1_2026.csv",
    "estrutura_mc1_2026.csv",
    "resumo_textual_mc1_2026.csv",
    "resumo_temporal_mc1_2026.csv",
    "frequencia_agent_id_mc1_2026.csv",
    "frequencia_agent_role_mc1_2026.csv",
    "frequencia_channel_mc1_2026.csv",
    "frequencia_message_type_mc1_2026.csv",
    "frequencia_declared_action_mc1_2026.csv",
    "frequencia_palavras_mc1_2026.csv",
    "frequencia_bigramas_mc1_2026.csv",
    "frequencia_trigramas_mc1_2026.csv",
    "resumo_outliers_qtd_palavras_mc1_2026.csv",
    "correlacao_textual_mc1_2026.csv"
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