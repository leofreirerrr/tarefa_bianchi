import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# ==========================================================
# CONFIGURAÇÕES
# ==========================================================

pasta_mc1 = Path("resultados_mc1_2026")
pasta_mc2 = Path("resultados_mc2_2026")

saida = Path("graficos_melhorados")
saida_mc1 = saída_mc1 = saida / "mc1"
saida_mc2 = saída_mc2 = saida / "mc2"

saida_mc1.mkdir(parents=True, exist_ok=True)
saida_mc2.mkdir(parents=True, exist_ok=True)


# ==========================================================
# FUNÇÕES AUXILIARES
# ==========================================================

def salvar_figura(caminho):
    plt.tight_layout()
    plt.savefig(caminho, dpi=300, bbox_inches="tight")
    plt.close()


def grafico_barras_horizontais(df, coluna_categoria, coluna_valor, titulo, xlabel, caminho, top_n=15):
    dados = df.head(top_n).copy()
    dados = dados.sort_values(coluna_valor, ascending=True)

    plt.figure(figsize=(10, 6))
    plt.barh(dados[coluna_categoria].astype(str), dados[coluna_valor])
    plt.title(titulo)
    plt.xlabel(xlabel)
    plt.ylabel("")
    salvar_figura(caminho)


def heatmap_matriz(matriz, titulo, caminho, xlabel="", ylabel=""):
    plt.figure(figsize=(10, 7))
    plt.imshow(matriz.values, aspect="auto")
    plt.colorbar(label="Frequência")
    plt.title(titulo)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(range(len(matriz.columns)), matriz.columns, rotation=45, ha="right")
    plt.yticks(range(len(matriz.index)), matriz.index)
    salvar_figura(caminho)


# ==========================================================
# MC1 - GRÁFICOS MELHORADOS
# ==========================================================

mc1_comms = pd.read_csv(pasta_mc1 / "mc1_communications_2026.csv", encoding="utf-8-sig")
mc1_comms["timestamp_dt"] = pd.to_datetime(mc1_comms["timestamp_dt"], errors="coerce")

freq_agent = pd.read_csv(pasta_mc1 / "frequencia_agent_id_mc1_2026.csv", encoding="utf-8-sig")
freq_channel = pd.read_csv(pasta_mc1 / "frequencia_channel_mc1_2026.csv", encoding="utf-8-sig")
freq_words = pd.read_csv(pasta_mc1 / "frequencia_palavras_mc1_2026.csv", encoding="utf-8-sig")


# 1. Linha temporal: mensagens por horário
mensagens_por_hora = (
    mc1_comms
    .groupby("timestamp_dt")
    .size()
    .reset_index(name="qtd_mensagens")
    .sort_values("timestamp_dt")
)

plt.figure(figsize=(12, 5))
plt.plot(mensagens_por_hora["timestamp_dt"], mensagens_por_hora["qtd_mensagens"], marker="o")
plt.title("MC1 - Evolução das mensagens ao longo do tempo")
plt.xlabel("Tempo")
plt.ylabel("Quantidade de mensagens")
plt.xticks(rotation=45, ha="right")
salvar_figura(saida_mc1 / "mc1_01_linha_mensagens_tempo.png")


# 2. Barras horizontais: mensagens por agente
grafico_barras_horizontais(
    freq_agent,
    "agent_id",
    "frequencia_absoluta",
    "MC1 - Agentes com mais mensagens",
    "Quantidade de mensagens",
    saida_mc1 / "mc1_02_barras_horizontais_agentes.png"
)


# 3. Barras horizontais: mensagens por canal
grafico_barras_horizontais(
    freq_channel,
    "channel",
    "frequencia_absoluta",
    "MC1 - Canais mais utilizados",
    "Quantidade de mensagens",
    saida_mc1 / "mc1_03_barras_horizontais_canais.png"
)


# 4. Histograma: tamanho das mensagens
plt.figure(figsize=(10, 5))
plt.hist(mc1_comms["qtd_palavras"], bins=35)
plt.title("MC1 - Distribuição do tamanho das mensagens")
plt.xlabel("Quantidade de palavras")
plt.ylabel("Frequência")
salvar_figura(saida_mc1 / "mc1_04_histograma_tamanho_mensagens.png")


# 5. Boxplot: tamanho das mensagens por canal
canais_principais = mc1_comms["channel"].value_counts().head(6).index
dados_box = [
    mc1_comms.loc[mc1_comms["channel"] == canal, "qtd_palavras"].dropna()
    for canal in canais_principais
]

plt.figure(figsize=(11, 6))
plt.boxplot(dados_box, labels=canais_principais, vert=True)
plt.title("MC1 - Distribuição do tamanho das mensagens por canal")
plt.xlabel("Canal")
plt.ylabel("Quantidade de palavras")
plt.xticks(rotation=45, ha="right")
salvar_figura(saida_mc1 / "mc1_05_boxplot_tamanho_por_canal.png")


# 6. Heatmap: agente x canal
matriz_agente_canal = pd.crosstab(mc1_comms["agent_id"], mc1_comms["channel"])
heatmap_matriz(
    matriz_agente_canal,
    "MC1 - Frequência de mensagens por agente e canal",
    saida_mc1 / "mc1_06_heatmap_agente_canal.png",
    xlabel="Canal",
    ylabel="Agente"
)


# 7. Dispersão: destinatários x tamanho da mensagem
plt.figure(figsize=(9, 6))
plt.scatter(mc1_comms["qtd_recipients"], mc1_comms["qtd_palavras"], alpha=0.35)
plt.title("MC1 - Relação entre destinatários e tamanho da mensagem")
plt.xlabel("Quantidade de destinatários")
plt.ylabel("Quantidade de palavras")
salvar_figura(saida_mc1 / "mc1_07_dispersao_destinatarios_tamanho.png")


# 8. Barras horizontais: palavras mais frequentes
grafico_barras_horizontais(
    freq_words,
    "palavra",
    "frequencia",
    "MC1 - Palavras mais frequentes nas comunicações",
    "Frequência",
    saida_mc1 / "mc1_08_barras_horizontais_palavras.png",
    top_n=20
)


# ==========================================================
# MC2 - GRÁFICOS MELHORADOS
# ==========================================================

mc2_eventos = pd.read_csv(pasta_mc2 / "mc2_events_2026.csv", encoding="utf-8-sig")
mc2_eventos["timestamp"] = pd.to_datetime(mc2_eventos["timestamp"], errors="coerce")

freq_short = pd.read_csv(pasta_mc2 / "frequencia_short_name_mc2_2026.csv", encoding="utf-8-sig")
freq_party_type = pd.read_csv(pasta_mc2 / "frequencia_party_type_mc2_2026.csv", encoding="utf-8-sig")
freq_party = pd.read_csv(pasta_mc2 / "frequencia_party_mc2_2026.csv", encoding="utf-8-sig")
freq_words_mc2 = pd.read_csv(pasta_mc2 / "frequencia_palavras_mc2_2026.csv", encoding="utf-8-sig")


# 1. Linha: eventos por dia
eventos_por_dia = (
    mc2_eventos
    .groupby(mc2_eventos["timestamp"].dt.date)
    .size()
    .reset_index(name="qtd_eventos")
)

plt.figure(figsize=(12, 5))
plt.plot(eventos_por_dia["timestamp"].astype(str), eventos_por_dia["qtd_eventos"], marker="o")
plt.title("MC2 - Evolução da quantidade de eventos por dia")
plt.xlabel("Data")
plt.ylabel("Quantidade de eventos")
plt.xticks(rotation=45, ha="right")
salvar_figura(saida_mc2 / "mc2_01_linha_eventos_por_dia.png")


# 2. Barras horizontais: tipos de evento
grafico_barras_horizontais(
    freq_short,
    "short_name",
    "frequencia_absoluta",
    "MC2 - Tipos de evento mais frequentes",
    "Quantidade de eventos",
    saida_mc2 / "mc2_02_barras_horizontais_tipos_evento.png",
    top_n=15
)


# 3. Barras horizontais: tipos de entidades
grafico_barras_horizontais(
    freq_party_type,
    "party_type",
    "frequencia_absoluta",
    "MC2 - Tipos de entidades envolvidas",
    "Frequência",
    saida_mc2 / "mc2_03_barras_horizontais_tipos_entidades.png",
    top_n=10
)


# 4. Barras horizontais: entidades mais frequentes
grafico_barras_horizontais(
    freq_party,
    "party",
    "frequencia_absoluta",
    "MC2 - Entidades mais frequentes nos eventos",
    "Frequência",
    saida_mc2 / "mc2_04_barras_horizontais_entidades.png",
    top_n=20
)


# 5. Histograma: quantidade de entidades envolvidas por evento
plt.figure(figsize=(10, 5))
plt.hist(mc2_eventos["qtd_parties"], bins=30)
plt.title("MC2 - Distribuição da quantidade de entidades por evento")
plt.xlabel("Quantidade de entidades envolvidas")
plt.ylabel("Frequência")
salvar_figura(saida_mc2 / "mc2_05_histograma_qtd_parties.png")


# 6. Boxplot: quantidade de entidades por tipo de evento
tipos_principais = mc2_eventos["short_name"].value_counts().head(8).index
dados_box_mc2 = [
    mc2_eventos.loc[mc2_eventos["short_name"] == tipo, "qtd_parties"].dropna()
    for tipo in tipos_principais
]

plt.figure(figsize=(12, 6))
plt.boxplot(dados_box_mc2, labels=tipos_principais, vert=True)
plt.title("MC2 - Distribuição de envolvidos por tipo de evento")
plt.xlabel("Tipo de evento")
plt.ylabel("Quantidade de envolvidos")
plt.xticks(rotation=45, ha="right")
salvar_figura(saida_mc2 / "mc2_06_boxplot_parties_por_tipo_evento.png")


# 7. Heatmap temporal: dia da semana x hora
mc2_eventos["dia_semana"] = mc2_eventos["timestamp"].dt.day_name()
mc2_eventos["hora_do_dia"] = mc2_eventos["timestamp"].dt.hour

ordem_dias = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

matriz_dia_hora = pd.crosstab(
    mc2_eventos["dia_semana"],
    mc2_eventos["hora_do_dia"]
)

matriz_dia_hora = matriz_dia_hora.reindex(ordem_dias)

plt.figure(figsize=(13, 6))
plt.imshow(matriz_dia_hora.values, aspect="auto")
plt.colorbar(label="Quantidade de eventos")
plt.title("MC2 - Concentração de eventos por dia da semana e hora")
plt.xlabel("Hora do dia")
plt.ylabel("Dia da semana")
plt.xticks(range(len(matriz_dia_hora.columns)), matriz_dia_hora.columns)
plt.yticks(range(len(matriz_dia_hora.index)), matriz_dia_hora.index)
salvar_figura(saida_mc2 / "mc2_07_heatmap_dia_hora.png")


# 8. Dispersão: envolvidos x tamanho textual
amostra = mc2_eventos.sample(n=min(6000, len(mc2_eventos)), random_state=42)

plt.figure(figsize=(9, 6))
plt.scatter(amostra["qtd_parties"], amostra["qtd_palavras_texto"], alpha=0.25)
plt.title("MC2 - Relação entre envolvidos e tamanho textual")
plt.xlabel("Quantidade de entidades envolvidas")
plt.ylabel("Quantidade de palavras no texto")
salvar_figura(saida_mc2 / "mc2_08_dispersao_parties_texto.png")


# 9. Barras horizontais: palavras mais frequentes
grafico_barras_horizontais(
    freq_words_mc2,
    "palavra",
    "frequencia",
    "MC2 - Palavras mais frequentes nos textos dos eventos",
    "Frequência",
    saida_mc2 / "mc2_09_barras_horizontais_palavras.png",
    top_n=20
)


# 10. Grafo simples do organograma
try:
    import networkx as nx

    org_nodes = pd.read_csv(pasta_mc2 / "mc2_org_nodes_2026.csv", encoding="utf-8-sig")
    org_edges = pd.read_csv(pasta_mc2 / "mc2_org_edges_2026.csv", encoding="utf-8-sig")

    G = nx.DiGraph()

    for _, row in org_nodes.iterrows():
        G.add_node(row["id"], label=row.get("label", row["id"]), node_type=row.get("type", ""))

    for _, row in org_edges.iterrows():
        G.add_edge(row["source"], row["target"])

    plt.figure(figsize=(14, 10))
    pos = nx.spring_layout(G, seed=42, k=0.55)

    nx.draw_networkx_edges(G, pos, arrows=True, alpha=0.35)
    nx.draw_networkx_nodes(G, pos, node_size=350, alpha=0.85)

    labels = {
        node: str(data.get("label", node))[:18]
        for node, data in G.nodes(data=True)
    }

    nx.draw_networkx_labels(G, pos, labels=labels, font_size=7)

    plt.title("MC2 - Organograma da Tenant Thread em forma de grafo")
    plt.axis("off")
    salvar_figura(saida_mc2 / "mc2_10_grafo_organograma.png")

except ImportError:
    print("NetworkX não instalado. Para gerar o grafo do organograma, rode: pip install networkx")


print("Gráficos melhorados gerados com sucesso!")
print(f"MC1: {saida_mc1}")
print(f"MC2: {saida_mc2}")