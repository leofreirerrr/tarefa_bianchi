import json
import re
from pathlib import Path
from collections import Counter

import pandas as pd
import matplotlib.pyplot as plt


# ==========================================================
# CONFIGURAÇÕES
# ==========================================================

arquivo_json = Path(r"VAST_Challenge_2026_MC1\VAST_Challenge_2026_MC1\MC1_final_00.json")

pasta_saida = Path("resultados_mc1_2026")
pasta_graficos = pasta_saida / "graficos"

pasta_saida.mkdir(exist_ok=True)
pasta_graficos.mkdir(exist_ok=True)


# ==========================================================
# FUNÇÕES AUXILIARES
# ==========================================================

def salvar_csv(df, nome):
    df.to_csv(pasta_saida / nome, index=False, encoding="utf-8-sig")


def texto_ou_vazio(valor):
    if valor is None:
        return ""
    return str(valor)


def juntar_lista(valor):
    if isinstance(valor, list):
        return "; ".join(map(str, valor))
    if valor is None:
        return ""
    return str(valor)


def contar_frases(texto):
    texto = texto_ou_vazio(texto).strip()
    if not texto:
        return 0
    frases = re.split(r"[.!?]+", texto)
    frases = [f.strip() for f in frases if f.strip()]
    return len(frases)


def limpar_texto(texto):
    texto = texto_ou_vazio(texto).lower()
    texto = re.sub(r"http\S+|www\S+", " ", texto)
    texto = re.sub(r"[^a-zA-ZÀ-ÿ0-9\s]", " ", texto)
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto


stopwords = {
    "the", "a", "an", "and", "or", "but", "if", "then", "else", "of", "to", "in",
    "on", "for", "with", "as", "by", "at", "from", "is", "are", "was", "were",
    "be", "been", "being", "this", "that", "these", "those", "it", "its", "we",
    "our", "you", "your", "they", "their", "i", "me", "my", "he", "she", "his",
    "her", "them", "not", "no", "yes", "do", "does", "did", "have", "has", "had",
    "will", "would", "can", "could", "should", "about", "into", "more", "need",
    "just", "now", "today", "there", "here", "all", "any", "so", "than"
}


def palavras_do_texto(texto, remover_stopwords=True):
    palavras = limpar_texto(texto).split()
    if remover_stopwords:
        palavras = [p for p in palavras if p not in stopwords and len(p) > 1]
    return palavras


def classificar_tipo(serie):
    if pd.api.types.is_bool_dtype(serie):
        return "booleana"
    elif pd.api.types.is_numeric_dtype(serie):
        return "numérica"
    elif pd.api.types.is_datetime64_any_dtype(serie):
        return "temporal"
    else:
        return "categórica/textual"


def tabela_estrutura(df, nome_tabela):
    return pd.DataFrame({
        "tabela": nome_tabela,
        "variavel": df.columns,
        "tipo_pandas": df.dtypes.astype(str),
        "tipo_interpretado": [classificar_tipo(df[col]) for col in df.columns],
        "qtd_valores_unicos": [df[col].nunique(dropna=True) for col in df.columns],
        "qtd_faltantes": [df[col].isna().sum() for col in df.columns],
        "percentual_faltantes": [(df[col].isna().mean() * 100).round(2) for col in df.columns]
    })


def tabela_faltantes(df, nome_tabela):
    return pd.DataFrame({
        "tabela": nome_tabela,
        "variavel": df.columns,
        "qtd_faltantes": df.isna().sum().values,
        "percentual_faltantes": (df.isna().mean() * 100).round(2).values
    })


def estatisticas_numericas(df, nome_tabela):
    numericas = df.select_dtypes(include=["int64", "float64", "int32", "float32"])

    if numericas.empty:
        return pd.DataFrame()

    return pd.DataFrame({
        "tabela": nome_tabela,
        "variavel": numericas.columns,
        "qtd_validos": numericas.count().values,
        "media": numericas.mean().values,
        "mediana": numericas.median().values,
        "moda": [
            numericas[col].mode().iloc[0] if not numericas[col].mode().empty else None
            for col in numericas.columns
        ],
        "minimo": numericas.min().values,
        "maximo": numericas.max().values,
        "amplitude": (numericas.max() - numericas.min()).values,
        "q1": numericas.quantile(0.25).values,
        "q2": numericas.quantile(0.50).values,
        "q3": numericas.quantile(0.75).values,
        "desvio_padrao": numericas.std().values,
        "variancia": numericas.var().values
    })


def detectar_outliers_iqr(df, coluna):
    serie = df[coluna].dropna()

    q1 = serie.quantile(0.25)
    q3 = serie.quantile(0.75)
    iqr = q3 - q1

    limite_inferior = q1 - 1.5 * iqr
    limite_superior = q3 + 1.5 * iqr

    outliers = df[(df[coluna] < limite_inferior) | (df[coluna] > limite_superior)]

    resumo = pd.DataFrame({
        "variavel": [coluna],
        "q1": [q1],
        "q3": [q3],
        "iqr": [iqr],
        "limite_inferior": [limite_inferior],
        "limite_superior": [limite_superior],
        "qtd_outliers": [len(outliers)]
    })

    return resumo, outliers


# ==========================================================
# LEITURA DO JSON
# ==========================================================

with open(arquivo_json, "r", encoding="utf-8") as arquivo:
    dados = json.load(arquivo)

rounds = dados["rounds"]


# ==========================================================
# 1. CONSTRUÇÃO DAS TABELAS
# ==========================================================

linhas_rounds = []
linhas_communications = []
linhas_participants = []
linhas_media_events = []
linhas_news = []

for idx, rodada in enumerate(rounds):
    hour = rodada.get("hour")
    contexto = rodada.get("environment_context", {})

    media_events = contexto.get("media_events", [])
    news = contexto.get("news", [])
    external_actions = contexto.get("external_actor_actions", [])
    alerts = contexto.get("social_manager_alerts", [])
    unavailable = contexto.get("agents_unavailable", [])
    deadlines = contexto.get("critical_deadlines", [])

    linhas_rounds.append({
        "round_index": idx,
        "hour": hour,
        "event_headline": contexto.get("event_headline"),
        "event_narrative": contexto.get("event_narrative"),
        "qtd_communications": len(rodada.get("communications", [])),
        "qtd_participants": len(rodada.get("participants", [])),
        "qtd_media_events": len(media_events),
        "qtd_news": len(news),
        "qtd_external_actor_actions": len(external_actions),
        "qtd_social_manager_alerts": len(alerts),
        "qtd_agents_unavailable": len(unavailable),
        "qtd_critical_deadlines": len(deadlines),
        "market_snapshot": json.dumps(contexto.get("market_snapshot", {}), ensure_ascii=False),
        "social_state": json.dumps(contexto.get("social_state", {}), ensure_ascii=False)
    })

    for comm in rodada.get("communications", []):
        estado = comm.get("internal_state") or {}

        content = comm.get("content")
        if content is None:
            content = comm.get("message_text")

        linhas_communications.append({
            "round_index": idx,
            "hour": hour,
            "message_id": comm.get("message_id"),
            "agent_id": comm.get("agent_id"),
            "agent_role": comm.get("agent_role"),
            "agent_label": comm.get("agent_label"),
            "channel": comm.get("channel"),
            "recipients": juntar_lista(comm.get("recipients")),
            "qtd_recipients": len(comm.get("recipients", [])) if isinstance(comm.get("recipients"), list) else 0,
            "message_type": comm.get("message_type"),
            "responding_to": comm.get("responding_to"),
            "content": content,
            "timestamp": comm.get("timestamp"),
            "internal_reacting": estado.get("reacting"),
            "internal_rationalizing": estado.get("rationalizing"),
            "internal_deliberating": estado.get("deliberating")
        })

    for participante in rodada.get("participants", []):
        metadata = participante.get("agent_round_metadata", {})

        linhas_participants.append({
            "round_index": idx,
            "hour": hour,
            "agent_id": participante.get("agent_id"),
            "agent_role": participante.get("agent_role"),
            "agent_label": participante.get("agent_label"),
            "declared_action": participante.get("declared_action"),
            "sentiment_at_turn": metadata.get("sentiment_at_turn"),
            "action_classification": metadata.get("action_classification")
        })

    for evento in media_events:
        if isinstance(evento, dict):
            linha = {"round_index": idx, "hour": hour}
            linha.update(evento)
            linhas_media_events.append(linha)
        else:
            linhas_media_events.append({
                "round_index": idx,
                "hour": hour,
                "media_event": str(evento)
            })

    for item_news in news:
        if isinstance(item_news, dict):
            linha = {"round_index": idx, "hour": hour}
            linha.update(item_news)
            linhas_news.append(linha)
        else:
            linhas_news.append({
                "round_index": idx,
                "hour": hour,
                "news": str(item_news)
            })


df_rounds = pd.DataFrame(linhas_rounds)
df_communications = pd.DataFrame(linhas_communications)
df_participants = pd.DataFrame(linhas_participants)
df_media_events = pd.DataFrame(linhas_media_events)
df_news = pd.DataFrame(linhas_news)

# Conversão temporal
for df in [df_rounds, df_communications, df_participants, df_media_events, df_news]:
    if not df.empty and "hour" in df.columns:
        df["hour_dt"] = pd.to_datetime(df["hour"], errors="coerce")

if not df_communications.empty and "timestamp" in df_communications.columns:
    df_communications["timestamp_dt"] = pd.to_datetime(df_communications["timestamp"], errors="coerce")


# ==========================================================
# 2. MÉTRICAS TEXTUAIS DAS COMUNICAÇÕES
# ==========================================================

df_communications["content"] = df_communications["content"].fillna("").astype(str)

df_communications["qtd_caracteres"] = df_communications["content"].apply(len)
df_communications["qtd_palavras"] = df_communications["content"].apply(lambda x: len(limpar_texto(x).split()))
df_communications["qtd_frases"] = df_communications["content"].apply(contar_frases)
df_communications["media_palavras_por_frase"] = df_communications.apply(
    lambda row: row["qtd_palavras"] / row["qtd_frases"] if row["qtd_frases"] > 0 else 0,
    axis=1
)
df_communications["texto_vazio"] = df_communications["content"].str.strip() == ""
df_communications["texto_quase_vazio"] = df_communications["qtd_palavras"] <= 3
df_communications["possui_link"] = df_communications["content"].str.contains(r"http\S+|www\S+", regex=True, na=False)
df_communications["possui_html"] = df_communications["content"].str.contains(r"<[^>]+>", regex=True, na=False)
df_communications["possui_hashtag"] = df_communications["content"].str.contains(r"#\w+", regex=True, na=False)
df_communications["possui_mencao"] = df_communications["content"].str.contains(r"@\w+", regex=True, na=False)
df_communications["possui_caracter_estranho"] = df_communications["content"].str.contains(r"[�]", regex=True, na=False)

todas_palavras = []
for texto in df_communications["content"]:
    todas_palavras.extend(palavras_do_texto(texto, remover_stopwords=True))

contador_palavras = Counter(todas_palavras)
total_palavras_sem_stopwords = sum(contador_palavras.values())
palavras_unicas = len(contador_palavras)
hapax = sum(1 for palavra, freq in contador_palavras.items() if freq == 1)

resumo_textual = pd.DataFrame({
    "qtd_textos": [len(df_communications)],
    "qtd_textos_vazios": [df_communications["texto_vazio"].sum()],
    "qtd_textos_quase_vazios": [df_communications["texto_quase_vazio"].sum()],
    "total_palavras": [df_communications["qtd_palavras"].sum()],
    "total_caracteres": [df_communications["qtd_caracteres"].sum()],
    "media_palavras_por_texto": [df_communications["qtd_palavras"].mean()],
    "mediana_palavras_por_texto": [df_communications["qtd_palavras"].median()],
    "min_palavras": [df_communications["qtd_palavras"].min()],
    "max_palavras": [df_communications["qtd_palavras"].max()],
    "q1_palavras": [df_communications["qtd_palavras"].quantile(0.25)],
    "q3_palavras": [df_communications["qtd_palavras"].quantile(0.75)],
    "media_caracteres_por_texto": [df_communications["qtd_caracteres"].mean()],
    "media_frases_por_texto": [df_communications["qtd_frases"].mean()],
    "media_palavras_por_frase": [df_communications["media_palavras_por_frase"].mean()],
    "palavras_unicas_sem_stopwords": [palavras_unicas],
    "total_palavras_sem_stopwords": [total_palavras_sem_stopwords],
    "razao_tipo_token": [palavras_unicas / total_palavras_sem_stopwords if total_palavras_sem_stopwords > 0 else 0],
    "hapax_legomena": [hapax],
    "proporcao_hapax": [hapax / palavras_unicas if palavras_unicas > 0 else 0],
    "qtd_textos_com_link": [df_communications["possui_link"].sum()],
    "qtd_textos_com_html": [df_communications["possui_html"].sum()],
    "qtd_textos_com_hashtag": [df_communications["possui_hashtag"].sum()],
    "qtd_textos_com_mencao": [df_communications["possui_mencao"].sum()],
    "qtd_textos_com_caracter_estranho": [df_communications["possui_caracter_estranho"].sum()],
    "qtd_textos_duplicados": [df_communications["content"].duplicated().sum()]
})

frequencia_palavras = pd.DataFrame(
    contador_palavras.most_common(50),
    columns=["palavra", "frequencia"]
)

# Bigramas e trigramas
bigrams = []
trigrams = []

for texto in df_communications["content"]:
    palavras = palavras_do_texto(texto, remover_stopwords=True)

    for i in range(len(palavras) - 1):
        bigrams.append((palavras[i], palavras[i + 1]))

    for i in range(len(palavras) - 2):
        trigrams.append((palavras[i], palavras[i + 1], palavras[i + 2]))

frequencia_bigramas = pd.DataFrame(
    [(" ".join(k), v) for k, v in Counter(bigrams).most_common(50)],
    columns=["bigrama", "frequencia"]
)

frequencia_trigramas = pd.DataFrame(
    [(" ".join(k), v) for k, v in Counter(trigrams).most_common(50)],
    columns=["trigrama", "frequencia"]
)


# ==========================================================
# 3. ESTRUTURA, QUALIDADE E ESTATÍSTICAS
# ==========================================================

estrutura = pd.concat([
    tabela_estrutura(df_rounds, "MC1 - Rounds"),
    tabela_estrutura(df_communications, "MC1 - Communications"),
    tabela_estrutura(df_participants, "MC1 - Participants"),
    tabela_estrutura(df_media_events, "MC1 - Media Events") if not df_media_events.empty else pd.DataFrame(),
    tabela_estrutura(df_news, "MC1 - News") if not df_news.empty else pd.DataFrame()
], ignore_index=True)

faltantes = pd.concat([
    tabela_faltantes(df_rounds, "MC1 - Rounds"),
    tabela_faltantes(df_communications, "MC1 - Communications"),
    tabela_faltantes(df_participants, "MC1 - Participants"),
    tabela_faltantes(df_media_events, "MC1 - Media Events") if not df_media_events.empty else pd.DataFrame(),
    tabela_faltantes(df_news, "MC1 - News") if not df_news.empty else pd.DataFrame()
], ignore_index=True)

resumo_qualidade = pd.DataFrame({
    "tabela": [
        "MC1 - Rounds",
        "MC1 - Communications",
        "MC1 - Participants",
        "MC1 - Media Events",
        "MC1 - News"
    ],
    "total_linhas": [
        len(df_rounds),
        len(df_communications),
        len(df_participants),
        len(df_media_events),
        len(df_news)
    ],
    "total_colunas": [
        df_rounds.shape[1],
        df_communications.shape[1],
        df_participants.shape[1],
        df_media_events.shape[1] if not df_media_events.empty else 0,
        df_news.shape[1] if not df_news.empty else 0
    ],
    "linhas_incompletas": [
        df_rounds.isna().any(axis=1).sum(),
        df_communications.isna().any(axis=1).sum(),
        df_participants.isna().any(axis=1).sum(),
        df_media_events.isna().any(axis=1).sum() if not df_media_events.empty else 0,
        df_news.isna().any(axis=1).sum() if not df_news.empty else 0
    ],
    "registros_duplicados": [
        df_rounds.duplicated().sum(),
        df_communications.duplicated().sum(),
        df_participants.duplicated().sum(),
        df_media_events.duplicated().sum() if not df_media_events.empty else 0,
        df_news.duplicated().sum() if not df_news.empty else 0
    ]
})

estatisticas = pd.concat([
    estatisticas_numericas(df_rounds, "MC1 - Rounds"),
    estatisticas_numericas(df_communications, "MC1 - Communications"),
    estatisticas_numericas(df_participants, "MC1 - Participants"),
    estatisticas_numericas(df_media_events, "MC1 - Media Events") if not df_media_events.empty else pd.DataFrame(),
    estatisticas_numericas(df_news, "MC1 - News") if not df_news.empty else pd.DataFrame()
], ignore_index=True)


# ==========================================================
# 4. FREQUÊNCIAS CATEGÓRICAS E CRUZAMENTOS
# ==========================================================

def frequencia_categoria(df, coluna, nome_arquivo):
    if df.empty or coluna not in df.columns:
        return pd.DataFrame()

    serie = df[coluna].fillna("Não informado").astype(str)

    freq_abs = serie.value_counts(dropna=False)
    freq_rel = serie.value_counts(normalize=True, dropna=False) * 100

    tabela = pd.DataFrame({
        coluna: freq_abs.index.astype(str),
        "frequencia_absoluta": freq_abs.values,
        "frequencia_relativa_percentual": freq_rel.round(2).values
    })

    salvar_csv(tabela, nome_arquivo)
    return tabela


freq_agent_id = frequencia_categoria(df_communications, "agent_id", "frequencia_agent_id_mc1_2026.csv")
freq_agent_role = frequencia_categoria(df_communications, "agent_role", "frequencia_agent_role_mc1_2026.csv")
freq_channel = frequencia_categoria(df_communications, "channel", "frequencia_channel_mc1_2026.csv")
freq_message_type = frequencia_categoria(df_communications, "message_type", "frequencia_message_type_mc1_2026.csv")
freq_declared_action = frequencia_categoria(df_participants, "declared_action", "frequencia_declared_action_mc1_2026.csv")
freq_action_classification = frequencia_categoria(df_participants, "action_classification", "frequencia_action_classification_mc1_2026.csv")

if "agent_id" in df_communications.columns and "channel" in df_communications.columns:
    cruzamento_agente_canal = pd.crosstab(df_communications["agent_id"], df_communications["channel"])
    cruzamento_agente_canal.to_csv(pasta_saida / "cruzamento_agente_canal_mc1_2026.csv", encoding="utf-8-sig")

if "agent_id" in df_communications.columns and "qtd_palavras" in df_communications.columns:
    media_texto_por_agente = (
        df_communications
        .groupby("agent_id")[["qtd_palavras", "qtd_caracteres", "qtd_frases"]]
        .mean()
        .sort_values("qtd_palavras", ascending=False)
    )
    media_texto_por_agente.to_csv(pasta_saida / "media_texto_por_agente_mc1_2026.csv", encoding="utf-8-sig")

if "channel" in df_communications.columns and "qtd_palavras" in df_communications.columns:
    media_texto_por_canal = (
        df_communications
        .groupby("channel")[["qtd_palavras", "qtd_caracteres", "qtd_frases"]]
        .mean()
        .sort_values("qtd_palavras", ascending=False)
    )
    media_texto_por_canal.to_csv(pasta_saida / "media_texto_por_canal_mc1_2026.csv", encoding="utf-8-sig")


# ==========================================================
# 5. TEMPORAL
# ==========================================================

mensagens_por_hora = (
    df_communications
    .groupby("hour_dt")
    .size()
    .reset_index(name="qtd_mensagens")
    .sort_values("hour_dt")
)

mensagens_por_hora.to_csv(pasta_saida / "mensagens_por_hora_mc1_2026.csv", index=False, encoding="utf-8-sig")

resumo_temporal = pd.DataFrame({
    "inicio": [df_communications["timestamp_dt"].min()],
    "fim": [df_communications["timestamp_dt"].max()],
    "qtd_horarios_distintos": [df_communications["hour_dt"].nunique()],
    "qtd_mensagens": [len(df_communications)]
})

salvar_csv(resumo_temporal, "resumo_temporal_mc1_2026.csv")


# ==========================================================
# 6. OUTLIERS EM TAMANHO DE TEXTO
# ==========================================================

resumo_outliers_palavras, outliers_palavras = detectar_outliers_iqr(df_communications, "qtd_palavras")
salvar_csv(resumo_outliers_palavras, "resumo_outliers_qtd_palavras_mc1_2026.csv")
salvar_csv(outliers_palavras, "outliers_qtd_palavras_mc1_2026.csv")


# ==========================================================
# 7. CORRELAÇÃO
# ==========================================================

colunas_corr = ["qtd_recipients", "qtd_caracteres", "qtd_palavras", "qtd_frases", "media_palavras_por_frase"]
correlacao = df_communications[colunas_corr].corr()
correlacao.to_csv(pasta_saida / "correlacao_textual_mc1_2026.csv", encoding="utf-8-sig")


# ==========================================================
# 8. SALVAR CSVs PRINCIPAIS
# ==========================================================

salvar_csv(df_rounds, "mc1_rounds_2026.csv")
salvar_csv(df_communications, "mc1_communications_2026.csv")
salvar_csv(df_participants, "mc1_participants_2026.csv")
salvar_csv(df_media_events, "mc1_media_events_2026.csv")
salvar_csv(df_news, "mc1_news_2026.csv")

salvar_csv(estrutura, "estrutura_mc1_2026.csv")
salvar_csv(faltantes, "faltantes_mc1_2026.csv")
salvar_csv(resumo_qualidade, "resumo_qualidade_mc1_2026.csv")
salvar_csv(estatisticas, "estatisticas_numericas_mc1_2026.csv")
salvar_csv(resumo_textual, "resumo_textual_mc1_2026.csv")
salvar_csv(frequencia_palavras, "frequencia_palavras_mc1_2026.csv")
salvar_csv(frequencia_bigramas, "frequencia_bigramas_mc1_2026.csv")
salvar_csv(frequencia_trigramas, "frequencia_trigramas_mc1_2026.csv")


# ==========================================================
# 9. GRÁFICOS
# ==========================================================

# Gráfico 1 - mensagens por hora
plt.figure(figsize=(12, 6))
plt.plot(mensagens_por_hora["hour_dt"], mensagens_por_hora["qtd_mensagens"], marker="o")
plt.title("MC1 - Quantidade de mensagens por horário")
plt.xlabel("Horário")
plt.ylabel("Quantidade de mensagens")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(pasta_graficos / "grafico_1_mensagens_por_hora.png", dpi=300)
plt.close()

# Gráfico 2 - mensagens por agente
if not freq_agent_id.empty:
    dados_plot = freq_agent_id.head(15)

    plt.figure(figsize=(10, 6))
    plt.bar(dados_plot["agent_id"], dados_plot["frequencia_absoluta"])
    plt.title("MC1 - Mensagens por agente")
    plt.xlabel("Agente")
    plt.ylabel("Quantidade de mensagens")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(pasta_graficos / "grafico_2_mensagens_por_agente.png", dpi=300)
    plt.close()

# Gráfico 3 - mensagens por canal
if not freq_channel.empty:
    dados_plot = freq_channel.head(15)

    plt.figure(figsize=(10, 6))
    plt.bar(dados_plot["channel"], dados_plot["frequencia_absoluta"])
    plt.title("MC1 - Mensagens por canal")
    plt.xlabel("Canal")
    plt.ylabel("Quantidade de mensagens")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(pasta_graficos / "grafico_3_mensagens_por_canal.png", dpi=300)
    plt.close()

# Gráfico 4 - tamanho das mensagens em palavras
plt.figure(figsize=(10, 6))
plt.hist(df_communications["qtd_palavras"], bins=30)
plt.title("MC1 - Distribuição do tamanho das mensagens")
plt.xlabel("Quantidade de palavras")
plt.ylabel("Frequência")
plt.tight_layout()
plt.savefig(pasta_graficos / "grafico_4_histograma_tamanho_mensagens.png", dpi=300)
plt.close()

# Gráfico 5 - boxplot do tamanho das mensagens
plt.figure(figsize=(10, 5))
plt.boxplot(df_communications["qtd_palavras"], vert=False)
plt.title("MC1 - Boxplot do tamanho das mensagens")
plt.xlabel("Quantidade de palavras")
plt.tight_layout()
plt.savefig(pasta_graficos / "grafico_5_boxplot_tamanho_mensagens.png", dpi=300)
plt.close()

# Gráfico 6 - palavras mais frequentes
if not frequencia_palavras.empty:
    dados_plot = frequencia_palavras.head(20)

    plt.figure(figsize=(12, 6))
    plt.bar(dados_plot["palavra"], dados_plot["frequencia"])
    plt.title("MC1 - Palavras mais frequentes nas mensagens")
    plt.xlabel("Palavra")
    plt.ylabel("Frequência")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(pasta_graficos / "grafico_6_palavras_mais_frequentes.png", dpi=300)
    plt.close()

# Gráfico 7 - ações declaradas
if not freq_declared_action.empty:
    dados_plot = freq_declared_action.head(15).copy()
    dados_plot["declared_action"] = dados_plot["declared_action"].fillna("Não informado").astype(str)

    plt.figure(figsize=(10, 6))
    plt.bar(dados_plot["declared_action"], dados_plot["frequencia_absoluta"])
    plt.title("MC1 - Ações declaradas pelos participantes")
    plt.xlabel("Ação declarada")
    plt.ylabel("Frequência")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(pasta_graficos / "grafico_7_acoes_declaradas.png", dpi=300)
    plt.close()

# Gráfico 8 - matriz de correlação
plt.figure(figsize=(8, 6))
plt.imshow(correlacao)
plt.colorbar()
plt.xticks(range(len(correlacao.columns)), correlacao.columns, rotation=45, ha="right")
plt.yticks(range(len(correlacao.columns)), correlacao.columns)
plt.title("MC1 - Correlação entre métricas textuais")
plt.tight_layout()
plt.savefig(pasta_graficos / "grafico_8_correlacao_textual.png", dpi=300)
plt.close()


# ==========================================================
# 10. RESUMO NO TERMINAL
# ==========================================================

print("\nAnálise do MC1 2026 concluída!")
print(f"Rounds: {len(df_rounds)}")
print(f"Comunicações: {len(df_communications)}")
print(f"Participantes: {len(df_participants)}")
print(f"Eventos de mídia: {len(df_media_events)}")
print(f"Notícias: {len(df_news)}")
print(f"Início: {resumo_temporal['inicio'].iloc[0]}")
print(f"Fim: {resumo_temporal['fim'].iloc[0]}")
print(f"Resultados salvos em: {pasta_saida}")
print(f"Gráficos salvos em: {pasta_graficos}")