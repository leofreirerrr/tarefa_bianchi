import json
import re
from pathlib import Path
from collections import Counter

import pandas as pd
import matplotlib.pyplot as plt


# ==========================================================
# CONFIGURAÇÕES
# ==========================================================

pasta_mc2 = Path(r"VAST_Challenge_2026_MC2\VAST_Challenge_2026_MC2")

arquivo_eventos = pasta_mc2 / "MC2 data.json"
arquivo_org = pasta_mc2 / "org_chart.json"

pasta_saida = Path("resultados_mc2_2026")
pasta_graficos = pasta_saida / "graficos"

pasta_saida.mkdir(exist_ok=True)
pasta_graficos.mkdir(exist_ok=True)


# ==========================================================
# FUNÇÕES AUXILIARES
# ==========================================================


def salvar_csv(df, nome):
    caminho = pasta_saida / nome

    df_copia = df.copy()

    # Evita erro ao salvar listas/dicts no CSV
    for col in df_copia.columns:
        df_copia[col] = df_copia[col].apply(
            lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, (dict, list)) else x
        )

    df_copia.to_csv(caminho, index=False, encoding="utf-8-sig")

def tornar_hashable(valor):
    if isinstance(valor, (list, dict, set)):
        return json.dumps(valor, ensure_ascii=False, sort_keys=True)
    return valor


def contar_unicos_seguro(serie):
    return serie.apply(tornar_hashable).nunique(dropna=True)


def contar_duplicados_seguro(df):
    df_copia = df.copy()

    for col in df_copia.columns:
        df_copia[col] = df_copia[col].apply(tornar_hashable)

    return df_copia.duplicated().sum()


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
        "qtd_valores_unicos": [contar_unicos_seguro(df[col]) for col in df.columns],
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


def extrair_tipo_entidade(valor):
    if not isinstance(valor, str):
        return "desconhecido"

    if ":" in valor:
        return valor.split(":", 1)[0]

    return "desconhecido"


def limpar_texto(texto):
    texto = "" if texto is None else str(texto)
    texto = texto.lower()
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


def coletar_textos_do_details(valor):
    """
    Percorre details e pega strings com tamanho razoável.
    Isso evita transformar IDs curtos em texto analítico.
    """
    textos = []

    if isinstance(valor, dict):
        for _, v in valor.items():
            textos.extend(coletar_textos_do_details(v))

    elif isinstance(valor, list):
        for item in valor:
            textos.extend(coletar_textos_do_details(item))

    elif isinstance(valor, str):
        if len(valor.strip()) >= 15:
            textos.append(valor.strip())

    return textos


# ==========================================================
# 1. LEITURA DOS JSONs
# ==========================================================

with open(arquivo_eventos, "r", encoding="utf-8") as f:
    dados_eventos = json.load(f)

with open(arquivo_org, "r", encoding="utf-8") as f:
    dados_org = json.load(f)

eventos = dados_eventos["events"]

df_base_eventos = pd.DataFrame(eventos)

# Normaliza o campo details
details_normalizado = pd.json_normalize(
    [evento.get("details") or {} for evento in eventos],
    sep="_"
)

details_normalizado = details_normalizado.add_prefix("details_")

df_eventos = pd.concat(
    [
        df_base_eventos.drop(columns=["details"], errors="ignore"),
        details_normalizado
    ],
    axis=1
)

# Garante colunas importantes
df_eventos["parties"] = df_eventos["parties"].apply(lambda x: x if isinstance(x, list) else [])
df_eventos["qtd_parties"] = df_eventos["parties"].apply(len)
df_eventos["parties_texto"] = df_eventos["parties"].apply(lambda x: "; ".join(map(str, x)))
df_eventos["tipos_parties"] = df_eventos["parties"].apply(
    lambda lista: "; ".join(sorted(set(extrair_tipo_entidade(p) for p in lista)))
)

# Timestamp: o campo when está em formato numérico compatível com tempo Unix
df_eventos["timestamp"] = pd.to_datetime(df_eventos["when"], unit="s", errors="coerce")
df_eventos["data"] = df_eventos["timestamp"].dt.date
df_eventos["hora"] = df_eventos["timestamp"].dt.hour
df_eventos["dia_semana"] = df_eventos["timestamp"].dt.day_name()

# Extrai textos presentes dentro de details
df_eventos["texto_evento"] = [
    " ".join(coletar_textos_do_details(evento.get("details") or {}))
    for evento in eventos
]

df_eventos["qtd_caracteres_texto"] = df_eventos["texto_evento"].apply(len)
df_eventos["qtd_palavras_texto"] = df_eventos["texto_evento"].apply(lambda x: len(limpar_texto(x).split()))
df_eventos["tem_texto"] = df_eventos["qtd_palavras_texto"] > 0
df_eventos["possui_link"] = df_eventos["texto_evento"].str.contains(r"http\S+|www\S+", regex=True, na=False)
df_eventos["possui_html"] = df_eventos["texto_evento"].str.contains(r"<[^>]+>", regex=True, na=False)
df_eventos["possui_hashtag"] = df_eventos["texto_evento"].str.contains(r"#\w+", regex=True, na=False)
df_eventos["possui_mencao"] = df_eventos["texto_evento"].str.contains(r"@\w+", regex=True, na=False)
df_eventos["possui_caracter_estranho"] = df_eventos["texto_evento"].str.contains(r"[�]", regex=True, na=False)

# Org chart
df_org_nodes = pd.DataFrame(dados_org["nodes"])
df_org_edges = pd.DataFrame(dados_org["edges"])

# Grau no organograma
grau_saida = df_org_edges["source"].value_counts()
grau_entrada = df_org_edges["target"].value_counts()

df_org_nodes["grau_saida"] = df_org_nodes["id"].map(grau_saida).fillna(0).astype(int)
df_org_nodes["grau_entrada"] = df_org_nodes["id"].map(grau_entrada).fillna(0).astype(int)
df_org_nodes["grau_total"] = df_org_nodes["grau_saida"] + df_org_nodes["grau_entrada"]


# ==========================================================
# 2. TABELA EXPLODIDA DE PARTIES
# ==========================================================

linhas_parties = []

for _, row in df_eventos[["id", "short_name", "timestamp", "parties"]].iterrows():
    for party in row["parties"]:
        linhas_parties.append({
            "event_id": row["id"],
            "short_name": row["short_name"],
            "timestamp": row["timestamp"],
            "party": party,
            "party_type": extrair_tipo_entidade(party)
        })

df_parties = pd.DataFrame(linhas_parties)


# ==========================================================
# 3. IDENTIFICAÇÃO GERAL
# ==========================================================

identificacao = pd.DataFrame({
    "arquivo": [
        "MC2 data.json",
        "org_chart.json",
        "MC2 data description.md",
        "VAST Challenge 2026 MC2 Answer Sheet.htm"
    ],
    "formato": [
        "JSON",
        "JSON",
        "Markdown",
        "HTML"
    ],
    "tamanho_mb": [
        round(arquivo_eventos.stat().st_size / (1024 * 1024), 2),
        round(arquivo_org.stat().st_size / (1024 * 1024), 2),
        round((pasta_mc2 / "MC2 data description.md").stat().st_size / (1024 * 1024), 4),
        round((pasta_mc2 / "VAST Challenge 2026 MC2 Answer Sheet.htm").stat().st_size / (1024 * 1024), 2)
    ],
    "descricao_inicial": [
        "Linha do tempo de eventos coletados por sistemas internos.",
        "Organograma da Tenant Thread em formato de grafo.",
        "Documento explicativo da base MC2.",
        "Arquivo auxiliar/template de resposta do desafio."
    ]
})


# ==========================================================
# 4. ESTRUTURA, QUALIDADE E ESTATÍSTICAS
# ==========================================================

estrutura = pd.concat([
    tabela_estrutura(df_eventos, "MC2 - Events"),
    tabela_estrutura(df_parties, "MC2 - Event Parties"),
    tabela_estrutura(df_org_nodes, "MC2 - Org Nodes"),
    tabela_estrutura(df_org_edges, "MC2 - Org Edges")
], ignore_index=True)

faltantes = pd.concat([
    tabela_faltantes(df_eventos, "MC2 - Events"),
    tabela_faltantes(df_parties, "MC2 - Event Parties"),
    tabela_faltantes(df_org_nodes, "MC2 - Org Nodes"),
    tabela_faltantes(df_org_edges, "MC2 - Org Edges")
], ignore_index=True)

resumo_qualidade = pd.DataFrame({
    "tabela": [
        "MC2 - Events",
        "MC2 - Event Parties",
        "MC2 - Org Nodes",
        "MC2 - Org Edges"
    ],
    "total_linhas": [
        len(df_eventos),
        len(df_parties),
        len(df_org_nodes),
        len(df_org_edges)
    ],
    "total_colunas": [
        df_eventos.shape[1],
        df_parties.shape[1],
        df_org_nodes.shape[1],
        df_org_edges.shape[1]
    ],
    "linhas_incompletas": [
        df_eventos.isna().any(axis=1).sum(),
        df_parties.isna().any(axis=1).sum(),
        df_org_nodes.isna().any(axis=1).sum(),
        df_org_edges.isna().any(axis=1).sum()
    ],
    "registros_duplicados": [
    contar_duplicados_seguro(df_eventos),
    contar_duplicados_seguro(df_parties),
    contar_duplicados_seguro(df_org_nodes),
    contar_duplicados_seguro(df_org_edges)
]
})

estatisticas = pd.concat([
    estatisticas_numericas(df_eventos, "MC2 - Events"),
    estatisticas_numericas(df_parties, "MC2 - Event Parties"),
    estatisticas_numericas(df_org_nodes, "MC2 - Org Nodes"),
    estatisticas_numericas(df_org_edges, "MC2 - Org Edges")
], ignore_index=True)


# ==========================================================
# 5. FREQUÊNCIAS CATEGÓRICAS
# ==========================================================

freq_short_name = frequencia_categoria(df_eventos, "short_name", "frequencia_short_name_mc2_2026.csv")
freq_party_type = frequencia_categoria(df_parties, "party_type", "frequencia_party_type_mc2_2026.csv")
freq_party = frequencia_categoria(df_parties, "party", "frequencia_party_mc2_2026.csv")
freq_org_node_type = frequencia_categoria(df_org_nodes, "type", "frequencia_org_node_type_mc2_2026.csv")
freq_org_relation = frequencia_categoria(df_org_edges, "relation", "frequencia_org_relation_mc2_2026.csv")

if "details_status" in df_eventos.columns:
    freq_status = frequencia_categoria(df_eventos, "details_status", "frequencia_status_mc2_2026.csv")
else:
    freq_status = pd.DataFrame()

if "details_from" in df_eventos.columns:
    freq_from = frequencia_categoria(df_eventos, "details_from", "frequencia_details_from_mc2_2026.csv")
else:
    freq_from = pd.DataFrame()

if "details_to" in df_eventos.columns:
    freq_to = frequencia_categoria(df_eventos, "details_to", "frequencia_details_to_mc2_2026.csv")
else:
    freq_to = pd.DataFrame()


# ==========================================================
# 6. ANÁLISE TEMPORAL
# ==========================================================

eventos_por_dia = (
    df_eventos
    .groupby("data")
    .size()
    .reset_index(name="qtd_eventos")
    .sort_values("data")
)

eventos_por_hora = (
    df_eventos
    .groupby(pd.Grouper(key="timestamp", freq="h"))
    .size()
    .reset_index(name="qtd_eventos")
    .sort_values("timestamp")
)

resumo_temporal = pd.DataFrame({
    "inicio": [df_eventos["timestamp"].min()],
    "fim": [df_eventos["timestamp"].max()],
    "qtd_eventos": [len(df_eventos)],
    "qtd_dias_distintos": [df_eventos["data"].nunique()],
    "qtd_horas_distintas": [df_eventos["timestamp"].dt.floor("h").nunique()]
})


# ==========================================================
# 7. OUTLIERS
# ==========================================================

resumo_outliers_qtd_parties, outliers_qtd_parties = detectar_outliers_iqr(df_eventos, "qtd_parties")
resumo_outliers_texto, outliers_texto = detectar_outliers_iqr(df_eventos, "qtd_palavras_texto")

# Outliers por concentração de eventos por hora
resumo_outliers_eventos_hora, outliers_eventos_hora = detectar_outliers_iqr(eventos_por_hora, "qtd_eventos")


# ==========================================================
# 8. ANÁLISE TEXTUAL
# ==========================================================

df_textos = df_eventos[df_eventos["tem_texto"]].copy()

todas_palavras = []

for texto in df_textos["texto_evento"]:
    todas_palavras.extend(palavras_do_texto(texto, remover_stopwords=True))

contador_palavras = Counter(todas_palavras)
total_palavras_sem_stopwords = sum(contador_palavras.values())
palavras_unicas = len(contador_palavras)
hapax = sum(1 for palavra, freq in contador_palavras.items() if freq == 1)

resumo_textual = pd.DataFrame({
    "qtd_eventos_total": [len(df_eventos)],
    "qtd_eventos_com_texto": [len(df_textos)],
    "qtd_eventos_sem_texto": [len(df_eventos) - len(df_textos)],
    "total_palavras": [df_eventos["qtd_palavras_texto"].sum()],
    "total_caracteres": [df_eventos["qtd_caracteres_texto"].sum()],
    "media_palavras_por_texto": [df_textos["qtd_palavras_texto"].mean() if len(df_textos) > 0 else 0],
    "mediana_palavras_por_texto": [df_textos["qtd_palavras_texto"].median() if len(df_textos) > 0 else 0],
    "min_palavras": [df_textos["qtd_palavras_texto"].min() if len(df_textos) > 0 else 0],
    "max_palavras": [df_textos["qtd_palavras_texto"].max() if len(df_textos) > 0 else 0],
    "q1_palavras": [df_textos["qtd_palavras_texto"].quantile(0.25) if len(df_textos) > 0 else 0],
    "q3_palavras": [df_textos["qtd_palavras_texto"].quantile(0.75) if len(df_textos) > 0 else 0],
    "palavras_unicas_sem_stopwords": [palavras_unicas],
    "total_palavras_sem_stopwords": [total_palavras_sem_stopwords],
    "razao_tipo_token": [palavras_unicas / total_palavras_sem_stopwords if total_palavras_sem_stopwords > 0 else 0],
    "hapax_legomena": [hapax],
    "proporcao_hapax": [hapax / palavras_unicas if palavras_unicas > 0 else 0],
    "qtd_textos_com_link": [df_eventos["possui_link"].sum()],
    "qtd_textos_com_html": [df_eventos["possui_html"].sum()],
    "qtd_textos_com_hashtag": [df_eventos["possui_hashtag"].sum()],
    "qtd_textos_com_mencao": [df_eventos["possui_mencao"].sum()],
    "qtd_textos_com_caracter_estranho": [df_eventos["possui_caracter_estranho"].sum()],
    "qtd_textos_duplicados": [df_eventos["texto_evento"].duplicated().sum()]
})

frequencia_palavras = pd.DataFrame(
    contador_palavras.most_common(50),
    columns=["palavra", "frequencia"]
)

bigrams = []
trigrams = []

for texto in df_textos["texto_evento"]:
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
# 9. RELAÇÕES ENTRE VARIÁVEIS
# ==========================================================

# Cruzamento entre tipo de evento e tipo de entidade participante
cruzamento_evento_party_type = pd.crosstab(
    df_parties["short_name"],
    df_parties["party_type"]
)

# Média de número de envolvidos por tipo de evento
media_parties_por_evento = (
    df_eventos
    .groupby("short_name")[["qtd_parties", "qtd_palavras_texto", "qtd_caracteres_texto"]]
    .mean()
    .sort_values("qtd_parties", ascending=False)
)

colunas_corr = ["qtd_parties", "qtd_caracteres_texto", "qtd_palavras_texto", "hora"]

correlacao = df_eventos[colunas_corr].corr()


# ==========================================================
# 10. SALVAR CSVs
# ==========================================================

salvar_csv(identificacao, "identificacao_mc2_2026.csv")

# Atenção: esse CSV pode ficar grande, mas é útil se você quiser consultar depois
salvar_csv(df_eventos, "mc2_events_2026.csv")
salvar_csv(df_parties, "mc2_event_parties_2026.csv")
salvar_csv(df_org_nodes, "mc2_org_nodes_2026.csv")
salvar_csv(df_org_edges, "mc2_org_edges_2026.csv")

salvar_csv(estrutura, "estrutura_mc2_2026.csv")
salvar_csv(faltantes, "faltantes_mc2_2026.csv")
salvar_csv(resumo_qualidade, "resumo_qualidade_mc2_2026.csv")
salvar_csv(estatisticas, "estatisticas_numericas_mc2_2026.csv")

salvar_csv(eventos_por_dia, "eventos_por_dia_mc2_2026.csv")
salvar_csv(eventos_por_hora, "eventos_por_hora_mc2_2026.csv")
salvar_csv(resumo_temporal, "resumo_temporal_mc2_2026.csv")

salvar_csv(resumo_outliers_qtd_parties, "resumo_outliers_qtd_parties_mc2_2026.csv")
salvar_csv(outliers_qtd_parties, "outliers_qtd_parties_mc2_2026.csv")
salvar_csv(resumo_outliers_texto, "resumo_outliers_texto_mc2_2026.csv")
salvar_csv(outliers_texto, "outliers_texto_mc2_2026.csv")
salvar_csv(resumo_outliers_eventos_hora, "resumo_outliers_eventos_hora_mc2_2026.csv")
salvar_csv(outliers_eventos_hora, "outliers_eventos_hora_mc2_2026.csv")

salvar_csv(resumo_textual, "resumo_textual_mc2_2026.csv")
salvar_csv(frequencia_palavras, "frequencia_palavras_mc2_2026.csv")
salvar_csv(frequencia_bigramas, "frequencia_bigramas_mc2_2026.csv")
salvar_csv(frequencia_trigramas, "frequencia_trigramas_mc2_2026.csv")

salvar_csv(cruzamento_evento_party_type.reset_index(), "cruzamento_evento_party_type_mc2_2026.csv")
salvar_csv(media_parties_por_evento.reset_index(), "media_parties_por_evento_mc2_2026.csv")
correlacao.to_csv(pasta_saida / "correlacao_mc2_2026.csv", encoding="utf-8-sig")


# ==========================================================
# 11. GRÁFICOS
# ==========================================================

# Gráfico 1 - eventos por dia
plt.figure(figsize=(12, 6))
plt.plot(eventos_por_dia["data"].astype(str), eventos_por_dia["qtd_eventos"], marker="o")
plt.title("MC2 - Quantidade de eventos por dia")
plt.xlabel("Data")
plt.ylabel("Quantidade de eventos")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(pasta_graficos / "grafico_1_eventos_por_dia.png", dpi=300)
plt.close()

# Gráfico 2 - top tipos de evento
if not freq_short_name.empty:
    dados_plot = freq_short_name.head(15)

    plt.figure(figsize=(12, 6))
    plt.bar(dados_plot["short_name"], dados_plot["frequencia_absoluta"])
    plt.title("MC2 - Tipos de evento mais frequentes")
    plt.xlabel("Tipo de evento")
    plt.ylabel("Frequência")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(pasta_graficos / "grafico_2_tipos_evento.png", dpi=300)
    plt.close()

# Gráfico 3 - tipos de entidades envolvidas
if not freq_party_type.empty:
    dados_plot = freq_party_type.head(15)

    plt.figure(figsize=(10, 6))
    plt.bar(dados_plot["party_type"], dados_plot["frequencia_absoluta"])
    plt.title("MC2 - Tipos de entidades envolvidas nos eventos")
    plt.xlabel("Tipo de entidade")
    plt.ylabel("Frequência")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(pasta_graficos / "grafico_3_tipos_entidades.png", dpi=300)
    plt.close()

# Gráfico 4 - top parties
if not freq_party.empty:
    dados_plot = freq_party.head(20)

    plt.figure(figsize=(12, 6))
    plt.bar(dados_plot["party"], dados_plot["frequencia_absoluta"])
    plt.title("MC2 - Entidades mais frequentes nos eventos")
    plt.xlabel("Entidade")
    plt.ylabel("Frequência")
    plt.xticks(rotation=70, ha="right")
    plt.tight_layout()
    plt.savefig(pasta_graficos / "grafico_4_entidades_mais_frequentes.png", dpi=300)
    plt.close()

# Gráfico 5 - histograma qtd parties
plt.figure(figsize=(10, 6))
plt.hist(df_eventos["qtd_parties"], bins=30)
plt.title("MC2 - Distribuição da quantidade de envolvidos por evento")
plt.xlabel("Quantidade de parties")
plt.ylabel("Frequência")
plt.tight_layout()
plt.savefig(pasta_graficos / "grafico_5_histograma_qtd_parties.png", dpi=300)
plt.close()

# Gráfico 6 - tamanho textual
plt.figure(figsize=(10, 6))
plt.hist(df_eventos["qtd_palavras_texto"], bins=40)
plt.title("MC2 - Distribuição do tamanho textual dos eventos")
plt.xlabel("Quantidade de palavras")
plt.ylabel("Frequência")
plt.tight_layout()
plt.savefig(pasta_graficos / "grafico_6_histograma_tamanho_texto.png", dpi=300)
plt.close()

# Gráfico 7 - tipos de nós do organograma
if not freq_org_node_type.empty:
    dados_plot = freq_org_node_type.head(15)

    plt.figure(figsize=(10, 6))
    plt.bar(dados_plot["type"], dados_plot["frequencia_absoluta"])
    plt.title("MC2 - Tipos de nós no organograma")
    plt.xlabel("Tipo de nó")
    plt.ylabel("Frequência")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(pasta_graficos / "grafico_7_tipos_nos_organograma.png", dpi=300)
    plt.close()

# Gráfico 8 - palavras mais frequentes
if not frequencia_palavras.empty:
    dados_plot = frequencia_palavras.head(20)

    plt.figure(figsize=(12, 6))
    plt.bar(dados_plot["palavra"], dados_plot["frequencia"])
    plt.title("MC2 - Palavras mais frequentes nos textos dos eventos")
    plt.xlabel("Palavra")
    plt.ylabel("Frequência")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(pasta_graficos / "grafico_8_palavras_mais_frequentes.png", dpi=300)
    plt.close()

# Gráfico 9 - correlação
plt.figure(figsize=(8, 6))
plt.imshow(correlacao)
plt.colorbar()
plt.xticks(range(len(correlacao.columns)), correlacao.columns, rotation=45, ha="right")
plt.yticks(range(len(correlacao.columns)), correlacao.columns)
plt.title("MC2 - Correlação entre variáveis numéricas")
plt.tight_layout()
plt.savefig(pasta_graficos / "grafico_9_correlacao_mc2.png", dpi=300)
plt.close()


# ==========================================================
# 12. RESUMO NO TERMINAL
# ==========================================================

print("\nAnálise do MC2 2026 concluída!")
print(f"Eventos: {len(df_eventos)}")
print(f"Parties explodidas: {len(df_parties)}")
print(f"Nós do organograma: {len(df_org_nodes)}")
print(f"Relações do organograma: {len(df_org_edges)}")
print(f"Início: {resumo_temporal['inicio'].iloc[0]}")
print(f"Fim: {resumo_temporal['fim'].iloc[0]}")
print(f"Resultados salvos em: {pasta_saida}")
print(f"Gráficos salvos em: {pasta_graficos}")