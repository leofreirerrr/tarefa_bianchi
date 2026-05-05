import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# =========================
# CONFIGURAÇÕES
# =========================

arquivo_json = Path("MC1_release") / "MC1_graph.json"

pasta_saida = Path("resultados_mc1")
pasta_graficos = pasta_saida / "graficos"

pasta_saida.mkdir(exist_ok=True)
pasta_graficos.mkdir(exist_ok=True)

# =========================
# LEITURA DO JSON
# =========================

with open(arquivo_json, "r", encoding="utf-8") as arquivo:
    dados = json.load(arquivo)

nodes = pd.DataFrame(dados["nodes"])
links = pd.DataFrame(dados["links"])

print("MC1 carregado com sucesso.")
print("Quantidade de nós:", len(nodes))
print("Quantidade de links:", len(links))

# Salvar bases transformadas
nodes.to_csv(pasta_saida / "mc1_nodes.csv", index=False)
links.to_csv(pasta_saida / "mc1_links.csv", index=False)

# =========================
# 1. IDENTIFICAÇÃO GERAL
# =========================

identificacao = pd.DataFrame({
    "arquivo": ["MC1_graph.json"],
    "formato": ["JSON"],
    "estrutura": ["Grafo direcionado e multigrafo"],
    "quantidade_nos": [len(nodes)],
    "quantidade_relacoes": [len(links)],
    "descricao_inicial": [
        "Base em formato de grafo, contendo entidades musicais como nós e relações entre essas entidades como links."
    ]
})

identificacao.to_csv(pasta_saida / "identificacao_mc1.csv", index=False)

# =========================
# 2. ESTRUTURA DOS DADOS
# =========================

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
    estrutura = pd.DataFrame({
        "tabela": nome_tabela,
        "variavel": df.columns,
        "tipo_pandas": df.dtypes.astype(str),
        "tipo_interpretado": [classificar_tipo(df[col]) for col in df.columns],
        "qtd_valores_unicos": [df[col].nunique(dropna=True) for col in df.columns],
        "qtd_faltantes": [df[col].isna().sum() for col in df.columns],
        "percentual_faltantes": [(df[col].isna().mean() * 100).round(2) for col in df.columns]
    })

    return estrutura

estrutura_nodes = tabela_estrutura(nodes, "MC1 - Nós")
estrutura_links = tabela_estrutura(links, "MC1 - Relações")

estrutura_nodes.to_csv(pasta_saida / "estrutura_mc1_nodes.csv", index=False)
estrutura_links.to_csv(pasta_saida / "estrutura_mc1_links.csv", index=False)

# =========================
# 3. QUALIDADE DOS DADOS
# =========================

def qualidade_dados(df, nome_tabela):
    faltantes = pd.DataFrame({
        "tabela": nome_tabela,
        "variavel": df.columns,
        "qtd_faltantes": df.isna().sum().values,
        "percentual_faltantes": (df.isna().mean() * 100).round(2).values
    })

    resumo = pd.DataFrame({
        "tabela": [nome_tabela],
        "total_linhas": [df.shape[0]],
        "total_colunas": [df.shape[1]],
        "linhas_incompletas": [df.isna().any(axis=1).sum()],
        "registros_duplicados": [df.duplicated().sum()]
    })

    return faltantes, resumo

faltantes_nodes, resumo_qualidade_nodes = qualidade_dados(nodes, "MC1 - Nós")
faltantes_links, resumo_qualidade_links = qualidade_dados(links, "MC1 - Relações")

faltantes_nodes.to_csv(pasta_saida / "faltantes_mc1_nodes.csv", index=False)
faltantes_links.to_csv(pasta_saida / "faltantes_mc1_links.csv", index=False)

resumo_qualidade = pd.concat(
    [resumo_qualidade_nodes, resumo_qualidade_links],
    ignore_index=True
)

resumo_qualidade.to_csv(pasta_saida / "resumo_qualidade_mc1.csv", index=False)

# =========================
# 4. ESTATÍSTICA DESCRITIVA
# =========================

def estatisticas_numericas(df, nome_tabela):
    numericas = df.select_dtypes(include=["int64", "float64", "int32", "float32"])

    if numericas.empty:
        return pd.DataFrame()

    resumo = pd.DataFrame({
        "tabela": nome_tabela,
        "variavel": numericas.columns,
        "qtd_validos": numericas.count().values,
        "media": numericas.mean().values,
        "mediana": numericas.median().values,
        "moda": [numericas[col].mode().iloc[0] if not numericas[col].mode().empty else None for col in numericas.columns],
        "minimo": numericas.min().values,
        "maximo": numericas.max().values,
        "amplitude": (numericas.max() - numericas.min()).values,
        "q1": numericas.quantile(0.25).values,
        "q2": numericas.quantile(0.50).values,
        "q3": numericas.quantile(0.75).values,
        "desvio_padrao": numericas.std().values,
        "variancia": numericas.var().values
    })

    return resumo

estat_nodes = estatisticas_numericas(nodes, "MC1 - Nós")
estat_links = estatisticas_numericas(links, "MC1 - Relações")

estatisticas_mc1 = pd.concat([estat_nodes, estat_links], ignore_index=True)
estatisticas_mc1.to_csv(pasta_saida / "estatisticas_numericas_mc1.csv", index=False)

# =========================
# 5. ANÁLISE CATEGÓRICA
# =========================

def salvar_frequencia(df, coluna, nome_arquivo):
    if coluna in df.columns:
        freq_abs = df[coluna].value_counts(dropna=False)
        freq_rel = df[coluna].value_counts(normalize=True, dropna=False) * 100

        tabela = pd.DataFrame({
            coluna: freq_abs.index.astype(str),
            "frequencia_absoluta": freq_abs.values,
            "frequencia_relativa_percentual": freq_rel.round(2).values
        })

        tabela.to_csv(pasta_saida / nome_arquivo, index=False)
        return tabela

freq_node_type = salvar_frequencia(nodes, "Node Type", "frequencia_node_type_mc1.csv")
freq_edge_type = salvar_frequencia(links, "Edge Type", "frequencia_edge_type_mc1.csv")

if "genre" in nodes.columns:
    freq_genre = salvar_frequencia(nodes, "genre", "frequencia_genre_mc1.csv")

if "single" in nodes.columns:
    freq_single = salvar_frequencia(nodes, "single", "frequencia_single_mc1.csv")

if "notable" in nodes.columns:
    freq_notable = salvar_frequencia(nodes, "notable", "frequencia_notable_mc1.csv")

# =========================
# 6. ANÁLISE TEMPORAL
# =========================

if "release_date" in nodes.columns:
    # Extrai o ano quando a data aparece como texto
    nodes["release_year"] = (
        nodes["release_date"]
        .astype(str)
        .str.extract(r"(\d{4})")
    )

    nodes["release_year"] = pd.to_numeric(nodes["release_year"], errors="coerce")

    resumo_temporal = pd.DataFrame({
        "variavel": ["release_year"],
        "menor_ano": [nodes["release_year"].min()],
        "maior_ano": [nodes["release_year"].max()],
        "qtd_validos": [nodes["release_year"].count()],
        "qtd_faltantes": [nodes["release_year"].isna().sum()]
    })

    resumo_temporal.to_csv(pasta_saida / "resumo_temporal_mc1.csv", index=False)

    freq_ano = nodes["release_year"].value_counts().sort_index()
    freq_ano.to_csv(pasta_saida / "frequencia_por_ano_mc1.csv")

# =========================
# 7. ANÁLISE DE GRAFO
# =========================

# Grau de entrada e saída
grau_saida = links["source"].value_counts()
grau_entrada = links["target"].value_counts()

graus = pd.DataFrame({
    "id": nodes["id"]
})

graus["grau_saida"] = graus["id"].map(grau_saida).fillna(0).astype(int)
graus["grau_entrada"] = graus["id"].map(grau_entrada).fillna(0).astype(int)
graus["grau_total"] = graus["grau_saida"] + graus["grau_entrada"]

nodes_com_grau = nodes.merge(graus, on="id", how="left")

top_grau = nodes_com_grau.sort_values("grau_total", ascending=False).head(20)
top_grau.to_csv(pasta_saida / "top_20_nos_por_grau_mc1.csv", index=False)

nodes_com_grau.to_csv(pasta_saida / "mc1_nodes_com_grau.csv", index=False)

# =========================
# 8. OUTLIERS
# =========================

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

resumo_outliers_grau, outliers_grau = detectar_outliers_iqr(nodes_com_grau, "grau_total")

resumo_outliers_grau.to_csv(pasta_saida / "resumo_outliers_grau_mc1.csv", index=False)
outliers_grau.to_csv(pasta_saida / "outliers_grau_mc1.csv", index=False)

# =========================
# 9. VISUALIZAÇÕES
# =========================

# Gráfico 1 - Tipos de nós
if "Node Type" in nodes.columns:
    contagem = nodes["Node Type"].value_counts().head(15)

    plt.figure(figsize=(10, 6))
    plt.bar(contagem.index.astype(str), contagem.values)
    plt.title("MC1 - Distribuição dos tipos de nós")
    plt.xlabel("Tipo de nó")
    plt.ylabel("Frequência")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(pasta_graficos / "grafico_1_tipos_de_nos.png", dpi=300)
    plt.close()

# Gráfico 2 - Tipos de relações
if "Edge Type" in links.columns:
    contagem = links["Edge Type"].value_counts().head(15)

    plt.figure(figsize=(10, 6))
    plt.bar(contagem.index.astype(str), contagem.values)
    plt.title("MC1 - Distribuição dos tipos de relações")
    plt.xlabel("Tipo de relação")
    plt.ylabel("Frequência")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(pasta_graficos / "grafico_2_tipos_de_relacoes.png", dpi=300)
    plt.close()

# Gráfico 3 - Gêneros mais frequentes
if "genre" in nodes.columns:
    contagem = nodes["genre"].value_counts().head(15)

    plt.figure(figsize=(12, 6))
    plt.bar(contagem.index.astype(str), contagem.values)
    plt.title("MC1 - Gêneros mais frequentes")
    plt.xlabel("Gênero")
    plt.ylabel("Frequência")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(pasta_graficos / "grafico_3_generos_mais_frequentes.png", dpi=300)
    plt.close()

# Gráfico 4 - Registros por ano
if "release_year" in nodes.columns:
    freq_ano = nodes["release_year"].value_counts().sort_index()

    plt.figure(figsize=(12, 6))
    plt.plot(freq_ano.index, freq_ano.values)
    plt.title("MC1 - Frequência de lançamentos por ano")
    plt.xlabel("Ano de lançamento")
    plt.ylabel("Quantidade de registros")
    plt.tight_layout()
    plt.savefig(pasta_graficos / "grafico_4_lancamentos_por_ano.png", dpi=300)
    plt.close()

# Gráfico 5 - Histograma do grau total
plt.figure(figsize=(10, 6))
plt.hist(nodes_com_grau["grau_total"], bins=50)
plt.title("MC1 - Histograma do grau total dos nós")
plt.xlabel("Grau total")
plt.ylabel("Frequência")
plt.tight_layout()
plt.savefig(pasta_graficos / "grafico_5_histograma_grau_total.png", dpi=300)
plt.close()

# Gráfico 6 - Boxplot do grau total
plt.figure(figsize=(10, 5))
plt.boxplot(nodes_com_grau["grau_total"], vert=False)
plt.title("MC1 - Boxplot do grau total dos nós")
plt.xlabel("Grau total")
plt.tight_layout()
plt.savefig(pasta_graficos / "grafico_6_boxplot_grau_total.png", dpi=300)
plt.close()

# =========================
# 10. RELAÇÕES ENTRE VARIÁVEIS
# =========================

# Relação entre tipo do nó e notabilidade
if "Node Type" in nodes_com_grau.columns and "notable" in nodes_com_grau.columns:
    cruzamento_node_notable = pd.crosstab(
        nodes_com_grau["Node Type"],
        nodes_com_grau["notable"],
        margins=True
    )

    cruzamento_node_notable.to_csv(pasta_saida / "cruzamento_node_type_notable_mc1.csv")

# Média de grau por tipo de nó
if "Node Type" in nodes_com_grau.columns:
    grau_por_tipo = (
        nodes_com_grau
        .groupby("Node Type")[["grau_entrada", "grau_saida", "grau_total"]]
        .mean()
        .sort_values("grau_total", ascending=False)
    )

    grau_por_tipo.to_csv(pasta_saida / "media_grau_por_tipo_no_mc1.csv")

# Correlação entre variáveis numéricas criadas
numericas_relacao = nodes_com_grau[["grau_entrada", "grau_saida", "grau_total"]]

if "release_year" in nodes_com_grau.columns:
    numericas_relacao = nodes_com_grau[["grau_entrada", "grau_saida", "grau_total", "release_year"]]

correlacao = numericas_relacao.corr()
correlacao.to_csv(pasta_saida / "correlacao_mc1.csv")

plt.figure(figsize=(8, 6))
plt.imshow(correlacao)
plt.colorbar()
plt.xticks(range(len(correlacao.columns)), correlacao.columns, rotation=45, ha="right")
plt.yticks(range(len(correlacao.columns)), correlacao.columns)
plt.title("MC1 - Matriz de correlação")
plt.tight_layout()
plt.savefig(pasta_graficos / "grafico_7_matriz_correlacao.png", dpi=300)
plt.close()



print("\nAnálise do MC1 concluída.")
print(f"Arquivos salvos na pasta: {pasta_saida}")

import re
from collections import Counter

# =========================
# 11. ANÁLISE TEXTUAL BÁSICA
# =========================

def limpar_texto(texto):
    texto = str(texto).lower()
    texto = re.sub(r"[^a-zA-ZÀ-ÿ0-9\s]", " ", texto)
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto

def metricas_textuais(df, coluna, nome_tabela):
    textos = df[coluna].dropna().astype(str)

    qtd_textos = len(textos)
    qtd_vazios = (textos.str.strip() == "").sum()

    palavras_por_texto = textos.apply(lambda x: len(limpar_texto(x).split()))
    caracteres_por_texto = textos.apply(len)

    todas_palavras = []
    for texto in textos:
        todas_palavras.extend(limpar_texto(texto).split())

    total_palavras = len(todas_palavras)
    palavras_unicas = len(set(todas_palavras))

    razao_tipo_token = palavras_unicas / total_palavras if total_palavras > 0 else 0

    contagem_palavras = Counter(todas_palavras)
    hapax = sum(1 for palavra, freq in contagem_palavras.items() if freq == 1)
    proporcao_hapax = hapax / palavras_unicas if palavras_unicas > 0 else 0

    resumo = pd.DataFrame({
        "tabela": [nome_tabela],
        "coluna_textual": [coluna],
        "qtd_textos_validos": [qtd_textos],
        "qtd_textos_vazios": [qtd_vazios],
        "total_palavras": [total_palavras],
        "total_caracteres": [caracteres_por_texto.sum()],
        "media_palavras_por_texto": [palavras_por_texto.mean()],
        "mediana_palavras_por_texto": [palavras_por_texto.median()],
        "min_palavras": [palavras_por_texto.min()],
        "max_palavras": [palavras_por_texto.max()],
        "q1_palavras": [palavras_por_texto.quantile(0.25)],
        "q3_palavras": [palavras_por_texto.quantile(0.75)],
        "media_caracteres_por_texto": [caracteres_por_texto.mean()],
        "palavras_unicas": [palavras_unicas],
        "razao_tipo_token": [razao_tipo_token],
        "hapax_legomena": [hapax],
        "proporcao_hapax": [proporcao_hapax]
    })

    palavras_frequentes = pd.DataFrame(
        contagem_palavras.most_common(30),
        columns=["palavra", "frequencia"]
    )

    return resumo, palavras_frequentes, palavras_por_texto


resumos_textuais = []

colunas_textuais_mc1 = ["name", "genre", "stage_name"]

for coluna in colunas_textuais_mc1:
    if coluna in nodes.columns:
        resumo_textual, palavras_frequentes, palavras_por_texto = metricas_textuais(
            nodes, coluna, "MC1 - Nós"
        )

        resumos_textuais.append(resumo_textual)

        palavras_frequentes.to_csv(
            pasta_saida / f"palavras_frequentes_{coluna}_mc1.csv",
            index=False
        )

        plt.figure(figsize=(10, 6))
        plt.hist(palavras_por_texto, bins=30)
        plt.title(f"MC1 - Distribuição do tamanho textual da coluna {coluna}")
        plt.xlabel("Quantidade de palavras")
        plt.ylabel("Frequência")
        plt.tight_layout()
        plt.savefig(pasta_graficos / f"grafico_texto_tamanho_{coluna}.png", dpi=300)
        plt.close()

if resumos_textuais:
    resumo_textual_mc1 = pd.concat(resumos_textuais, ignore_index=True)
    resumo_textual_mc1.to_csv(pasta_saida / "resumo_textual_mc1.csv", index=False)