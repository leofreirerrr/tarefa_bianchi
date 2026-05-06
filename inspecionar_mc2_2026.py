from pathlib import Path
import json
import pandas as pd

# Ajuste esse caminho se o nome da sua pasta for diferente
pasta_mc2 = Path(r"VAST_Challenge_2026_MC2\VAST_Challenge_2026_MC2")

print("=" * 100)
print("INSPEÇÃO DO MC2 2026")
print("=" * 100)

if not pasta_mc2.exists():
    print("Pasta não encontrada:")
    print(pasta_mc2)
    print("\nConfira o nome da pasta do MC2.")
    exit()

arquivos = [arq for arq in pasta_mc2.rglob("*") if arq.is_file()]

print(f"\nTotal de arquivos encontrados: {len(arquivos)}\n")

for arq in arquivos:
    tamanho_mb = arq.stat().st_size / (1024 * 1024)
    print(f"- {arq} | {tamanho_mb:.2f} MB")

print("\n" + "=" * 100)
print("TENTANDO LER OS ARQUIVOS PRINCIPAIS")
print("=" * 100)

for arq in arquivos:
    extensao = arq.suffix.lower()

    print("\n" + "-" * 100)
    print(f"Arquivo: {arq}")
    print(f"Extensão: {extensao}")

    if extensao == ".json":
        try:
            with open(arq, "r", encoding="utf-8") as f:
                dados = json.load(f)

            print("Tipo principal:", type(dados))

            if isinstance(dados, dict):
                print("Chaves principais:", dados.keys())

                for chave, valor in dados.items():
                    print(f"\nChave: {chave}")
                    print("Tipo:", type(valor))

                    if isinstance(valor, list):
                        print("Quantidade de registros:", len(valor))
                        if len(valor) > 0:
                            print("Primeiro registro:")
                            print(valor[0])

                    elif isinstance(valor, dict):
                        print("Subchaves:", valor.keys())

            elif isinstance(dados, list):
                print("Quantidade de registros:", len(dados))
                if len(dados) > 0:
                    print("Primeiro registro:")
                    print(dados[0])

        except Exception as e:
            print("Erro ao ler JSON:", e)

    elif extensao == ".csv":
        try:
            df = pd.read_csv(arq)
            print("Linhas e colunas:", df.shape)
            print("Colunas:", df.columns.tolist())
            print("\nTipos:")
            print(df.dtypes)
            print("\nPrimeiras linhas:")
            print(df.head().to_string(index=False))

        except Exception as e:
            print("Erro ao ler CSV:", e)

    elif extensao in [".txt", ".md"]:
        try:
            with open(arq, "r", encoding="utf-8") as f:
                texto = f.read()

            print("Quantidade de caracteres:", len(texto))
            print("\nPrimeiros 1000 caracteres:")
            print(texto[:1000])

        except Exception as e:
            print("Erro ao ler texto:", e)

    elif extensao in [".xlsx", ".xls"]:
        try:
            abas = pd.ExcelFile(arq).sheet_names
            print("Abas encontradas:", abas)

            for aba in abas:
                df = pd.read_excel(arq, sheet_name=aba)
                print(f"\nAba: {aba}")
                print("Linhas e colunas:", df.shape)
                print("Colunas:", df.columns.tolist())
                print(df.head().to_string(index=False))

        except Exception as e:
            print("Erro ao ler Excel:", e)

    else:
        print("Arquivo não analisado automaticamente nessa inspeção.")