import json
import pandas as pd
from pathlib import Path

caminho = Path(r"MC1_release\MC1_graph.json")

with open(caminho, "r", encoding="utf-8") as arquivo:
    dados = json.load(arquivo)

print("Chaves principais do JSON:")
print(dados.keys())

print("\nResumo das chaves:")
for chave, valor in dados.items():
    print("-" * 60)
    print("Chave:", chave)
    print("Tipo:", type(valor))

    if isinstance(valor, list):
        print("Quantidade de registros:", len(valor))
        if len(valor) > 0:
            print("Primeiro registro:")
            print(valor[0])

    elif isinstance(valor, dict):
        print("Subchaves:")
        print(valor.keys())