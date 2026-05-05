import json
from pathlib import Path

arquivo_json = Path(r"VAST_Challenge_2026_MC1\VAST_Challenge_2026_MC1\MC1_final_00.json")

with open(arquivo_json, "r", encoding="utf-8") as arquivo:
    dados = json.load(arquivo)

print("Arquivo lido com sucesso!")
print("Chaves principais:", dados.keys())

rounds = dados["rounds"]

print("\nQuantidade de rounds:", len(rounds))

primeiro_round = rounds[0]

print("\nChaves do primeiro round:")
print(primeiro_round.keys())

print("\nHorário do primeiro round:")
print(primeiro_round.get("hour"))

print("\nChaves de environment_context:")
print(primeiro_round.get("environment_context", {}).keys())

print("\nQuantidade de communications no primeiro round:")
print(len(primeiro_round.get("communications", [])))

if primeiro_round.get("communications"):
    print("\nPrimeira communication:")
    print(primeiro_round["communications"][0])

print("\nQuantidade de participants no primeiro round:")
print(len(primeiro_round.get("participants", [])))

if primeiro_round.get("participants"):
    print("\nPrimeiro participant:")
    print(primeiro_round["participants"][0])

media_events = primeiro_round.get("environment_context", {}).get("media_events", [])

print("\nQuantidade de media_events no primeiro round:")
print(len(media_events))

if media_events:
    print("\nPrimeiro media_event:")
    print(media_events[0])