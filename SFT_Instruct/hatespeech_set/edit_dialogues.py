import json
import random

with open('./selected.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

print(len(data))

