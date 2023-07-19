import json
import random

with open('./dialogues.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

for category in data:
    li = data[category]
    for item in li:
        if len(item['input'])<10  or len(item['output'])<10:
            li.remove(item) 
    
    new=[]
    random_numbers = random.sample(range(0, len(li)), 65)
    for idx in random_numbers:
        new.append(li[idx])
    
    data[category] = new

with open('./selected.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)
