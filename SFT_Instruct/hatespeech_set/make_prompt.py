import os
import json
import random

with open('./test/talksets-train-4.txt', 'r', encoding='utf-8') as f:
    data = f.readlines()

results=[]
random_numbers = random.sample(range(0, len(data)), 1000)
for idx in random_numbers:
    tmp = data[idx]
    tmp = (tmp.split('|')[0])[1:]
    results.append(tmp)

with open('../conversation_set/conversation_prompt.txt', 'r', encoding='utf-8') as f:
    data = f.readlines()
    
"""random_numbers = random.sample(range(0, len(data)), 300)
for idx in random_numbers:
    tmp = data[idx]
    tmp = tmp.replace("\n","")
    results.append(data[idx])"""

random.shuffle(results)
results='\n'.join(results)
with open('./hatespeech_prompt.txt', 'w') as file:
    file.write(results)