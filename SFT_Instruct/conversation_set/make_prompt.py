import os
import json

directory1 = './test/'

files=[]
for filename in os.listdir(directory1):
    if filename.endswith('.txt'):
        files.append(os.path.join(directory1, filename))

results=[]
for file in files:
    with open(file, 'r', encoding='utf-8') as f:
        data = (f.readlines()[0]).split('1 : ')[1].replace("키키","").replace("\n","")
        results.append(data)

results='\n'.join(results)
with open('./conversation_promt.txt', 'w') as file:
    file.write(results)