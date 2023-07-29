import json

with open('hatespeech_score_final.json', 'r', encoding='utf-8') as f:
    json_data = json.load(f)

results = [] # 검수 통과 데이터
out = [] # 제외할 데이터

count = 0

for item in json_data:

  score1 = item['sum1']
  score2 = item['sum2']

  response1 = item['response1'].strip("<|endoftext|>")
  response2 = item['response2'].strip("<|endoftext|>")
  if score1 < 20 and score2 < 20:
    out.append(item)
  elif score1 == score2:
    out.append(item)
  elif score1 > score2:
    temp = {'id': item['id'], 'prompt': item['prompt']}
    temp['chosen'] = response1
    temp['rejected'] = response2
    results.append(temp)
    count = count + 1
  else:
    temp = {'id': item['id'], 'prompt': item['prompt']}
    temp['chosen'] = response2
    temp['rejected'] = response1
    results.append(temp)
    count = count + 1


with open("hatespeech_selected.json", 'w', encoding='utf-8') as outfile:
  json.dump(results, outfile, indent="\t", ensure_ascii=False)
with open("hatespeech_rejected.json", 'w', encoding='utf-8') as outfile:
  json.dump(out, outfile, indent="\t", ensure_ascii=False)
print(count)