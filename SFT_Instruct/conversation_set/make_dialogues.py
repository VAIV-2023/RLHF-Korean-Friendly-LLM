import os
import json

directory1 = './train/NIKL_DIALOGUE_2020_v1.3/'
directory2 = './train/NIKL_DIALOGUE_2021_v1.0/'

files=[]
for filename in os.listdir(directory1):
        if filename.endswith('.json'):
            files.append(os.path.join(directory1, filename))
for filename in os.listdir(directory2):
        if filename.endswith('.json'):
            files.append(os.path.join(directory2, filename))

categories = [
    '여행지(국내/해외)', 
    '반려동물',
    '먹거리',
    '가족',
    '계절/날씨',
    '건강/다이어트',
    '꿈(목표)',
    '스포츠/레저',
    '회사/학교',
    '선물',
    '영화',
    '방송/연예',
    '아르바이트',
    '연애/결혼',
    '성격',
    '음악',
    '휴가',
    '취직',
    '경제/재테크',
    '관혼상제',
    '쇼핑',
    '우정',
    '대중교통'
]

results={}
for file in files:
    with open(file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    category = data['document'][0]['metadata']['topic'].split(' > ')[0]
    if category not in categories: continue
    if category not in results: results[category]=[]
    utterances = data['document'][0]['utterance']

    result = {}
    result['input']=""
    result['output']=""
    speaker_id = []

    for utterance in utterances:
        if len(speaker_id)==0:
            speaker_id.append(utterance['speaker_id'])
            result['input']+=utterance['form']+' '
        else:
            if len(speaker_id)==1 and utterance['speaker_id']==speaker_id[0]:
                result['input']+=utterance['form']+' '
            elif len(speaker_id)==2 and utterance['speaker_id']==speaker_id[0]:
                break
            else:
                if len(speaker_id)==1: speaker_id.append(utterance['speaker_id'])
                result['output']+=utterance['form']+' '

    result['input'] = result['input'].replace("아~","")
    result['input'] = result['input'].replace("어~","")
    result['input'] = result['input'].replace("그~","")
    result['output'] = result['output'].replace("아~","")
    result['output'] = result['output'].replace("어~","")
    result['output'] = result['output'].replace("그~","")

    results[category].append(result)

with open('./dialogues.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=4)
