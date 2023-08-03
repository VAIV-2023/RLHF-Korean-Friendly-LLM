import json

def read_text_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    return [line.strip() for line in lines]

# memo.txt 파일의 경로를 지정합니다.
file_path = './final_conversation.txt'

# memo.txt 파일로부터 텍스트를 읽어서 input_texts 리스트에 저장합니다.
input_texts = read_text_from_file(file_path)

def create_json_file(text_list, output_file):
    json_list = []
    for i, text in enumerate(text_list):
        json_entry = {
            "id3": f"conversation_{i}",  # 새로운 텍스트마다 고유한 id를 생성합니다. 여기서는 "hatespeech_숫자" 형태로 지정합니다.
            "prompt": text,  # 주어진 텍스트를 instruction으로 사용합니다.
        }
        json_list.append(json_entry)

    # 생성된 json_list를 json 파일로 저장합니다.
    with open(output_file, 'w',encoding='utf-8') as json_file:
        json.dump(json_list, json_file, indent=4, ensure_ascii=False)

output_file_path = "step3_conversation.json"  # 생성할 json 파일의 경로와 이름을 지정합니다.
create_json_file(input_texts, output_file_path)