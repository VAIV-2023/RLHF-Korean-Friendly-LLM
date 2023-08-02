"""
import json

def extract_instruction(input_file, output_file):
    with open(input_file, 'r',encoding='UTF8') as f:
        lines = f.readlines()

    instructions = [json.loads(line.strip())["instruction"] for line in lines]

    with open(output_file, 'w',encoding='UTF8') as f:
        for instruction in instructions:
            json.dump({"instruction": instruction}, f, ensure_ascii=False)
            f.write("\n")

if __name__ == "__main__":
    input_file = "self-evol-instruction.jsonl"  # input 파일명
    output_file = "output.json"  # output 파일명

    extract_instruction(input_file, output_file)
"""

import json

input_file_path = 'output.json'
output_file_path = 'output.json'

# 라인 단위로 JSON 객체를 읽어 리스트에 저장
data_list = []
with open(input_file_path, 'r', encoding='utf-8') as input_file:
    for line in input_file:
        data_list.append(json.loads(line))

# 쉼표를 포함한 JSON 배열 형식으로 변환하여 저장
with open(output_file_path, 'w', encoding='utf-8') as output_file:
    output_file.write(json.dumps(data_list, indent=2, ensure_ascii=False) + '\n')