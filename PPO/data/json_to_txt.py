import json

def create_txt_from_json(input_json_file, output_txt_file):
    with open(input_json_file, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)

    instructions = [entry['instruction'] for entry in data]

    with open(output_txt_file, 'w', encoding='utf-8') as txt_file:
        for instruction in instructions:
            txt_file.write(instruction + '\n')

if __name__ == "__main__":
    input_json_file = "hatespeech.json"  # JSON 파일 경로
    output_txt_file = "hatespeech.txt"         # 생성할 TXT 파일 경로

    create_txt_from_json(input_json_file, output_txt_file)