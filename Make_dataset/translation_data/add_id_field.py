import json

def add_id_to_json(json_data):
    updated_json = []
    for idx, value in enumerate(json_data):        
        updated_item = {
            "id": f"evol_instruction_{idx + 1}",
            "prompt": value['instruction']
        }
        updated_json.append(updated_item)
    return updated_json

def save_json_file(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    input_file_path = './output.json'
    output_file_path = './step2_prompt.json'

    # Load data from JSON file
    with open(input_file_path, 'r', encoding='utf-8') as json_file:
        json_data = json.load(json_file)

    # Add 'id' field to the JSON data
    json_data_with_id = add_id_to_json(json_data)

    # Save the updated JSON data to a new file
    save_json_file(json_data_with_id, output_file_path)
