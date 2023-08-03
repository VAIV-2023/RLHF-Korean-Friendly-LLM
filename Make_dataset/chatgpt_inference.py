import openai

def infer_from_gpt(instruction) :
    response = openai.ChatCompletion.create(
        model = "gpt-3.5-turbo",
        messages = [
            {"role": "system", 
             "content": "두 사람 간의 대화가 주어집니다. 다음의 지시문(Instruction)을 받게 될 것입니다.\
                        당신의 작업은 지시문과 입력에 해당하는 답변(response)을 생성하는 것입니다.\
                        너무 길지 않고 간결하게 답변을 생성해주세요."
            },
            {'role':'user',
             'content': f'Instruction: {instruction}\
                        Response: '
            },
        ],
        temperature = 0.5)
    return response['choices'][0]['message']['content']