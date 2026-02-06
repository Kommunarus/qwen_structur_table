from unsloth import FastVisionModel
import torch
from langchain_core.tools import tool
import PIL
import os
from datasets import Dataset, Image
import uuid
import json


from qwen_structur_table.python_code import create_data_agent

data_app, config_data = create_data_agent()


model, tokenizer = FastVisionModel.from_pretrained(
    model_name="./model/lora_model",
    load_in_4bit=True,
)
FastVisionModel.for_inference(model)  # Enable for inference!



@tool
def structure_table(path: str) -> str:
    '''Распознает структуру и содержимое ячеек таблицы счета фактуры.
    входные переменные:
       path - абсолютный путь к изображению счета фактуры, jpg
    выходные переменные:
       строка json - список строк '[ [,,,], [,,,,], [,,,,]...]', где каждая строка представляет собой список из перечисленных через запятую значений ячеек этой строки.
    '''
    ds = Dataset.from_dict({"image": [path,], "text": ["",]}).cast_column("image", Image())

    size_func = lambda x: x.size[0]
    img = ds[0]["image"]
    image_size = 1024

    w, h = img.size
    # integer math rounding
    new_w = (w * image_size + size_func(img) // 2) // size_func(img)
    new_h = (h * image_size + size_func(img) // 2) // size_func(img)


    image = img.resize((new_w, new_h), PIL.Image.Resampling.LANCZOS)

    # image = image.resize((512, 512))
    prompt_text = "Распознай структуру и содержимое таблицы. Ответ запиши построчно в json формате. Строки отделяй символом \\n"

    messages = [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": prompt_text}
        ]}
    ]

    input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    inputs = tokenizer(
        image,
        input_text,
        add_special_tokens=False,
        return_tensors="pt",
    ).to('cuda')

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=2048,
            use_cache=True,
            temperature=1.0,
            do_sample=False,
            min_p=0.1,
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    n_stop = generated_text.find("assistant\n")
    if n_stop > 0:
        generated_text = generated_text[(n_stop + len("assistant\n")):]

    arr = [row.split('\t') for row in generated_text.split('\n')]

    name_file = str(uuid.uuid4())[:10]
    out_file = os.path.join('./sandbox',name_file+'.json')
    with open(out_file, "w") as f:
        json.dump(arr, f, ensure_ascii=False, indent=4)
        # f.write(generated_text)
    return f'Распознанный текст сохранен в файл {out_file}'

@tool
def data(query):
    """
    Facilitates automated data analysis by orchestrating an iterative
    process of code generation and execution. The function communicates
    with a data application to interpret the input query, generate
    corresponding Python code, and execute it within a managed state.
    It includes robust error handling that captures execution failures
    and attempts to regenerate the code using the error feedback. The
    process continues until a successful result is achieved, a code
    generation failure occurs, or the maximum number of retries is
    exhausted.

    :param query: The input instructions or conversational history used
        to derive the data analysis logic.
    :type query: str
    :return: A set containing either the final analysis result string
        or an error message indicating why the process failed.
    :rtype: set
    """
    max_retries = config_data["configurable"]["max_retries"]
    # создание кода
    data_app.invoke({
        "messages": query,
        "code": "",
        "result": "",
        "error_message": ""
    }, config_data)

    while True:
        current_state = data_app.get_state(config_data)
        if not current_state.values.get('code'):
            return {"Не удалось сгенерировать код для анализа данных."}
        print(f"\n==== Сгенерированный код (попытка {current_state.values.get('retry_count', 0) + 1}) ====")
        print(current_state.values['code'])


        print("\n[+] Для подтверждения. Либо исправленный код, а потом [+]")
        lines = []
        while True:
            line = input()
            if line.strip().upper() == "+":
                user_input = line
                break
            lines.append(line)
        if lines:
            user_input = "\n".join(lines)

        if user_input.strip().upper() != "+":
            # Обновляем код
            data_app.update_state(config_data, {"code": user_input})
            updated_state = data_app.get_state(config_data)
            print(f"\n==== Обновлённый код ====")
            print(updated_state.values['code'])

        # print("\n=== Выполнение кода ===")
        data_app.invoke(None, config_data)
        final_state = data_app.get_state(config_data)

        if not final_state.values.get('error_message'):
            return {final_state.values.get('result', ''),
            }

        if final_state.values.get('retry_count', 0) >= max_retries:
            return {f"Не удалось выполнить код после {max_retries} попыток. Последняя ошибка: {final_state.values.get('error_message')}",
            }
        # повторное создание кода, с добавленой ошибкой в промпт
        data_app.invoke(None, config_data)



if __name__ == "__main__":
    path_to_img = './data/f2.jpg'
    print(structure_table(path_to_img))