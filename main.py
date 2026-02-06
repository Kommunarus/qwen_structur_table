from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from qwen_structur_table.tool import structure_table, data

import os
from dotenv import load_dotenv


load_dotenv('.env')
MODEL = os.getenv('MODEL')
API_KEY = os.getenv('API_KEY')
API_BASE = os.getenv('API_BASE')

model = ChatOpenAI(
    model=MODEL,
    openai_api_key=API_KEY,
    openai_api_base=API_BASE,
    temperature=0.1,
    max_retries=3,
)

system_prompt = f'''Ты помощник дата аналитика.

**Доступные инструменты:**

1. **structure_table** — распознает таблицу документа и возвращает строку распознанного текста. 

2. **data** — позволяет генерировать python код и выполнять его в песочнице. Можно дать команду что-то посчитать
и инструмент составит код, выполнит его и  вернет результат в текстовом виде."


**Правила выбора инструмента:**
- если тебя просят распознать документ - вызови structure_table
- если тебя просят проанализировать документ, прочитать файл, создать файл - то отправляй запрос
инструменту data.


'''

agent = create_agent(
    model=model,
    tools=[structure_table, data],
    system_prompt=system_prompt,
)



def interactive_data_analyst():

    while True:
        user_input = input("👤 Вы: ").strip()

        if user_input.lower() in ['exit', 'quit', 'выход']:
            print("\n👋 До свидания!")
            break

        if not user_input:
            continue

        response = agent.invoke(
            {"messages": [HumanMessage(content=user_input)]},
        )

        answer = response["messages"][-1].content
        print(f"\n🤖 Агент: {answer}\n")


interactive_data_analyst()