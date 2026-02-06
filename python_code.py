import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, MessagesState, END, START
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
import glob
from pydantic import BaseModel, Field
from typing import List
import pathlib
import autogen
import ast

ALLOWED_MODULES = {'pandas', 'numpy', 'matplotlib', 'seaborn', 'scipy', 'datetime',
                   'math', 'json', 'csv', 're'}

FORBIDDEN_NAMES = {'os', 'sys', 'subprocess', 'socket', 'requests', 'urllib',
                   'eval', 'exec', '__import__', 'pickle', 'shutil', "compile",
                   "input", "__builtins__"}

allowed = ', '.join(ALLOWED_MODULES)
forbidden = ', '.join(FORBIDDEN_NAMES)

dir_files = './sandbox'
SANDBOX_DIR = pathlib.Path(dir_files)


class CustomMessagesState(MessagesState):
    files: list[str] = []
    files_content: list[str] = []
    code: str = ""
    img_path: list[str] = []
    retry_count: int = 0
    error_message: str = ""
    result: str = ""


def node_read_files(state: CustomMessagesState):
    files = glob.glob(dir_files + "/*.*")
    names_list = [os.path.basename(row) for row in files]
    content = []
    return {'files': names_list, 'files_content': content}


class RouterProfile(BaseModel):
    code: str = Field(description="Код")
    img_path: List[str] = Field(
        description="Список изображений, которые будут созданы на диске, и на которые есть ссылки в коде")


def node_write_code(state: CustomMessagesState):
    load_dotenv('my.env')
    MODEL = os.getenv("MODEL")
    API_KEY = os.getenv("API_KEY")
    API_BASE = os.getenv("API_BASE")


    model_data = ChatOpenAI(model_name=MODEL, openai_api_key=API_KEY, openai_api_base=API_BASE)
    coder_model_data = model_data.with_structured_output(RouterProfile).bind(temperature=0.1)

    if len(state["files"]) > 0 and len(state["files_content"]) > 0 and len(state["files"]) == len(
            state["files_content"]):
        text_about_files = 'Тебе доступны следующие файлы:\n'
        for name, content in zip(state["files"], state["files_content"]):
            text_about_files += f"{name}\n"
            # text_about_files += f"{name}\nЕго содержимое:\n{content}\n"
    else:
        text_about_files = 'На диске нет файлов.\n'

    CODER_PROMPT = f"""Ты аналитик данных. 
    ТВОЯ ЗАДАЧА:
    Тебе нужно подготавливать Python‑код для выполнения задачи, которую тебя попросят решить.
    Если речь идёт о визуальном отображении, прописывай в коде сохранение графиков с уникальными именами в песочнице 
    и возвращай ссылки пути до файлов.
    Если код не выводит данные для отображения через print, а просто сохраняет изображение, то все равно выводи через print() информацию 
    о том, что сделано.

    КОНТЕКСТ:
    При генерации кода учитывай, что:
    - Есть доступ к записи и чтению файлов в песочнице по адресу '../{SANDBOX_DIR.name}'
      Если тебе дается просто имя файла, то ищи его песочнице, всегда добавляя к имени папку: '../{SANDBOX_DIR.name}'

    - {text_about_files}

    КОНТЕКСТ:
    При генерации кода учитывай что:
    - В распоряжении есть такие библиотеки как: {allowed}
    - Запрещено использовать библиотеки: {forbidden}

    ФОРМАТ ОТВЕТА:
    code: python код без комментариев и форматирования
    img_path: Путь на диске к создаваемым изображениям. Напиши путь используя '../{SANDBOX_DIR.name}'
    Обязательно перечисляй все новый изображений в img_path.
    """

    if state.get("error_message"):
        user_message = f"""{state['messages'][-1].content}

        ПРЕДЫДУЩАЯ ПОПЫТКА ЗАВЕРШИЛАСЬ ОШИБКОЙ:

        КОД, КОТОРЫЙ ВЫЗВАЛ ОШИБКУ:
        ```python
        {state.get('code', '')}
        ```

        ТЕКСТ ОШИБКИ:
        {state['error_message']}

        ИСПРАВЬ КОД С УЧЁТОМ ЭТОЙ ОШИБКИ."""
    else:
        user_message = state["messages"][-1].content

    prompt_template = ChatPromptTemplate.from_messages([
        SystemMessage(content=CODER_PROMPT),
        HumanMessage(content=user_message)
    ])
    chain = prompt_template | coder_model_data
    result = chain.invoke({})
    return {
        "code": result.code,
        "img_path": result.img_path
    }


_executor = None


def get_executor():
    global _executor
    WORK_DIR = pathlib.Path("./workdir")
    if _executor is None:
        _executor = autogen.UserProxyAgent(
            name="executor",
            human_input_mode="NEVER",
            code_execution_config={"work_dir": str(WORK_DIR), "use_docker": False}
        )
    return _executor


class CodeValidator(ast.NodeVisitor):
    def __init__(self):
        self.error = None

    def visit_Import(self, node):
        for alias in node.names:
            root = alias.name.split(".")[0]
            if root not in ALLOWED_MODULES:
                self.error = f"❌ Модуль '{root}' не разрешён"
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        if node.module is None:
            self.error = "❌ Относительные импорты запрещены"
            return
        root = node.module.split(".")[0]
        if root not in ALLOWED_MODULES:
            self.error = f"❌ Модуль '{root}' не разрешён"
        self.generic_visit(node)

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name):
            if node.func.id in FORBIDDEN_NAMES:
                self.error = f"❌ Запрещённая функция: {node.func.id}"

        if isinstance(node.func, ast.Attribute):
            if node.func.attr in FORBIDDEN_NAMES:
                self.error = f"❌ Запрещённый атрибут: {node.func.attr}"
        self.generic_visit(node)

    def visit_Attribute(self, node):
        if node.attr.startswith("__"):
            self.error = "❌ Доступ к dunder-атрибутам запрещён"
        self.generic_visit(node)

    def visit_Constant(self, node):
        if isinstance(node.value, str):
            try:
                p = pathlib.Path(node.value)
                if p.is_absolute() or "/" in node.value or "\\" in node.value:
                    resolved = (SANDBOX_DIR / p).resolve()
                    if not resolved.is_relative_to(SANDBOX_DIR.resolve()):
                        self.error = f"❌ Доступ вне sandbox запрещён: {node.value}"
            except Exception:
                pass
        self.generic_visit(node)


def validate_code(code: str) -> tuple[bool, str]:
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, f"❌ SyntaxError: {e}"

    validator = CodeValidator()
    validator.visit(tree)
    if validator.error:
        return False, validator.error
    return True, "✅ Код безопасен для выполнения в sandbox"


def node_run_code(state: CustomMessagesState):
    code = state['code']
    is_safe, message = validate_code(code)
    if not is_safe:
        return {
            "result": "",
            "error_message": message,
            "retry_count": state.get("retry_count", 0) + 1
        }
    try:
        executor = get_executor()
        exit_code, output = executor.execute_code_blocks([("python", code)])
        # print(output)
        if exit_code == 0:
            return {
                "result": output,
                "error_message": "",
                "retry_count": state.get("retry_count", 0)
            }
        else:
            return {
                "result": "",
                "error_message": output,
                "retry_count": state.get("retry_count", 0) + 1
            }
    except Exception as e:
        return {
            "result": "",
            "error_message": str(e),
            "retry_count": state.get("retry_count", 0) + 1
        }


def should_retry(state: CustomMessagesState, config: RunnableConfig) -> str:
    max_retries = config.get("configurable", {}).get("max_retries", 10)

    if state.get("error_message") and state.get("retry_count", 0) < max_retries:
        return "retry"
    else:
        return "end"

def create_data_agent():


    graph = StateGraph(CustomMessagesState)

    graph.add_node("read_files", node_read_files)
    graph.add_node("write_code", node_write_code)
    graph.add_node("run_code", node_run_code)
    graph.add_edge(START, "read_files")
    graph.add_edge("read_files", "write_code")
    graph.add_edge("write_code", "run_code")
    graph.add_conditional_edges(
        "run_code",
        should_retry,
        {
            "retry": "write_code",  # Возвращаемся к генерации кода
            "end": END  # Заканчиваем работу
        }
    )
    checkpointer = MemorySaver()
    app = graph.compile(interrupt_before=["run_code"], checkpointer=checkpointer, )


    config = {
        "configurable": {
            "thread_id": "code-data-1",
            "max_retries": 10  # Здесь настраиваем лимит попыток
        }
    }

    return app, config





