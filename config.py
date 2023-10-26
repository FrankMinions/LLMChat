MODELS = [
    "LLaMA-2-7B-Chat",
    "LLaMA-2-13B-Chat",
    "LLaMA-2-70B-Chat",
    "Chinese-CodeLLaMA-2-7B"
    "ChatGLM2-6B",
    "Qwen-7B-Chat",
    "Qwen-14B-Chat",
    "Baichuan2-7B-Chat",
    "Baichuan2-13B-Chat"
]

MODEL_MAPPING_PATH = {
    "LLaMA-2-7B-Chat": "",
    "Chinese-CodeLLaMA-2-7B": "",
    "Qwen-7B-Chat": "",
    "Baichuan2-7B-Chat": ""
}

CODE_PROMPT_TEMPLATE = (
        "下面是描述一项任务的指令，并且与一则输入配对用来提供更多的上下文。请给出尽可能满足请求的回答.\n"
        "### 指令:\n{instruction}\n### 输入:\n{input}\n### 回答:\n"
)