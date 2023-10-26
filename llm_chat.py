import torch
import gradio as gr
from config import MODEL_MAPPING_PATH
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer, LlamaForCausalLM

if torch.cuda.is_available():
    device = torch.device(0)
else:
    device = torch.device('cpu')

no_change_btn = gr.Button.update()
enable_btn = gr.Button.update(interactive=True, visible=True)
disable_btn = gr.Button.update(interactive=False)
invisible_btn = gr.Button(interactive=False, visible=False)


tokenizer = None
model = None
config = None
message = []
history_token_ids = torch.tensor([[]], dtype=torch.long)


def load_model(model_name: str):
    global tokenizer
    global model
    global config
    state = None
    model_path = MODEL_MAPPING_PATH[model_name]
    if 'llama' in model_name.lower():
        tokenizer = LlamaTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = LlamaForCausalLM.from_pretrained(model_path, trust_remote_code=True, device_map="auto")
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, device_map="auto")
        if model_name.lower().startswith("qwen"):
            from transformers.generation import GenerationConfig
            config = GenerationConfig.from_pretrained(model_path, trust_remote_code=True)
        elif model_name.lower().startswith("baichuan"):
            from transformers.generation.utils import GenerationConfig
            model.generation_config = GenerationConfig.from_pretrained(model_path, trust_remote_code=True)
        else:
            pass
    return (state, [], "", "") + (enable_btn,) * 3


def llama_chat(prompt, history, temperature, top_p, max_new_tokens):
    global history_token_ids
    if not history:
        history = []
        history_token_ids = torch.tensor([[]], dtype=torch.long)
    generation_config = dict(
        temperature=temperature,
        top_k=0,
        top_p=top_p,
        do_sample=True,
        max_new_tokens=max_new_tokens
    )
    history_max_len = 2048
    input_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids
    history_token_ids = torch.concat((history_token_ids, input_ids), dim=1)
    model_input_ids = history_token_ids[:, -history_max_len:].to(device)
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=model_input_ids,
            eos_token_id=tokenizer.eos_token_id,
            **generation_config)
    model_input_ids_len = model_input_ids.size(1)
    response_ids = generation_output[:, model_input_ids_len:]
    history_token_ids = torch.concat((history_token_ids, response_ids.cpu()), dim=1)
    response = tokenizer.batch_decode(response_ids)[0].strip()
    history.append((prompt, response))
    return (history, history, None,) + (enable_btn,) * 3


def code_llama_chat(prompt, history, temperature, top_p, max_new_tokens):
    global history_token_ids
    if not history:
        history = []
    generation_config = dict(
        temperature=temperature,
        top_k=0,
        top_p=top_p,
        do_sample=True,
        max_new_tokens=max_new_tokens
    )
    history_max_len = 2048
    input_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids
    history_token_ids = torch.concat((history_token_ids, input_ids), dim=1)
    model_input_ids = history_token_ids[:, -history_max_len:].to(device)
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=model_input_ids,
            eos_token_id=tokenizer.eos_token_id,
            **generation_config)
    model_input_ids_len = model_input_ids.size(1)
    response_ids = generation_output[:, model_input_ids_len:]
    history_token_ids = torch.concat((history_token_ids, response_ids.cpu()), dim=1)
    response = tokenizer.batch_decode(response_ids)[0].strip()
    history.append((prompt.split('### 指令:\n')[-1].split('### 输入:\n')[0].strip(), response.replace('<s>', '').replace('</s>', '')))
    return (history, history, None,) + (enable_btn,) * 3


def _parse_text(text):
    lines = text.split("\n")
    lines = [line for line in lines if line != ""]
    count = 0
    for i, line in enumerate(lines):
        if "```" in line:
            count += 1
            items = line.split("`")
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
            else:
                lines[i] = f"<br></code></pre>"
        else:
            if i > 0:
                if count % 2 == 1:
                    line = line.replace("`", r"\`")
                    line = line.replace("<", "&lt;")
                    line = line.replace(">", "&gt;")
                    line = line.replace(" ", "&nbsp;")
                    line = line.replace("*", "&ast;")
                    line = line.replace("_", "&lowbar;")
                    line = line.replace("-", "&#45;")
                    line = line.replace(".", "&#46;")
                    line = line.replace("!", "&#33;")
                    line = line.replace("(", "&#40;")
                    line = line.replace(")", "&#41;")
                    line = line.replace("$", "&#36;")
                lines[i] = "<br>" + line
    text = "".join(lines)
    return text


def qwen_chat(prompt, history, temperature, top_p, max_new_tokens):
    if not history:
        history = []
    full_response = ""
    config.temperature = temperature
    config.top_p = top_p
    config.max_new_tokens = max_new_tokens
    for response in model.chat_stream(tokenizer, prompt, history=history, generation_config=config):
        full_response = _parse_text(response)
    history.append((prompt, full_response))
    return (history, history, None,) + (enable_btn,) * 3


def baichuan_chat(prompt, history, temperature, top_p, max_new_tokens):
    global message
    if not history:
        history = []
        message = []
    message.append({"role": "user", "content": prompt})
    full_response = ""
    model.generation_config.temperature = temperature
    model.generation_config.top_p = top_p
    model.generation_config.max_new_tokens = max_new_tokens
    for response in model.chat(tokenizer, message, stream=True):
        full_response = response
    message.append({"role": "assistant", "content": full_response})
    history.append((prompt, full_response))
    return (history, history, None,) + (enable_btn,) * 3