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

def load_model(model_name: str):
    global tokenizer
    global model
    global config
    state = None
    model_path = MODEL_MAPPING_PATH[model_name]
    if model_name.lower().startswith("llama"):
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
    return (state, [], "") + (enable_btn,) * 2


def llama_chat(prompt, history, temperature, top_p, max_new_tokens):
    if not history:
        history = []
    generation_config = dict(
        temperature=temperature,
        top_k=0,
        top_p=top_p,
        do_sample=True,
        max_new_tokens=max_new_tokens
    )
    with torch.no_grad():
        tokenized_data = tokenizer(prompt, return_tensors="pt")
        generation_output = model.generate(
            input_ids=tokenized_data["input_ids"].to(device),
            attention_mask=tokenized_data['attention_mask'].to(device),
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            **generation_config)
        response = tokenizer.decode(generation_output[0], skip_special_tokens=True)
    history.append((prompt, response))
    return (history, history,) + (enable_btn,) * 2


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
    return (history, history,) + (enable_btn,) * 2


def baichuan_chat(prompt, history, temperature, top_p, max_new_tokens):
    global message
    if not history:
        history = []
    message.append({"role": "user", "content": prompt})
    full_response = ""
    model.generation_config.temperature = temperature
    model.generation_config.top_p = top_p
    model.generation_config.max_new_tokens = max_new_tokens
    for response in model.chat(tokenizer, message, stream=True):
        full_response = response
    message.append({"role": "assistant", "content": full_response})
    history.append((prompt, full_response))
    return (history, history,) + (enable_btn,) * 2