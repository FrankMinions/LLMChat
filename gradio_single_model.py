import os
import json
import time
import datetime
import gradio as gr
from llm_chat import load_model, llama_chat, code_llama_chat, qwen_chat, baichuan_chat
from llm_chat import no_change_btn, enable_btn, disable_btn, invisible_btn
from config import MODELS, CODE_PROMPT_TEMPLATE

__file__ = os.path.abspath('file')

LOGDIR = "./logs"


def get_conv_log_filename():
    t = datetime.datetime.now()
    name = os.path.join(LOGDIR, f"{t.year}-{t.month:02d}-{t.day:02d}-conv.json")
    return name


def vote_last_response(state, vote_type, model_selector, request: gr.Request):
    with open(get_conv_log_filename(), mode="a", encoding='utf-8') as f:
        data = {
            "tstamp": round(time.time(), 4),
            "type": vote_type,
            "model": model_selector,
            "state": state,
            "ip": request.client.host,
        }
        f.write(json.dumps(data, ensure_ascii=False) + "\n")

        
def upvote_last_response(state, model_selector, request:gr.Request):
    vote_last_response(state[-1], "upvote", model_selector, request)
    return (state, state,) + (disable_btn,) * 3


def downvote_last_response(state, model_selector, request:gr.Request):
    vote_last_response(state[-1], "downvote", model_selector, request)
    return (state, state,) + (disable_btn,) * 3


def show_code(state):
    return state[-1][1]

def clear_history(request: gr.Request):
    state = None
    return (state, [], "", None,) + (enable_btn,) * 3


def chat(model_selector, prompt, history, temperature, top_p, max_new_tokens):
    if 'llama' in model_selector.lower():
        if 'code' in model_selector.lower():
            prompt = CODE_PROMPT_TEMPLATE.format_map({'instruction': prompt, 'input': ''})
            response = code_llama_chat(prompt, history, temperature, top_p, max_new_tokens)
        else:
            response = llama_chat(prompt, history, temperature, top_p, max_new_tokens)
    elif model_selector.lower().startswith("qwen"):
        response = qwen_chat(prompt, history, temperature, top_p, max_new_tokens)
    elif model_selector.lower().startswith("baichuan"):
        response = baichuan_chat(prompt, history, temperature, top_p, max_new_tokens)
    return response


if __name__ == '__main__':

    with gr.Blocks() as demo:

        state = gr.State([])
        notice_markdown = """# <center>⚔️ 大语言模型竞技场 ⚔️</center>"""
        gr.Markdown(notice_markdown, elem_id="notice_markdown")

        with gr.Row(elem_id="model_selector_row"):

            model_selector = gr.Dropdown(
                choices=MODELS,
                value=MODELS[0] if len(MODELS) > 0 else "",
                interactive=True,
                show_label=False,
                container=False,
            )

        chatbot = gr.Chatbot(
            [],
            label="向下滚动并开始聊天",
            avatar_images=((os.path.join(os.path.dirname(__file__), "images/human.png")),
                (os.path.join(os.path.dirname(__file__), "images/bot.png"))),).style(height=350)

        with gr.Row():

            with gr.Column(scale=0.85):
                textbox = gr.Textbox(
                    show_label=False,
                    placeholder="请在此输入您的提示词并按Enter键:",
                    container=False,
                    elem_id="input_box"
                )

            with gr.Column(scale=0.15, min_width=0):
                send = gr.Button(value="发送", variant="primary")

        with gr.Row():
            upvote_btn = gr.Button(value="👍 赞成")
            downvote_btn = gr.Button(value="👎 否决")
            regenerate_btn = gr.Button(value="🔄 重新生成")
            clear_btn = gr.Button(value="🗑️ 清除历史")

        with gr.Row():
            with gr.Column():
                code_column = gr.Code(value=None, language='python')

        with gr.Row():
            show_code_btn = gr.Button(value='显示格式化代码', variant='primary')

        with gr.Accordion("Parameters", open=False):

            temperature = gr.Slider(minimum=0.0,
                                    maximum=1.0,
                                    value=0.7,
                                    step=0.1,
                                    interactive=True,
                                    label="Temperature")

            top_p = gr.Slider(minimum=0.0,
                              maximum=1.0,
                              value=1.0,
                              step=0.1,
                              interactivate=True,
                              label="Top p")

            max_new_tokens = gr.Slider(minimum=16,
                                       maximum=2048,
                                       value=512,
                                       step=1,
                                       interactivate=True,
                                       label="Max new tokens")

        load_model(model_selector.value)

        model_selector.change(load_model,
                              inputs=[model_selector],
                              outputs=[state, chatbot, textbox] + [upvote_btn, downvote_btn, show_code_btn],
                              show_progress=True,
                             )

        textbox.submit(
            chat,
            inputs=[model_selector, textbox, state, temperature, top_p, max_new_tokens],
            outputs=[chatbot, state, code_column] + [upvote_btn, downvote_btn, show_code_btn],
        )

        send.click(
            chat,
            inputs=[model_selector, textbox, state, temperature, top_p, max_new_tokens],
            outputs=[chatbot, state, code_column] + [upvote_btn, downvote_btn, show_code_btn],
        )

        upvote_btn.click(
            upvote_last_response,
            inputs=[state, model_selector],
            outputs=[chatbot, state] + [upvote_btn, downvote_btn, show_code_btn],
        )

        downvote_btn.click(
            downvote_last_response,
            inputs=[state, model_selector],
            outputs=[chatbot, state] + [upvote_btn, downvote_btn, show_code_btn],
        )

        regenerate_btn.click(
            chat,
            inputs=[model_selector, textbox, state, temperature, top_p, max_new_tokens],
            outputs=[chatbot, state, code_column] + [upvote_btn, downvote_btn, show_code_btn],
        )

        clear_btn.click(
            clear_history,
            inputs=None,
            outputs=[state, chatbot, textbox, code_column] + [upvote_btn, downvote_btn, show_code_btn],
        )

        show_code_btn.click(
            show_code,
            inputs=[state],
            outputs=[code_column],
        )

    demo.queue()
    demo.launch()
