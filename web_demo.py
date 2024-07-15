# Copyright (c) Alibaba Cloud.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""A simple web interactive chat demo based on gradio."""
import json
import os
from argparse import ArgumentParser

import gradio as gr
import mdtex2html

import torch
from modelscope import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, snapshot_download
from peft import AutoPeftModelForCausalLM

DEFAULT_CKPT_PATH = 'qwen/Qwen-14B-Chat-Int4'
model_dir = snapshot_download(DEFAULT_CKPT_PATH)

OUTPUT_PATH = 'output_qwen'

print(model_dir)  # /mnt/workspace/.cache/modelscope/qwen/Qwen-14B-Chat-Int4


def _get_args():
    parser = ArgumentParser()
    parser.add_argument("-c", "--checkpoint-path", type=str, default=model_dir,
                        help="Checkpoint name or path, default to %(default)r")
    parser.add_argument("--cpu-only", action="store_true", help="Run demo with CPU only")

    parser.add_argument("--share", action="store_true", default=False,
                        help="Create a publicly shareable link for the interface.")
    parser.add_argument("--inbrowser", action="store_true", default=False,
                        help="Automatically launch the interface in a new tab on the default browser.")
    parser.add_argument("--server-port", type=int, default=8000,
                        help="Demo server port.")
    parser.add_argument("--server-name", type=str, default="127.0.0.1",
                        help="Demo server name.")

    args = parser.parse_args()
    return args


def _load_model_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint_path, trust_remote_code=True, resume_download=True,
    )

    if args.cpu_only:
        device_map = "cpu"
    else:
        device_map = "auto"

    model = AutoPeftModelForCausalLM.from_pretrained(
        OUTPUT_PATH,  # path to the output directory
        device_map=device_map,
        trust_remote_code=True
    ).eval()

    config = GenerationConfig.from_pretrained(
        args.checkpoint_path, trust_remote_code=True, resume_download=True,
    )

    return model, tokenizer, config


def postprocess(self, y):
    if y is None:
        return []
    for i, (message, response) in enumerate(y):
        y[i] = (
            None if message is None else mdtex2html.convert(message),
            None if response is None else mdtex2html.convert(response),
        )
    return y


gr.Chatbot.postprocess = postprocess


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


def _gc():
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _parse_response(init, data):
    print(f"Response Load: {data}")
    split = data.split('}{')
    if data == "":
        return "没有检索到历史记录，请详细描述您遇到的问题"

    result = init
    msg = ""
    data_list = []
    for index, s in enumerate(split):
        if index == 0:
            s += '}'
        elif index == len(split) - 1:
            s = '{' + s
        else:
            s = '{' + s + '}'
        try:
            loads = json.loads(s)
            ret_code = int(loads['code'])
            if ret_code == 0:
                data = loads['data']
                if data not in data_list:
                    data_list.append(data)
                    result += data + "\n"
            else:
                msg = loads['message']
        except Exception:
            print(f"Error Load: {s}")
    if len(result) != len(init):
        return result
    if msg != "":
        return msg
    return "没有检索到历史记录，请详细描述您遇到的问题"


def _launch_demo(args, model, tokenizer, config):
    def predict(_radio, _query, _chatbot, _task_history):
        print(f"User: {_parse_text(_query)}")
        _chatbot.append((_parse_text(_query), ""))
        full_response = ""
        if _radio == "用户描述":

            prompt_template = '''
            给定用户描述：“%s”，请你按步骤要求工作。
            步骤1：识别这句话中的空调的故障原因

            请问，这个空调的故障原因，并返回前3个最有可能的原因：
            '''
            prompt = prompt_template % (_query,)
            for response in model.chat_stream(tokenizer, prompt, history=_task_history, generation_config=config):
                _chatbot[-1] = (_parse_text(_query), _parse_text(response))
                yield _chatbot
                full_response = _parse_text(response)
        elif _radio == "维修描述":
            prompt_template = '''
            给定维修描述：“%s”，请你按步骤要求工作。
            请识别这句话中的维修关键问题

            请问，请问概率最高的5个维修关键问题是,用换行符\n连接：
            '''
            prompt = prompt_template % (_query,)
            for response in model.chat_stream(tokenizer, prompt, history=_task_history, generation_config=config):
                _chatbot[-1] = (_parse_text(_query), _parse_text(response))
                yield _chatbot
                full_response = _parse_text(response)
        else:
            model.generation_config.top_p = 0  # 只选择概率最高的token
            for response in model.chat_stream(tokenizer, _query, history=_task_history, generation_config=config):
                _chatbot[-1] = (_parse_text(_query), _parse_text(response))

                yield _chatbot
                full_response = _parse_text(response)
                print(f"full_response: {full_response}")

        print(f"History: {_task_history}")
        _task_history.append((_query, full_response))
        print(f"Qwen-Chat: {_parse_text(full_response)}")

    def regenerate(_chatbot, _task_history):
        if not _task_history:
            yield _chatbot
            return
        item = _task_history.pop(-1)
        _chatbot.pop(-1)
        yield from predict(item[0], _chatbot, _task_history)

    def reset_user_input():
        return gr.update(value="")

    def reset_state(_chatbot, _task_history):
        _task_history.clear()
        _chatbot.clear()
        _gc()
        return _chatbot

    with gr.Blocks() as demo:
        gr.Markdown("""\
<p align="center"><img src="https://qianwen-res.oss-cn-beijing.aliyuncs.com/logo_qwen.jpg" style="height: 30px"/><p>""")
        gr.Markdown("""<center><font size=6>Qwen-Chat Bot</center>""")
        gr.Markdown(
            """\
<center><font size=3> (美的空调维修智能语义分析。)</center>""")
        #         gr.Markdown("""\
        # <center><font size=4>
        # Qwen-7B <a href="https://modelscope.cn/models/qwen/Qwen-7B/summary">🤖 </a> |
        # <a href="https://huggingface.co/Qwen/Qwen-7B">🤗</a>&nbsp ｜
        # Qwen-7B-Chat <a href="https://modelscope.cn/models/qwen/Qwen-7B-Chat/summary">🤖 </a> |
        # <a href="https://huggingface.co/Qwen/Qwen-7B-Chat">🤗</a>&nbsp ｜
        # Qwen-14B <a href="https://modelscope.cn/models/qwen/Qwen-14B/summary">🤖 </a> |
        # <a href="https://huggingface.co/Qwen/Qwen-14B">🤗</a>&nbsp ｜
        # Qwen-14B-Chat <a href="https://modelscope.cn/models/qwen/Qwen-14B-Chat/summary">🤖 </a> |
        # <a href="https://huggingface.co/Qwen/Qwen-14B-Chat">🤗</a>&nbsp ｜
        # &nbsp<a href="https://github.com/QwenLM/Qwen">Github</a></center>""")

        chatbot = gr.Chatbot(label='Chat-Log', elem_classes="control-height")
        radio = gr.Radio(choices=["用户描述", "维修描述", "普通问题"], value="用户描述")
        query = gr.Textbox(lines=2, label='请输入问题')
        task_history = gr.State([])

        with gr.Row():
            empty_btn = gr.Button("🧹 Clear History (清除历史)")
            submit_btn = gr.Button("🚀 Submit (发送)")
            regen_btn = gr.Button("🤔️ Regenerate (重试)")

        submit_btn.click(predict, [radio, query, chatbot, task_history], [chatbot], show_progress='full')
        submit_btn.click(reset_user_input, [], [query])
        empty_btn.click(reset_state, [chatbot, task_history], outputs=[chatbot], show_progress='full')
        regen_btn.click(regenerate, [chatbot, task_history], [chatbot], show_progress='full')

    #         gr.Markdown("""\
    # <font size=2>Note: This demo is governed by the original license of Qwen. \
    # We strongly advise users not to knowingly generate or allow others to knowingly generate harmful content, \
    # including hate speech, violence, pornography, deception, etc. \
    # (注：本演示受Qwen的许可协议限制。我们强烈建议，用户不应传播及不应允许他人传播以下内容，\
    # 包括但不限于仇恨言论、暴力、色情、欺诈相关的有害信息。)""")

    demo.queue().launch(
        share=args.share,
        inbrowser=args.inbrowser,
        server_port=args.server_port,
        server_name=args.server_name,
    )


def main():
    args = _get_args()

    model, tokenizer, config = _load_model_tokenizer(args)
    # model.generation_config.top_p = 5  # 只选择概率最高的token
    _launch_demo(args, model, tokenizer, config)


if __name__ == '__main__':
    main()
