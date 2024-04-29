from modelscope import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import AutoPeftModelForCausalLM

#model_dir = snapshot_download('qwen/Qwen-14B-Chat-Int4')

tokenizer = AutoTokenizer.from_pretrained('output_qwen', trust_remote_code=True)


model = AutoPeftModelForCausalLM.from_pretrained(
    'output_qwen', # path to the output directory
    device_map="auto",
    trust_remote_code=True
).eval()

prompt_template = '''
给定一句话：“%s”，请你按步骤要求工作。

步骤1：识别这句话中的空调的故障原因
步骤2：根据故障原因，生成JSON字符串，格式为{"fix_desc":描述,"fix_name":名称,"err_kind":类型,"err_reason":原因}

请问，这个JSON字符串是：
'''

model.generation_config.top_p=4 # 只选择概率最高的token

Q_list=['不制热/制热效果差']
for Q in Q_list:
    prompt=prompt_template%(Q,)
    A,hist=model.chat(tokenizer,prompt,history=None)
    print('Q:%s\nA:%s\n'%(Q,A))