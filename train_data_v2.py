import random
import json
import time

import pandas as pd

name = '维修档案(家用含配件)'

excel = pd.read_excel('./air.xlsx', name)

train_data = []
prompt_template_user = '''
给定用户描述：“%s”，请你按步骤要求工作。
步骤1：识别这句话中的空调的故障原因

请问，这个空调的故障原因：
'''
prompt_template_fix = '''
给定维修描述：“%s”，请你按步骤要求工作。
步骤1：识别这句话中的维修关键问题
请问，这个维修关键问题是：
'''

total = 900

index = 0

key_list1 = ['hello', '你好', '你是谁']
key_list2 = ['其他', '说明', ';', '；', '哈哈哈', '无']

for key in key_list1:
    train_data.append({

        'id': 'identity_{}'.format(index),
        'conversations': [
            {
                'from': 'user',
                'value': prompt_template_user % (key,)
            },
            {
                'from': 'assistant',
                'value': "请问有什么可以帮助你",
            }
        ]
    })
    index += 1
    train_data.append({

        'id': 'identity_{}'.format(index),
        'conversations': [
            {
                'from': 'user',
                'value': prompt_template_fix % (key,)
            },
            {
                'from': 'assistant',
                'value': "请问有什么可以帮助你",
            }
        ]
    })
    index += 1

for key in ['其他', '说明', ';', '；', '哈哈哈', '无']:
    train_data.append({

        'id': 'identity_{}'.format(index),
        'conversations': [
            {
                'from': 'user',
                'value': prompt_template_user % (key,)
            },
            {
                'from': 'assistant',
                'value': "请问有什么可以帮助你",
            }
        ]
    })
    index += 1
    train_data.append({

        'id': 'identity_{}'.format(index),
        'conversations': [
            {
                'from': 'user',
                'value': prompt_template_fix % (key,)
            },
            {
                'from': 'assistant',
                'value': "请问有什么可以帮助你",
            }
        ]
    })
    index += 1

for _, row in excel.iterrows():
    if row[2] in key_list1:
        continue
    if row[2] in key_list2:
        continue
    # print(row)
    temp = {
        'key': row[2],
        'fix_desc': row[3],
        'fix_name': row[4].replace("更换", "可能需要更换"),
        'err_kind': row[5],
        'err_reason': row[6]
    }
    user = temp['fix_name']

    fix = row[1]
    user_example = {

        'id': 'identity_{}'.format(index),
        'conversations': [
            {
                'from': 'user',
                'value': prompt_template_user % (temp['key'],)
            },
            {
                'from': 'assistant',
                'value': user
            }
        ]
    }
    index += 1
    fix_example = {

        'id': 'identity_{}'.format(index),
        'conversations': [
            {
                'from': 'user',
                'value': prompt_template_fix % (temp['fix_desc'],)
            },
            {
                'from': 'assistant',
                'value': fix
            }
        ]
    }
    index += 1
    train_data.append(user_example)
    train_data.append(fix_example)
    if index == total:
        print("data is enough for ", total, 'break!!!')
        break

with open('train_air.json', 'w', encoding='utf-8') as fp:
    fp.write(json.dumps(train_data))
