# -*- coding: utf-8 -*-
import csv
import glob
import os
import json
import random
from concurrent.futures import ThreadPoolExecutor
import openai
from retry import retry
from tqdm import tqdm
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_chat_files_folder", type=str, default="./Eval_results")
parser.add_argument("--apikey", type=str, default="sk-eionFWpNThMNy4eeFdC25789F60a4cC2A66b2c94D3948bA6")
parser.add_argument("--chatgpt_judge_output_folder", type=str, default="./ChatGPT_Judge")
args = parser.parse_args()


def chat_classify(gpt_input, model: str = "gpt-3.5-turbo-0613"):
    @retry(tries=3, delay=10)
    def request_openai_api():
        #===============
        messages = [
            {"role": "system",
             "content": 'As a language expert, please complete the following task.'},
            {"role": "assistant",
             "content": "You are now an answer selection expert, and I will provide you with a question with several options, "
             "as well as a target sentence. Please return the alphabet of the option with the highest probability of matching "
             "this target sentence. Given question with options and the target sequence:\n" + str(gpt_input)},
            {"role": "user",
             "content": 'Please output your responses in the form of a dictionary {"maximum probability":"xxx"}  '
             'where xxx is A or B or C or ...'
            }
        ]   
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
        )
        return response
    return request_openai_api()

def process_file( eval_file):
    # time.sleep(5)
    openai.api_key = args.apikey

    with open(eval_file, 'r', encoding='utf-8') as f:
        eval_data = json.load(f)
    try:
        retry_delay = 2  # 重试延迟（秒）
        for qid_vid, item in eval_data.items():
            gpt_input = {'target sequcene': item['output_sequence'], 
                        'question': item['question'],
                        }
            eval_item_copy = item.copy()
            try:
                output_folder = os.path.join(args.chatgpt_judge_output_folder, os.path.basename(eval_file).replace('_eval.json', '_chatgpt_eval'))
                os.makedirs(output_folder, exist_ok=True)
                output_file =  os.path.join(output_folder, f'{qid_vid}.json')
                if os.path.exists(output_file):
                    pass
                    # print(f'{output_file} is existing!')
                else:
                    res = chat_classify(gpt_input)
                    content = res["choices"][0]["message"]["content"]
                    output_chatgpt_choice = json.loads(content)["maximum probability"]
                    if output_chatgpt_choice not in ['A','B','C','D','E','F']:
                        raise KeyError 
                    eval_item_copy['output_chatgpt_choice'] = output_chatgpt_choice
                    save_to_file({qid_vid:eval_item_copy}, output_file)
                
            except Exception as e:
                print(f'{e}, {eval_file}, {qid_vid}')
                time.sleep(retry_delay)
        print(f'{eval_file} is finished!!!!!!!')
    except Exception as e:
        print(f'{eval_file} is error!!!!!!!!')

def save_to_file(data, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    print(f'{output_file} is saved!')

if __name__ == '__main__':
    
    os.makedirs(args.chatgpt_judge_output_folder, exist_ok=True)
    evaljson_list = glob.glob(f'{args.model_chat_files_folder}/*_eval.json')
    print(evaljson_list)

    try:
        with ThreadPoolExecutor(64) as executor:
            results = list(
                tqdm(executor.map(process_file, evaljson_list), total=len(evaljson_list), desc="Processing and saving files"))
    except Exception as e:
        print(e)
        
    
