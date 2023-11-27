
[Arxiv](xxx)
<p align="center">
    <img src="assets/pie_fig.jpg" width="300" style="margin-bottom: 0.2;"/>
<p>
<h2 align="center"> <a href="https://arxiv.org/abs/2311.10122">Video-Bench: A Comprehensive Benchmark and Toolkit for Evaluating Video-based Large Language Models</a></h2>
<!-- <h5 align="center"> If you like our project, please give us a star ⭐ on GitHub for latest update.  </h2> -->


 

* **We introduce Video-Bench, the first comprehensive evaluation benchmark for Video-LLMs, featuring a three-level ability assessment that systematically evaluates models in video-exclusive understanding, prior knowledge incorporation, and video-based decision-making abilities.**
* **We provide a user-friendly evaluation toolkit. Accompanied by our datasets and QA pairs, the toolkit can streamline the performance assessment of Video-LLMs.**
* **We conduct extensive experiments to evaluate prominent Video-LLMs, summarizing their behaviors, analyzing main causes for observed limitations, and proposing future directions for improvement.**


## 📰 News
**[2023.11.27]** Video-Bench is released! Data and evaluation code is available now.

## 📣 Leaderboard
Welcome to [Video_Benchmark Leaderboard](https://github.com/PKU-YuanGroup/Video-Bench)!

🚩🚩🚩 We are delighted to have witnessed the remarkable advancements in video understanding and artificial intelligence alongside the community over the past year. We are proud to announce the launch of Video-Bench, a platform designed to assist developers and users in the field of video analysis.

🔥🔥🔥  Video-Bench is committed to promoting the progress of video understanding models and facilitating their evaluation. We are pleased to announce the inaugural Video-Bench Leaderboard. This leaderboard aims to systematically evaluate the performance of video understanding models across various capabilities, including **Prior Knowledge based QA, Comprehension Decision-making, Video-exclusive Understanding,** and more.
The leaderboard will feature rankings for open-source models, providing an inclusive and comprehensive reference for the industry and research community. We invite developers and researchers working on video understanding models to join Video-Bench and showcase their models' performance advantages in different domains.

👋👋👋 We also welcome valuable suggestions and contributions from the community to foster collaborative growth and advancement in video understanding models. If you have any questions or would like to get involved, please feel free to contact us. Let's eagerly anticipate the release of the Video-Bench Leaderboard and the continued progress in video understanding and artificial intelligence!

## 🤗 Evaluation

### 📂 Data Preparation
The video data can easily downloaded from [Huggingface](https://huggingface.co/datasets/Ai4Happiness/Video-Bench) 


### 🏗️ Evaluate on your own model
The code below is just a generalized framework for dataset evaluation, you will need to refine the model loading part according to your own model. Once the code execution is complete, you will find some JSON files named `./Eval_results/{dataset_name}.json`. 

```python 
Eval_QA_root = './'
Eval_Video_root = './'

dataset_qajson = {
  "Ucfcrime": f"{Eval_QA_root}/Eval_QA/Ucfcrime_QA_new.json",
  "Youcook2": f"{Eval_QA_root}/Eval_QA/Youcook2_QA_new.json",
  "TVQA": f"{Eval_QA_root}/Eval_QA/TVQA_QA_new.json",
  "MSVD": f"{Eval_QA_root}/Eval_QA/MSVD_QA_new.json",
  "MSRVTT": f"{Eval_QA_root}/Eval_QA/MSRVTT_QA_new.json",
  "Driving-decision-making": f"{Eval_QA_root}/Eval_QA/Driving-decision-making_QA_new.json",
  "NBA": f"{Eval_QA_root}/Eval_QA/NBA_QA_new.json",
  "SQA3D": f"{Eval_QA_root}/Eval_QA/SQA3D_QA_new.json",
  "Driving-exam": f"{Eval_QA_root}/Eval_QA/Driving-exam_QA_new.json",
  "MV": f"{Eval_QA_root}/Eval_QA/MV_QA_new.json",
  "MOT": f"{Eval_QA_root}/Eval_QA/MOT_QA_new.json",
  "ActivityNet": f"{Eval_QA_root}/Eval_QA/ActivityNet_QA_new.json",
  "TGIF": f"{Eval_QA_root}/Eval_QA/TGIF_QA_new.json"
}

if args.dataset_name is None:
        dataset_name_list = list(dataset_qajson.keys())
    else:
        dataset_name_list = [args.dataset_name]
        print(f'Specifically run {args.dataset_name}')
    print(dataset_name_list)
    os.makedirs('./Eval_results', exist_ok=True)
    
    for dataset_name in dataset_name_list:
        qa_json = dataset_qajson[dataset_name]
        print(f'Dataset name:{dataset_name}, {qa_json=}!')
        with open(qa_json, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        eval_dict = {}
        for idx, (q_id, item) in enumerate(data.items()):
            try:   
                video_id = item['video_id']
                question = item['question'] 
                answer_ = item['answer']
                if len(item['choices']) == 6:
                    question += f"Choices: A.{item['choices']['A']} B.{item['choices']['B']} C.{item['choices']['C']} D.{item['choices']['D']} E.{item['choices']['E']} F.{item['choices']['F']} \n Among the six options A, B, C, D, E, F above, the one closest to the correct answer is:"
                    candidates = ['A', 'B', 'C', 'D', 'E', 'F']
                    candidates_long = [f" A.{item['choices']['A']}", f"B.{item['choices']['B']}", f"C.{item['choices']['C']}", f"D.{item['choices']['D']}", f"E.{item['choices']['E']}", f"F.{item['choices']['F']}"]
                elif len(item['choices']) == 5:
                    question += f" A.{item['choices']['A']} B.{item['choices']['B']} C.{item['choices']['C']} D.{item['choices']['D']} E.{item['choices']['E']} \n Among the five options A, B, C, D, E above, the one closest to the correct answer is: "
                    candidates = ['A', 'B', 'C', 'D', 'E']
                    candidates_long = [f" A.{item['choices']['A']}", f"B.{item['choices']['B']}", f"C.{item['choices']['C']}", f"D.{item['choices']['D']}", f"E.{item['choices']['E']}"]
                elif len(item['choices']) == 4:
                    question += f" A.{item['choices']['A']} B.{item['choices']['B']} C.{item['choices']['C']} D.{item['choices']['D']} \n Among the four options A, B, C, D above, the one closest to the correct answer is:"
                    candidates = ['A', 'B', 'C', 'D']
                    candidates_long = [f" A.{item['choices']['A']}", f"B.{item['choices']['B']}", f"C.{item['choices']['C']}", f"D.{item['choices']['D']}"]
                elif len(item['choices']) == 3:
                    question += f" A.{item['choices']['A']} B.{item['choices']['B']} C.{item['choices']['C']} \n Among the three options A, B, C above, the one closest to the correct answer is: "
                    candidates = ['A', 'B', 'C']
                    candidates_long = [f" A.{item['choices']['A']}", f"B.{item['choices']['B']}", f"C.{item['choices']['C']}"]
                elif len(item['choices']) == 2:
                    question += f" A.{item['choices']['A']} B.{item['choices']['B']} \n Among the two options A, B above, the one closest to the correct answer is: "
                    candidates = ['A', 'B']
                    candidates_long = [f" A.{item['choices']['A']}", f"B.{item['choices']['B']}"]
                vid_rela_path = item['vid_path']
                vid_path = os.path.join(Eval_Video_root, vid_rela_path)
                output, output_scores = ask(args, question, model, tokenizer, image_processor, vid_path)

                eval_dict[q_id] = {
                    'video_id': video_id,
                    'question': question,
                    'output_sequence': output,
                    'correct': answer_
                }  
                print(f'q_id:{q_id}, output:{output}, correct answer:{answer_}!\n')
            except Exception as e:
                traceback.print_exc()  
        # eval results
        eval_dataset_json = f'./Eval_results/{dataset_name}_eval.json'
        with open(eval_dataset_json, 'w', encoding='utf-8') as f:
            json.dump(eval_dict, f, indent=2)

```

After obtaining the `Eval/{dataset_name}.json` files, you can utilize ChatGPT or T5 model as experts to assess the correctness of the model's output answer. The specific code is as follows:

#### ChatGPT Evaluation
```python
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

apikeys = [
        'sk-B8uCs6QZE33bmVr63WnJT3BlbkFJaNHlSZ3Bxxxxxxxx',
    ]  
save_dir = './ChatGPT_Evaluation'
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
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

def process_file( eval_file, qa_file=None):
    time.sleep(5)
    openai.api_key = random.choice(apikeys)
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
                output_folder = os.path.join(save_dir, os.path.basename(eval_file).replace('_eval.json', '_chatgpt_eval'))
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
    evaljson_list = []
    evaljson_list.extend(glob.glob('./Eval_results/*_eval.json'))
    evaljson_list.sort()
    print(evaljson_list)
    import multiprocessing
    try:
        with ThreadPoolExecutor(64) as executor:
            results = list(
                tqdm(executor.map(process_file, evaljson_list), total=len(evaljson_list), desc="Processing and saving files"))
    except Exception as e:
        print(e)
```

#### T5 Evaluation
```python
from sentence_transformers import SentenceTransformer, util
import numpy as np
import os, json, glob
import copy
import pprint
import traceback

def T5_similarity(output_sequence=None, chocies_list = None):
    sentences = [output_sequence]
    sentences2 = chocies_list
    model = SentenceTransformer('sentence-transformers/sentence-t5-large',cache_folder='./')
    model = model.cuda()
    # model = SentenceTransformer('DrishtiSharma/sentence-t5-large-quora-text-similarity')
    embeddings = model.encode(sentences)
    embeddings2 = model.encode(sentences2)
    #Compute cosine-similarities
    cosine_scores = util.cos_sim(embeddings, embeddings2)
    index = np.argmax(cosine_scores)
    return index
dataset_qajson = {
  "Ucfcrime": f"{Eval_QA_root}/Eval_QA/Ucfcrime_QA_new.json",
  "Youcook2": f"{Eval_QA_root}/Eval_QA/Youcook2_QA_new.json",
  "TVQA": f"{Eval_QA_root}/Eval_QA/TVQA_QA_new.json",
  "MSVD": f"{Eval_QA_root}/Eval_QA/MSVD_QA_new.json",
  "MSRVTT": f"{Eval_QA_root}/Eval_QA/MSRVTT_QA_new.json",
  "Driving-decision-making": f"{Eval_QA_root}/Eval_QA/Driving-decision-making_QA_new.json",
  "NBA": f"{Eval_QA_root}/Eval_QA/NBA_QA_new.json",
  "SQA3D": f"{Eval_QA_root}/Eval_QA/SQA3D_QA_new.json",
  "Driving-exam": f"{Eval_QA_root}/Eval_QA/Driving-exam_QA_new.json",
  "MV": f"{Eval_QA_root}/Eval_QA/MV_QA_new.json",
  "MOT": f"{Eval_QA_root}/Eval_QA/MOT_QA_new.json",
  "ActivityNet": f"{Eval_QA_root}/Eval_QA/ActivityNet_QA_new.json",
  "TGIF": f"{Eval_QA_root}/Eval_QA/TGIF_QA_new.json"
}
def json_T5_eval(T5_save_folder=None, jsonfile=None):
    # dataset 的question choices answer jsonfile
    dataset_name = os.path.basename(jsonfile).split('_')[1]
    print(f'Dataset name: {dataset_name}')
    qa_choice_json = dataset_qajson[dataset_name]
    with open(qa_choice_json, 'r', encoding='utf-8') as f:
        qa_choice_data = json.load(f)

    # model output jsonfile
    with open(jsonfile, 'r', encoding='utf-8') as f:
        data = json.load(f)
    candidates = ['A', 'B', 'C', 'D', 'E', 'F']
    try:
        new_data = {}
        for qid_vid, item in data.items():
            os.makedirs(os.path.join(T5_save_folder, os.path.basename(jsonfile).split('.')[0]), exist_ok=True)
            T5_qidvid_jsonfile = os.path.join(T5_save_folder, os.path.basename(jsonfile).split('.')[0], qid_vid+'.json')
            if not os.path.exists(T5_qidvid_jsonfile):
                new_item = copy.deepcopy(item)
                output_sequence = item['output_sequence']
                video_id = item['video_id']
                qid = qid_vid.replace(f'_{video_id}', '')
                choices = qa_choice_data[qid]['choices']
                choices = [ f'{alpha}. {choice}' for alpha, choice in choices.items()]
                answer_index = T5_similarity(str(output_sequence), choices)
                T5_answer = candidates[answer_index]
                new_item['t5-answer']= T5_answer
                new_item['choices'] = choices
                pprint.pprint(new_item)
                new_data[qid_vid] = new_item
                with open(T5_qidvid_jsonfile, 'w', encoding='utf-8') as f:
                    json.dump({qid_vid:new_item}, f, indent=2)
                print(T5_qidvid_jsonfile, 'is saved!')
            else:
                print(f'{T5_qidvid_jsonfile} is existing!')
        T5_dataset_jsonfile = os.path.join(T5_save_folder, os.path.basename(jsonfile))
        with open(T5_dataset_jsonfile, 'w', encoding='utf-8') as f:
            json.dump(new_data, f, indent=2)
    except Exception as e:
        print(traceback.print_exc())
def main():
    evaljson_list = glob.glob('./Eval_results/*_eval.json', recursive=True)
    pprint.pprint(f'{len(evaljson_list)}')
    import random
    random.shuffle(evaljson_list)
    for evaljson in evaljson_list:
        try:
            json_T5_eval('T5_jsonfolder', evaljson)
        except Exception as e:
            print(e)
main()
```
## 🐳  License
Video-Bench is released under Apache License Version 2.0.
