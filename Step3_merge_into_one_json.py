import json
import os
import glob
import argparse

def merge_json(args):
  sub_folder_name_list = os.listdir(args.chatgpt_judge_files_folder)
  datastet_name_list = [ sub.replace('_chatgpt_eval', '')  for sub in sub_folder_name_list]
  datasets_dict = {}
  for i, datastet_name in enumerate(datastet_name_list):
    single_dataset_dict = {}
    jsonfiles = glob.glob(os.path.join(args.chatgpt_judge_files_folder, sub_folder_name_list[i],'*.json' ))
    for jsonfile in jsonfiles:
      with open(jsonfile, 'r', encoding='utf-8') as f:
        data = json.load(f)
        single_dataset_dict.update(data)
    
    datasets_dict[datastet_name] = single_dataset_dict
  
  with open(args.merge_file, 'w', encoding='utf-8') as f:
    json.dump(datasets_dict, f, indent=2)    
    

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--chatgpt_judge_files_folder", type=str, default="/remote-home/share/VideoBenchmark/Video_Benchmark/VLLM-3metrics/Video-LLaVA/ChatGPT_Judge")
  parser.add_argument("--merge_file", type=str, default="./Video-Bench-Input.json")
  args = parser.parse_args()
  dataset_score_dict = merge_json(args)
