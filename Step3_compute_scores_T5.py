import json, os, glob
import pprint
import traceback
import tqdm
import argparse

def T5_json(args):
    folder_list = [ os.path.join(args.T5_judge_files_folder, sub_folder) for sub_folder in os.listdir(args.T5_judge_files_folder)]
    dataset_list_eval_dict = {}
    for folder in folder_list:
        if not os.path.isdir(folder):
            continue
        print(folder)

        
        folder_jsonlist = glob.glob(f'{folder}/*.json')
        # 每一个folder下面重新数一次
        dataset_eval_dict = {}
        correct = 0
        total_num = 0  
        
        for jsonfile in tqdm.tqdm(folder_jsonlist):
          with open(jsonfile, 'r', encoding='utf-8') as f:
            data = json.load(f)
            #其实只有一个{key:value}
            for qid_vid, item in data.items():
              total_num += 1
              if item["t5-answer"]==item["correct"]:
                correct += 1 
        dataset_eval_dict['total_num'] = total_num
        dataset_eval_dict['correct'] = correct
        dataset_eval_dict['score'] = correct/total_num
        dataset_list_eval_dict[folder] = dataset_eval_dict
    
    with open(args.score_output_file, 'w', encoding='utf-8') as f:
        json.dump(dataset_list_eval_dict, f, indent=2)
    print(f'{args.score_output_file} is saved!')
    

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--T5_judge_files_folder", type=str, default="./T5_Judge")
  parser.add_argument("--score_output_file", type=str, default="./Final_score_table_T5.json")
  args = parser.parse_args()
  T5_json(args)
