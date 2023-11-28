import json, os, glob
import pprint
import traceback
import tqdm
import argparse

def chatgpt_json(args):
    folder_list = [ os.path.join(args.chatgpt_judge_files_folder, sub_folder) for sub_folder in os.listdir(args.chatgpt_judge_files_folder)]
    dataset_score_dict = {}
    for folder in folder_list:
        dataset_name = os.path.basename(folder).replace('_chatgpt_eval','')
        folder_jsonlist = glob.glob(folder+'/*.json')
        # 每一个folder下面重新数一次
        correct = 0
        total_num = 0
        for jsonfile in tqdm.tqdm(folder_jsonlist):
          with open(jsonfile, 'r', encoding='utf-8') as f:
            data = json.load(f)
            #其实只有一个{key:value}
            for qid_vid, item in data.items():
              total_num += 1
              if item["output_chatgpt_choice"]==item["correct"]:
                correct += 1 
        score = correct/total_num
        dataset_score_dict[dataset_name] = score
    return dataset_score_dict
    #  
def compute_scores(args, dataset_score_dict):
  dataset_weight = {
    1:
      {
      "ActivityNet":1,
      "MSVD":1,
      "MSRVTT":1,
      "TGIF":1,
      "Youcook2":1,
      "Ucfcrime":1,
      "MOT":0.5,
      },
    
    2:
      {
        "TVQA":1,
        "MV":1,
        "NBA":1,
      },
      
    3:
      {
        "Driving-exam":0.5,
        "Driving-decision-making":1,
        "SQA3D":1,
      }
    
  }
  
  
  # Video-exclusive Understanding score
  exclusive_understanding_weight = dataset_weight[1]
  weights_sum = sum(exclusive_understanding_weight.values())
  exclusive_understanding_score = 0
  # import ipdb; ipdb.set_trace()
  for dataset_name, weight in exclusive_understanding_weight.items():
    exclusive_understanding_score += weight * dataset_score_dict[dataset_name] / weights_sum
  
  # Prior Knowledge-based Question-answer
  prior_QA_weight = dataset_weight[2]
  weights_sum = sum(prior_QA_weight.values())
  prior_QA_score = 0
  for dataset_name, weight in prior_QA_weight.items():
    prior_QA_score += weight * dataset_score_dict[dataset_name] / weights_sum
    
  # Comprehension and Decision-making
  com_and_dec_QA_weight = dataset_weight[3]
  weights_sum = sum(com_and_dec_QA_weight.values())
  com_and_dec_QA_score = 0
  for dataset_name, weight in com_and_dec_QA_weight.items():
    com_and_dec_QA_score += weight * dataset_score_dict[dataset_name] / weights_sum
  
  dataset_score_dict['Exclusive_understanding'] = exclusive_understanding_score
  dataset_score_dict['Prior_Knowledge'] = prior_QA_score
  dataset_score_dict['Comprehension_and_Decision-making'] = com_and_dec_QA_score
  
  
  # final score 
  final_score = sum([exclusive_understanding_score, prior_QA_score, com_and_dec_QA_score])/3
  dataset_score_dict['final_score'] = final_score
    
  
  print(dataset_score_dict)
  with open(args.score_output_file, 'w', encoding='utf-8') as f:
    json.dump(dataset_score_dict, f, indent=2)
  print(f'{args.score_output_file} is saved!')
    


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--chatgpt_judge_files_folder", type=str, default="/remote-home/share/VideoBenchmark/Video_Benchmark/VLLM-3metrics/Video-LLaVA/ChatGPT_Judge")
  parser.add_argument("--score_output_file", type=str, default="./Final_score_table.json")
  args = parser.parse_args()
  dataset_score_dict = chatgpt_json(args)
  compute_scores(args, dataset_score_dict)
  
    
