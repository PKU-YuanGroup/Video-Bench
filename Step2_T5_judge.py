from sentence_transformers import SentenceTransformer, util
import numpy as np
import os, json, glob
import copy
import pprint
import argparse

def T5_similarity(output_sequence=None, chocies_list = None):
    sentences = [output_sequence]
    sentences2 = chocies_list
    model = SentenceTransformer('sentence-transformers/sentence-t5-large', cache_folder='/remote-home/share/VideoBenchmark/Video_Benchmark/T5_evaluation')
    model = model.cuda()
    embeddings = model.encode(sentences)
    embeddings2 = model.encode(sentences2)
    #Compute cosine-similarities
    cosine_scores = util.cos_sim(embeddings, embeddings2)
    index = np.argmax(cosine_scores)
    return index

import traceback
def json_T5_eval(T5_save_folder=None, jsonfile=None, args=None):
    dataset_qajson = {
                    "Ucfcrime": f"{args.Eval_QA_root}/Eval_QA/Ucfcrime_QA_new.json",
                    "Youcook2": f"{args.Eval_QA_root}/Eval_QA/Youcook2_QA_new.json",
                    "TVQA": f"{args.Eval_QA_root}/Eval_QA/TVQA_QA_new.json",
                    "MSVD": f"{args.Eval_QA_root}/Eval_QA/MSVD_QA_new.json",
                    "MSRVTT": f"{args.Eval_QA_root}/Eval_QA/MSRVTT_QA_new.json",
                    "Driving-decision-making": f"{args.Eval_QA_root}/Eval_QA/Driving-decision-making_QA_new.json",
                    "NBA": f"{args.Eval_QA_root}/Eval_QA/NBA_QA_new.json",
                    "SQA3D": f"{args.Eval_QA_root}/Eval_QA/SQA3D_QA_new.json",
                    "Driving-exam": f"{args.Eval_QA_root}/Eval_QA/Driving-exam_QA_new.json",
                    "MV": f"{args.Eval_QA_root}/Eval_QA/MV_QA_new.json",
                    "MOT": f"{args.Eval_QA_root}/Eval_QA/MOT_QA_new.json",
                    "ActivityNet": f"{args.Eval_QA_root}/Eval_QA/ActivityNet_QA_new.json",
                    "TGIF": f"{args.Eval_QA_root}/Eval_QA/TGIF_QA_new.json"
        }
    # dataset 的question-choices-answer jsonfile
    dataset_name = os.path.basename(jsonfile).split('_eval.json')[0]
    print(f'Dataset name: {dataset_name}')
    qa_choice_json = dataset_qajson[dataset_name]
    with open(qa_choice_json, 'r', encoding='utf-8') as f:
        qa_choice_data = json.load(f)

    # model chat jsonfile
    with open(jsonfile, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    candidates = ['A', 'B', 'C', 'D', 'E', 'F']
    try:
        new_data = {}
        for qid_vid, item in data.items():
            # 单独qa_t5 eval结果保存
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

        #一个model的一个dataset 所有qa保存
        T5_dataset_jsonfile = os.path.join(T5_save_folder, os.path.basename(jsonfile))
        with open(T5_dataset_jsonfile, 'w', encoding='utf-8') as f:
            json.dump(new_data, f, indent=2)
    except Exception as e:
        print(traceback.print_exc())
        import ipdb
        ipdb.set_trace()
 
def main(args):
    evaljson_list = glob.glob(f'{args.model_chat_files_folder}/*_eval.json', recursive=True)
    print(f'{len(evaljson_list)}') #{evaljson_list},
    for evaljson in evaljson_list:
        try:
            json_T5_eval(args.T5_judge_output_folder, evaljson, args)
        except Exception as e:
            print(e)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_chat_files_folder", type=str, default="./Chat_results")
    parser.add_argument("--T5_judge_output_folder", type=str, default="./T5_Judge")
    parser.add_argument("--Eval_QA_root", type=str, default="/remote-home/share/VideoBenchmark/Video_Benchmark")
    args = parser.parse_args()
    main(args)
