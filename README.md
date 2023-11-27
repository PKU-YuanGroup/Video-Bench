
[Arxiv](https://arxiv.org/abs/2307.16125)
<p align="center">
    <img src="assets/pie_fig.jpg" width="300" style="margin-bottom: 0.2;"/>
<p>
<h2 align="center"> <a href="https://arxiv.org/abs/2311.10122">Video-Bench: A Comprehensive Benchmark and Toolkit for Evaluating Video-based Large Language Models</a></h2>
<!-- <h5 align="center"> If you like our project, please give us a star ⭐ on GitHub for latest update.  </h2> -->


 
 SEED-Bench consists of 19K multiple-choice questions with accurate human annotations, covering 12 evaluation dimensions
including both the spatial and temporal understanding.
## News
**[2023.11.27]** SEED-Bench is released! Data and evaluation code is available now.

## Leaderboard
Welcome to [Video_Benchmark Leaderboard](https://github.com/munanning/Video_Benchmark)!

### Evaluation

The code below is just a generalized framework for dataset evaluation, you will need to refine the model loading part according to your own model. Once the code execution is complete, you will find some JSON files named 'Eval/{dataset_name}.json'. Then you can utilize ChatGPT or T5 model as experts to assess the correctness of the model's output answer. 

```python 
Eval_QA_root = '/remote-home/share/VideoBenchmark/Video_Benchmark'
Eval_Video_folder = '/remote-home/share/VideoBenchmark/Video_Benchmark'

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
  "MOT": "../../Eval_QA/MOT_QA_new.json",
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
                vid_path = os.path.join(Eval_Video_folder, vid_rela_path)
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

Then you can upload 'results.json' in [SEED-Bench Leaderboard](https://huggingface.co/spaces/AILab-CVC/SEED-Bench_Leaderboard).

After submitting, please press refresh button to get the latest results.

## Data Preparation

You can download the data of SEED-Bench released on HuggingFace repo [SEED-Bench](https://huggingface.co/datasets/AILab-CVC/SEED-Bench).
Please refer to [DATASET.md](DATASET.md) for image and video data preparation.

## Installation

Please refer to [INSTALL.md](INSTALL.md).

## Run Evaluation

The evaluation metric is provided in [eval.py](eval.py). We use [InstructBLIP](https://arxiv.org/abs/2305.06500) as an example. To run the following evaluation code, please refer to [repo](https://github.com/salesforce/LAVIS) for the environment preparation.

```shell
python eval.py --model instruct_blip --anno_path SEED-Bench.json --output-dir results --task all
```

After the evaluation is finished, you can obtain the accuracy of each evaluation dimension and also 'results.json' in 'results' folder, which can be submitted to [SEED-Bench Leaderboard](https://huggingface.co/spaces/AILab-CVC/SEED-Bench_Leaderboard).

If you want to evaluate your own models, please provide the interface like [instruct_blip_interface.py](https://github.com/AILab-CVC/SEED-Bench/blob/main/model/instruct_blip_interface.py).

Note that to evaluate models with multiple-choice questions, we adopt the answer ranking strategy
following GPT-3. Specifically, for each choice of a question, we compute the likelihood 
that a model generates the content of this choice given the question. 
We select the choice with the highest likelihood as model's prediction. 
Our evaluation strategy does not rely on the instruction-following capabilities 
of models to output 'A' or 'B' or 'C' or 'D'.



## License
SEED-Bench is released under Apache License Version 2.0.

## Declaration
For the images of SEED-Bench, we use the data from Conceptual Captions Dataset (https://ai.google.com/research/ConceptualCaptions/)
following its license (https://github.com/google-research-datasets/conceptual-captions/blob/master/LICENSE).
Tencent does not hold the copyright for these images and the copyright belongs to the original owner of Conceptual Captions Dataset. 

For the videos of SEED-Bench, we use tha data from Something-Something v2 (https://developer.qualcomm.com/software/ai-datasets/something-something),
Epic-kitchen 100 (https://epic-kitchens.github.io/2023) and 
Breakfast (https://serre-lab.clps.brown.edu/resource/breakfast-actions-dataset/). We only provide the video name. Please download them in their official websites.
