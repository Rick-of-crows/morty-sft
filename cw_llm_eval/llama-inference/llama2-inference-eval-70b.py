from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import LlamaTokenizer
import transformers
import torch
import argparse
import time
import re
import json

def main(ARGS):


    model = f"{ARGS.model}"
    #model = "falcon-sft"

    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)

    test_list = [f"{ARGS.data_path}/helpfulness/helpfulness.json", f"{ARGS.data_path}/honest/honest.json", f"{ARGS.data_path}/harmness/harmness.json"]
    generator = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )

    model_input = ""

    for test_data_path in test_list:
        print(f"*** Processing data {test_data_path}")

        data = None
        try:
            data  = json.load(open(test_data_path, "r", encoding="utf-8"))
        except:
            data=[]
            with open(test_data_path, 'r', encoding='utf-8') as fr:
                for line in fr:
                    data.append(json.loads(line.strip('\n')))


        for d in data:
            instruction = d["instruction"]
            model_input = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n ### Instruction:\n{instruction}\n### Response:"

            start = time.time()
            #sequences = generator(model_input, max_new_tokens=1024, do_sample=False, top_k=ARGS.top_k, num_return_sequences=1, eos_token_id=tokenizer.eos_token_id)
            # outputs = model.generate(**batch, max_new_tokens=1024, top_p=1.0, do_sample=False, use_cache=True, top_k=10, repetition_penalty=1.1)
            sequences = generator(model_input, max_new_tokens=1024, do_sample=False, use_cache=True, top_k=50, repetition_penalty=1.1, num_return_sequences=1)
            inference_time = time.time() - start
            
            for seq in sequences:
                #m = re.search("### Response:", seq["generated_text"])
                #resp = seq['generated_text'][m.end():].strip()
                resp = seq['generated_text'][len(model_input):].strip()
                end = len(resp)

                # for falcon
                # m = re.search("### Feedback:",  seq["generated_text"])
                # for llama
                m = re.search("### Instruction:",  seq["generated_text"])

                if m and m.end():
                    end = m.end()
                
                resp = resp[:end]

                d.update({"output": resp})
                print(f"{'-'*60}\n{resp}\n")

        
        result_output = ARGS.output_path + '/result-' + test_data_path.split('/')[-1]
        with open(result_output, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-m',
                        '--model',
                        type=str,
                        default="40b",
                        help="select 7b or 40b version of falcon")
    parser.add_argument('-d',
                        '--data_path',
                        type=str,
                        default="lm_eval/instruct_eval/instruct_data",
                        help="lm_eval path")
    parser.add_argument('-o',
                        '--output_path',
                        type=str,
                        default="./result",
                        help="output path")
    parser.add_argument('-ml',
                        '--max_length',
                        type=int,
                        default="1024",
                        help="used to control the maximum length of the generated text in text generation tasks")
    parser.add_argument('-tk',
                        '--top_k',
                        type=int,
                        default="10",
                        help="specifies the number of highest probability tokens to consider at each step")

    ARGS = parser.parse_args()
    main(ARGS)
