from transformers import LlamaTokenizer,LlamaForCausalLM
import transformers
import torch
import argparse
import time
import re
import json

def main(ARGS):


    model = f"{ARGS.model}"
    #model = "falcon-sft"
    '''
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    generator = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )
    '''

    test_list = [f"{ARGS.data_path}/helpfulness/helpfulness.json", f"{ARGS.data_path}/honest/honest.json", f"{ARGS.data_path}/harmness/harmness.json"]

    #tokenizer = LlamaTokenizer.from_pretrained(model)
    tokenizer = LlamaTokenizer.from_pretrained(model, add_eos_token=True, legacy=False)
    #tokenizer = LlamaTokenizer.from_pretrained(model,legacy=False)
    #tokenizer.add_special_tokens({"pad_token": "<PAD>"})
    model = LlamaForCausalLM.from_pretrained(model, return_dict=True, device_map="auto")
    model.eval()

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

            prefix = tokenizer.encode("Below is an instruction that describes a task. Write a response that appropriately completes the request.\n### Instruction:")
            sep = tokenizer.encode("\n### Response:\n")
            human = tokenizer.encode("\nHuman: ")
            ai = tokenizer.encode("\nAssistant: ")
            que = tokenizer.encode('\n'+ instruction)
 
            tokens = [prefix + que + sep]
            att_m = [[1]*len(tokens[0])]
            batch = {'input_ids': torch.LongTensor(tokens),'attention_mask':torch.LongTensor(att_m)}

            #batch = tokenizer(model_input, return_tensors="pt")
            #sequences = generator(model_input, max_length=ARGS.max_length, do_sample=False, top_k=ARGS.top_k, num_return_sequences=1, eos_token_id=tokenizer.eos_token_id)
            batch = {k: v.to("cuda") for k, v in batch.items()}

            with torch.no_grad():
                start = time.time()
                outputs = model.generate(**batch, max_new_tokens=1024, top_p=1.0, do_sample=False, use_cache=True, top_k=50, repetition_penalty=1.1)
                inference_time = time.time() - start

                output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

                resp = output_text.split("### Response:")[1].strip()

                end = len(resp)

                # for falcon
                # m = re.search("### Feedback:",  seq["generated_text"])
                # for llama
                m = re.search("### Response:",  resp)

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
                        default="512",
                        help="used to control the maximum length of the generated text in text generation tasks")
    parser.add_argument('-tk',
                        '--top_k',
                        type=int,
                        default="10",
                        help="specifies the number of highest probability tokens to consider at each step")

    ARGS = parser.parse_args()
    main(ARGS)
