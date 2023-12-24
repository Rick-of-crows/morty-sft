## LLM BLOOM Eval

For Eval each Bloom-SFT models based on Cloudwalk Test DataSets with GPT-4 as juge


### Environment Setup

- Docker images

  `git pull artifact.cloudwalk.work/rd_docker_dev/nlp/bloom-megatron-deepspeed:v1.0.5`

- Dependence repo

  Eval based on lm_eval project developed by wangxi@cloudwalk.com  

  `git clone https://gitlab-research.cloudwalk.work/chatgpt/lm_eval.git`

  `cp eval_config/* lm_eval/instruct_eval/configs/`

- OpenAI account
 
  To use GPT-4, need set API_KEY first, this canbe export in current shell env or set in eval script

```
OPENAI_API_KEY=""

ORGANIZATION=""

OPENAI_API_BASE=""
```


### Run Eval 

- Download latest models with tokenizer
  
  Notice: tokenizer and models are in the same dir

- run scripts
 
  `run_bloom_score.sh ${MODEL_PATH}`

  result will create at current dir in outputs_${MODEL_NAME}

