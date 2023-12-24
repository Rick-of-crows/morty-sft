# pip env
pip install lm-eval -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install evaluate -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install sqlitedict -i https://pypi.tuna.tsinghua.edu.cn/simple

set -x

export SACREBLEU=./caches
export TRANSFORMERS_CACHE=./caches/models
export HF_DATASETS_CACHE=./caches/datasets
export HF_MODULES_CACHE=./caches/modules
export HF_METRICS_CACHE=./caches/metrics
export TOKENIZERS_PARALLELISM=Fasle

VARIANT="bloom-7b1-chinese"
MODEL_NAME_OR_PATH="./output_dir/transformers/bloom-7b1"
TOKENIZER_NAME_OR_PATH="./output_dir/transformers/bloom-7b1"
BATCH_SIZE=8
MAX_SEQ_LENGTH=2048
MAX_GENERATE_TOKENS=256
DEVICE_ID="5"
NUM_FEWSHOT=0
RESULTS_PATH=$VARIANT-results.json
TEST_NUMBERS=-1

TASK_LIST="ocnli,cmnli,afqmc,tnews,nlpcc" # 中文benchmark; 也支持cmrc2018,drcd,dureader 超级慢且指标差
# TASK_LIST="cb,rte,anli_r1,anli_r2,anli_r3,mnli,qqp,mrpc,sst,race,multirc,record,webqs,boolq,mathqa,piqa,openbookqa" # 英文Benchmark


python ./evaluate_benchmark_bloom.py \
--model_name_or_path ${MODEL_NAME_OR_PATH} \
--tokenizer_name_or_path ${TOKENIZER_NAME_OR_PATH} \
--batch_size ${BATCH_SIZE} \
--max_seq_length ${MAX_SEQ_LENGTH} \
--max_generate_tokens ${MAX_GENERATE_TOKENS} \
--fp16 \
--device_id ${DEVICE_ID} \
--num_fewshot ${NUM_FEWSHOT} \
--task_list ${TASK_LIST} \
--test_numbers ${TEST_NUMBERS} \
--results_path ${RESULTS_PATH} 2>&1 | tee $VARIANT-eval-harness.log
