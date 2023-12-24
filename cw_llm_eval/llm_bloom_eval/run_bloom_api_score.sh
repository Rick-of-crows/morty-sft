#/bin/sh

 python inference/bloom-api.py  --ip 120.48.37.175 --port 41701 --data_path ./lm_eval/instruct_eval/instruct_data --res_path ./result


if [ "x"$1 == "x" ]; then
    echo "Usage: run_bloom_inference.sh MODE_PATH"
    exit;
fi

MODEL_NAME=`echo $1 | awk -F '/' '{print $NF}'`
echo $MODEL_NAME

if [ "x"$MODEL_NAME == "x" ]; then
    echo "ERROR: Empty model file"
    exit;
fi

DATA_PATH=./lm_eval/instruct_eval/instruct_data # test data path
OUT_PATH=./result

# hf model with tokenizer all in MODEL_PATH
MODEL_PATH=$1 
TOEKENIZER_PATH=$1 # tokenizer path which contains tokenizer.json, tokenizer_config.json, special_tokens_map.json

mkdir $OUT_PATH

deepspeed --num_gpus 1 ./inference/bloom-ds-zero-inference.py \
--gpu_id "1" \
--model_name_or_path $MODEL_PATH \
--tokenizer_name_or_path $TOEKENIZER_PATH \
--data_path $DATA_PATH \
--res_path $OUT_PATH \
--batch_size 1

rm -rf lm_eval/instruct_eval/result
mv result lm_eval/instruct_eval/

cd lm_eval/instruct_eval/
rm -rf outputs
mkdir -p outputs/honest
mkdir -p outputs/helpful
mkdir -p outputs/harmness


export OPENAI_API_KEY=""
export ORGANIZATION=""
export OPENAI_API_BASE="https://shanghai-chat-service-open-api.ai2open.win/v1"

python eval.py -data instruct_data/honest/honest.json -config configs/bloomz-nemo-sft-honest.yaml -output outputs/honest
python eval.py -data instruct_data/helpfulness/helpfulness.json -config configs/bloomz-nemo-sft-helpful.yaml -output outputs/helpful
python eval.py -data instruct_data/harmness/harmness.json -config configs/bloomz-nemo-sft-harm.yaml -output outputs/harmness

cd -
mv lm_eval/instruct_eval/outputs outputs_${MODEL_NAME}

python convert-md.py outputs_${MODEL_NAME}
