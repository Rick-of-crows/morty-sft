#/bin/sh

DATA_PATH=./lm_eval/instruct_eval/instruct_data/MRC/mrc.json # test data path
OUT_PATH=./result
RES_PATH=$OUT_PATH/result-mrc.json # save res path, must be the same as the name in mrc.yaml
MODEL_PATH=/yckj4437/yckj4437/models/transformers/llama2-cw-20230816


mkdir $OUT_PATH

python ./inference/llm_inference.py \
--gpu_id "0,1" \
--model_type "llama" \
--model_name_or_path $MODEL_PATH \
--tokenizer_name_or_path $MODEL_PATH \
--data_path $DATA_PATH \
--res_path $RES_PATH \


rm -rf lm_eval/instruct_eval/result
mv result lm_eval/instruct_eval/

cd lm_eval/instruct_eval/
mkdir -p outputs/mrc

# export OPENAI_API_KEY=""
# export ORGANIZATION=""
# export OPENAI_API_BASE="https://shanghai-chat-service-open-api.ai2open.win/v1"

python eval.py -data instruct_data/MRC/mrc.json -config configs/mrc.yaml -output outputs/mrc