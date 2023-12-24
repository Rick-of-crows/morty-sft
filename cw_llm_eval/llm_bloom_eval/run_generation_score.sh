#/bin/sh

DATA_PATH=./lm_eval/instruct_eval/instruct_data/generation/generation.json # test data path
OUT_PATH=./result
RES_PATH=$OUT_PATH/result-generation.json # save res path, must be the same as the name in generation.yaml
MODEL_PATH=/yckj4437/yckj4437/models/transformers/bloomz-7b-20230607


mkdir $OUT_PATH

python ./inference/llm_inference.py \
--gpu_id "0" \
--model_type "bloom" \
--model_name_or_path $MODEL_PATH \
--tokenizer_name_or_path $MODEL_PATH \
--data_path $DATA_PATH \
--res_path $RES_PATH \


rm -rf lm_eval/instruct_eval/result
mv result lm_eval/instruct_eval/

cd lm_eval/instruct_eval/
mkdir -p outputs/generation

# export OPENAI_API_KEY=""
# export ORGANIZATION=""
# export OPENAI_API_BASE="https://shanghai-chat-service-open-api.ai2open.win/v1"

python eval.py -data instruct_data/generation/generation.json -config configs/generation.yaml -output outputs/generation