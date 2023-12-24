# pip env
pip install accelerate -i https://pypi.tuna.tsinghua.edu.cn/simple some-package
pip install loguru -i https://pypi.tuna.tsinghua.edu.cn/simple some-package


export PYTHONPATH=./


VARIANT="bloom-1b1" # a unique identifier for the current eval ideally correspnding to the modelname

DATA_PATH=./inference/cw_eval.json # test data path

OUT_PATH=./inference/test_$VARIANT.json # res path

MODEL_PATH=/yckj4437/yanchao/yanc/bloom-1b1 # hugging-face model path

TOEKENIZER_PATH=/yckj4437/yanchao/yanc/bloom-1b1 # tokenizer path which contains tokenizer.json, tokenizer_config.json, special_tokens_map.json


deepspeed --num_gpus 1 ./inference/bloom-ds-zero-inference.py \
--gpu_id "2" \
--model_name_or_path $MODEL_PATH \
--tokenizer_name_or_path $TOEKENIZER_PATH \
--data_path $DATA_PATH \
--res_path $OUT_PATH \
--batch_size 1 2>&1 | tee $VARIANT-inference.txt
