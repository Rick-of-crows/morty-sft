BLOOM_PATH=/workspace/2023-05-28/bloomz-7b-cw-padding-v13-2/global_step11862 # bloom path global_step6744  global_step6768  global_step6792  global_step6816
MODEL_PATH=/workspace/transformers/7b-models/0528/bloomz-sft-v13-v2 # hugging-face model path

mkdir -p $MODEL_PATH

# deepspeed-->hf
python ./convert_bloom_meg_to_hf.py \
--bloom_checkpoint_path $BLOOM_PATH \
--pytorch_dump_folder_path $MODEL_PATH \
--shard_model \
--pretraining_tp 2
