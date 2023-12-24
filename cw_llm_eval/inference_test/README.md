## inference

模型推理脚本仅支持hugging-face格式的模型文件。

### deepspeed to hf

将训练好的bloom模型文件格式转成hugging-face格式，转换代码源于transformers.models.bloom.convert_bloom_original_checkpoint_to_pytorch.py文件，运行脚本：

```shell
sh run_convert_bloom_to_hf.sh
```

参数说明：
- BLOOM_PATH: 训练好的bloom模型路径；
- MODEL_PATH：转好的模型路径；
- pretraining_tp: TP_SIZE, 和训练的时候保持一致。

备注：**模型转换成功后，需要在生成好的模型文件夹里手动更改config.json文件里的部分参数与模型训练时一致，否则里面的部分参数取值是初始化的。主要更改hidden_size, n_head,n_layer.**


### inference

运行脚本：
```shell
sh run_inference_bloom.sh
```

参数说明：
- DATA_PATH：测试数据路径，json格式
- OUT_PATH： 输出结果路径， txt格式
- MODEL_PATH： hugging-face格式的模型文件，即上面生成的模型文件
- TOEKENIZER_PATH： tokenizer路径，包括tokenizer.json, tokenizer_config.json, special_tokens_map.json文件

备注：注意tokenizer文件要一致，否则会推理失败。
