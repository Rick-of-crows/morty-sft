## Benchmark

支持hugging-face格式模型，跑中英文Benchmark.

运行脚本
```shell
sh run_benchmark_bloom.sh
```

参数说明：
- MODEL_NAME_OR_PATH: hugging face格式模型路径
- TOKENIZER_NAME_OR_PATH: tokenizer路径
- BATCH_SIZE: 默认为8，7b以下可调到16
- MAX_SEQ_LENGTH: 最大长度
- MAX_GENERATE_TOKENS: 生成的最大tokens数
- DEVICE_ID: GPU id
- NUM_FEWSHOT: 默认为0, zero-shot
- RESULTS_PATH: 评估结果文件
- TEST_NUMBERS: 表示测试样本数量，当为-1时表示全量，如果想少量测试脚本，可以设为10或者100等，跑通整个流程
- TASK_LIST: 需要评估的benchmark，支持部分中英文，脚本中有注释

