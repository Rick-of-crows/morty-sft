# Project 简介
Bigscience 的Bloom模型训练使用的是Megatron-Deepspeed框架，在微软原生Megatron-Deepspeed上有针对性的优化和修改

工程目录核心内容如下：
```
megatron_deepspeed/
├── CODEOWNERS
├── data                                             # 数据存放目录
├── ds_config.json
├── examples                                         # 脚本示例
├── finetune_t0_non_causal_decoder.py                # multi-task finetune脚本
├── inference                                        # 推理脚本
├── megatron                                         # megatron 核心代码
├── pretrain_bert.py
├── pretrain_gpt_no_concaten.py                      # padding sft 
├── pretrain_gpt.py                                  # pretrain sft
├── pretrain_ict.py
├── pretrain_prefix_lm_no_concaten.py
├── pretrain_prefix_lm.py
├── pretrain_t5.py
├── pretrain_vit.py
├── pyproject.toml
├── README.md
├── run_deepspeed_fp16_bloomz_7b_padding.sh        # padding sft train-script
├── run_deepspeed_fp16_bloomz_7b_pretrain.sh       # pretrain sft train-script
├── scripts_pretrain                               # cw-exp 脚本
├── tasks                                          # 传统测试集 eval脚本
├── tests
└── tools                                          # data sample build
```
Megatron-Deepspeed 使用3D 进行模型训练，其中TP 基于Megatron的， DP和PP 使用的是DeepSpeed ZerO方案

# 训练流程
## 1. 数据准备
原始数据rawdata以jsonl形式保持，默认key 为“text”
```
{"text": "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n### Instruction:\n请问中华人民共和国成立的背景是什么？\n### Response:\n中华人民共和国成立的背景是中国革命的胜利和国共内战的结束。1949年10月1日，毛泽东在北京天安门广场宣布中华人民共和国的成立，标志着中国从半殖民地半封建社会向社会主义社会的转变。在此之前，中国经历了多年的抗日战争和国共内战，国共两党在政治、经济、文化等方面的分歧越来越大，最终导致了国共内战的爆发。1949年，中共领导的人民解放军在全国范围内取得了胜利，国共内战结束，中华人民共和国成立。"}
{"text": "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n### Instruction:\n预测明天的苹果公司的股票涨跌情况。\n### Response:\n我作为AI语言模型，是不能投资的，也无法预测股票市场的涨跌情况，无法对任何投资决策提供任何建议。建议您进行自己的市场分析和研究，以做出明智的投资决策。"}
```
准备好rawdata数据后，使用tools/preprocess_data.py来进行生成训练所需的数据集
```
python tools/preprocess_data.py  --input ./cw_pretrain_v2.json  --json-keys text  --output-prefix data/cw/cw_pretrain_v2  --dataset-impl mmap  --tokenizer-type PretrainedFromHF  --tokenizer-name-or-path /workspace/tokenizer --pad-vocab-size-to 250880  --append-eod  --workers 10
```
--output-prefix data/cw/cw_pretrain_v2  指定文件生成目录，在对应 data/cw/ 目录下生成以下两个文件
```
cw_pretrain_v2_text_document.bin   # train samples
cw_pretrain_v2_text_document.idx   # sample index
```
使用其他模型时，注意tokenizer-type，例如llama模型：
```
--tokenizer-type LlamaTokenizer
```

## 2. 训练超参配置
Bloom有两种训练方式，为方便训练，以两个独立脚本实现
Pretrain ： 将所有sample 按照seql_len进行拼接  ， 使用 run_deepspeed_7b_pretrain.sh
Padding： 将sample 按照seql_len进行padding， 使用 run_deepspeed_7b_padding.sh
常用参数如下：
```
--tensor-model-parallel-size， TP_SIZE
--pipeline-model-parallel-size ， PP_SIZE， 其中 NGPU = TP_SIZE*PP_SIZE
--lr-decay-iters
--lr-warmup-iters
--lr-decay-style
--save-interval   # 可用于 Early Stop
--exit-interval
--train-iters
```
Megatron-Deepspeed 有samples和iter两种计算epoch方式，推荐使用iter
1. Padding 计算方式：
```
TRAIN_ITERS = TOTAL_SAMPLES  /  GLOBAL_BATCH_SIZE
```
2. Pretrain 计算方式：
```
TRAIN_ITERS = TOKENS_PER_EPOCH  /  SEQ_LEN / GLOBAL_BATCH_SIZE
```

TOKENS_PER_EPOCH 可以从训练日志里通过“tokens per epoch”查找得到，例如：

```
llm002: tokens per epoch: 25225014, num samples: 100864, need 5 epoch to trian.
```
TOKENS_PER_EPOCH = 25225014



其他详细参数内容可见  megatron/arguments.py
### 单机与多机训练
单机训练使用 
```
LAUNCHER="python -u -m torch.distributed.launch \
        --nproc_per_node $N_GPUS \
        --nnodes 1 \
        --master_port 60400
```
多机训练可基于Deepspeed （详见 多机多卡训练）
```
 deepspeed --hostfile=hostfile --no_ssh_check
 ```

### 中断与增量训练
SFT训练中，有可能出现机器掉卡，训练中断，或需要进行增量训练，在进行二次训练的时候，需要将模型load的路径修改为最新的CKP路径
同时，需要注释掉 CUSTOM_ARGS 参数，不重写lr-scheduler，load optim
```
CUSTOM_ARGS=" \
    --no-load-optim \
    --finetune \
    --override-lr-scheduler \
    "
```

## 3. 模型
训练完成后，模型默认保存权重和优化器状态
模型推理需将 Deepspeed格式转换为HF格式 即可，转换代码如下：
```
python ./inference/convert_bloom_meg_to_hf.py --bloom_checkpoint_path ${latest_ckp} --pytorch_dump_folder_path ${target_hf_dir} --shard_model --pretraining_tp 4
```
注意： --pretraining_tp 的参数需与模型训练的TP数保持一致

## 4. 环境部署
推荐使用Docker来部署训练及推理环境，docker镜像地址：artifact.cloudwalk.work/rd_docker_dev/nlp/bloom-megatron-deepspeed:v1.0.3

**V100 与A100 docker有所区分**

docker 运行命令如下：
```
sudo docker run -itd  --shm-size=64g --privileged --gpus=all --net host  --ulimit memlock=-1 --name yanc -v /mnt/sfs_turbo_jy/yanc:/workspace artifact.cloudwalk.work/rd_docker_dev/nlp/bloom-megatron-deepspeed:v1.0.3 bash
```

docker 运行成功后，通过以下命令，进入docker
```
sudo docker exec -it ${Container} bash
```

**多机环境**
多机环境下，V100镜像，A100镜像运行有细微差别，A100上可以直接后台运行，而V100上，需要前台运行（手动输入 yes确认）

**注意：** 新启动的docker，需要运行ssh服务，使用2222端口（可更改） 进行多机间的免密通信，ssh 服务启动命令：

```
service ssh restart
```

若出现SSH/PSDH 错误，属于多机间ssh配置环境问题，需export以下环境变量
```
export PDSH_SSH_ARGS_APPEND="-o StrictHostKeyChecking=no" 
```

免ssh 确认输入 ，需修改 ssh_config 文件， disable 掉 StrictHostKeyChecking
```
vim  /etc/ssh/ssh_config
StrictHostKeyChecking no
```
