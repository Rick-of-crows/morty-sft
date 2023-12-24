# Language Model Evaluation Harness:

## 简介
lm_eval 基于 [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) 扩展，目前支持megatron、nemo等框架测试.

## 概述

Features:

- 支持多种框架推理,如megatron、nemo以及huggingface模型等.
- 支持多种模型测试,如GPT-2, GPT-3, GPTNeo.
- 支持多种测试任务,如文本推理、情感分类、机器翻译、文章摘要、人机对话等.

## 安装依赖

- git clone https://gitlab-research.cloudwalk.work/chatgpt/lm_eval

- cd lm_eval

- python setup.py install

## 适配测试

- 根据不同框架实现adaptor类，目的是告诉测试工具，该框架下如何进行inference以及tokenizer的实现，并重载对应函数.

- 提供megatron和nemo的实现样例供参考，详见adaptor文件夹.

## examples
以nemo框架为例:

- cd ./lm_eval

- 修改./adaptor/nemo/gpt2_1.5b_eval.sh为您对应的测试配置.
  - **NEMO_FILE**:测试模型路径.
  - **TEST_DATA_DIR**:测试数据路径.
  - **NUM_FEWSHOT**: few-shot数目，设置为0即zero-shot.
  - **MAX_GENERATE_NUM**: 文本生成类任务时，最大生成的token数.
  - **RATIO**: 生成mini测试集的比例，0~1之间的浮点数，默认1.0全量测试。

- sh ./adaptor/nemo/gpt2_1.5b_eval.sh 进行测试.


## 支持的metric
| 指标名称 | 范围 | 简介 | 
|---|---|---|
| acc | [0,1] | accuracy, 正确命中的数量除以总数.
| acc_norm | [0,1] | 归一化后的accuracy，主要适用于multiple choice类任务。通常情况下针对选择题，会把每个答案代入上下文，分别计算loglikelihood sum，概率最高的即为答案。acc_norm会在这个概率基础上分别除以token的数量，避免模型倾向于选择token数量较多的答案。
| ppl | > 0 | perplexity，主要是根据每个词来估计一句话出现的概率，本质用的交叉熵来计算，ppl越低表示预测单词和句子的越准，性能越好。
| f1 | [0,1] | 模糊匹配结果, 计算预测出的答案与原始答案字符之间的overlap, 根据overlap的数目与原始ground truth字符数目计算回召率;overlap的数目与预测出的字符数目计算准确率.
| em | [0,1] | exact match, 精确匹配结果,也就是模型给出的答案与标准答案一模一样才算正确。
| bleu | [0,100] | bilingual evaluation understudy，用于机器翻译任务的评价，分别计算candidate句和reference句的N-grams模型精度，BLEU分数是基于1~4-gram的精度平均.
| rouge-N|[0,1]| 常用于文章摘要任务评价，将模型生成的结果和标准结果按N-gram拆分后，计算召回率。一般来说N越大，分值容易越小。
| rouge-L|[0,1]| 常用于文章摘要任务评价，生成结果与reference最长公共子序列的重合率计算。


## 支持的task

- 开源模型在lm_eval上测试结果,可作为您的基线参考[benckmark](./models/experiments_data.md)

|      Task Name       |  评价指标       | train | valid | test |      简介     |
|----------------------|-----------------|----|----|----|---------------------|
| lambada_openai | acc, ppl | | | ✓|上下文的理解问题，预测给定段落的最后一个词。|
| wikitext       | ppl      | ✓| ✓| ✓|由维基百科的长文组成, 在长文章中进行滑窗，预测下一个单词。|
| hellaswag      | acc, acc_norm | ✓| ✓| ✓|阅读理解任务，它由一个段落和四个结尾组成，涉及选择故事或指令集的最佳结局.|
| cbt-cn         | acc      | ✓| ✓| ✓|Children’s Book Test，从每一个儿童故事中提取20个连续的句子作为文档，第21个句子作为问题，并从中剔除一个实体类单词作为答案，cn代表的是以普通名词为答案.
| cbt-ne         | acc      | ✓| ✓| ✓|Children’s Book Test，ne代表的是以命名实体为答案.
| mnli           | acc      | ✓| ✓| ✓|GLUE子集，自然语言推断任务，给定前提语句和假设语句，根据前提判断假设是否成立，矛盾或者不能推断,三分类.|
| rte            | acc      |✓| ✓| ✓ |GLUE子集, 同样是自然语言推断任务，将矛盾和不能推断统一转换为不能推断，变为二分类任务.
| boolq          | acc      | ✓| ✓| ✓|SuperGLUE子集，一项QA任务，每个例子由一段短文和一个关于这段话的是/否问题组成，回答yes or no.
| wsc273         | acc      |✓ |✓ |✓|SuperGLUE子集, Winograd Schema Challenge, 是一项指代消解任务，其中的例子包括一个带有代词的句子和该句子中的名词短语列表。系统必须从提供的选择中确定代词的正确所指。
| winogrande     | acc      |✓ |✓ |✓|指代消解任务，明确句子中代词所指，消除歧义，二分类。
| coqa           | f1, em   | ✓| ✓||Conversational Question Answering Challenge, 多轮对话问答数据集,注重模型在对话过程中回答关联问题的能力，答案类型包含四类：yes、no、unknown、描述性短语.
| wmt14-en-fr    | bleu     | | | ✓|机器翻译测试集，英语转法语
| wmt14-fr-en    | bleu     | | | ✓|机器翻译测试集，法语转英语
| arc_easy | acc, acc_norm |✓|✓|✓| AI2 Reasoning Challenge, ARC数据集包含7787个真实的纯文字科学测验多项选择题（美国3到9年级水平，通常有4个答案选项）
| arc_challenge | acc, acc_norm |✓|✓|✓| 只包含了用统计和检索算法不能正确回答的问题,更具有难度.
| race_middle | acc, acc_norm |✓|✓|✓|中国中学生英语阅读理解题目，给定一篇文章和5道4选1的选择题，初中难度.
| race_high | acc, acc_norm |✓|✓|✓|中国中学生英语阅读理解题目，高中难度.
|piqa| acc, acc_norm |✓|✓|✓|Physical Interaction Question Answering, 关于物理常识推理的任务，从两个选项中选出符合常识的答案.
| cnn_dailymail | rouge1, rouge2, rougeL | ✓|✓|✓| 文章摘要任务，测试集涵盖11488篇CNN每日新闻文章.
|triviaqa | acc | ✓|✓ | | 问答任务，但问题的答案无法通过上下文直接推断得到，需要模型根据常识直接回答.
|hendrycksTest-*|acc, acc_norm||✓|✓|mmlu任务，涵盖57个子任务，英文版，选择题|
|cmmlu-*|acc, acc_norm ||✓|✓|中文版mmlu任务，选择题|
|ceval-*|acc, acc_norm||✓|✓|中文版，选择题，52个学科任务|
|gaokao-*|acc, acc_norm|||✓|中文版，选择题，高考单项选择题|
|AGIEval-*|acc, acc_norm|||✓|涵盖高考题、公务员考试、司法考试、sat考试单项选择题|