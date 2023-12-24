## zero-shot任务

- 文本生成类任务默认设置最大生成token数：MAX_GENERATE_NUM=32

#### 文本推理
Setting | lambada<br>(ppl)| lambada<br>(acc)| cbt-cn<br>(acc) | cbt-ne<br>(acc) | wikitext<br>(ppl) | hellaswag<br>(acc_norm) |
---|---|---|---|---|---|---|
gpt2-medium | 18.2 | 42.7 | 77.8 | 81.1 | 26.7 | 39.4 | 
nemo-gpt-345m(nv开源) | 13.9 | 46.3 | 85.1 | 80.1| 22.6 | 38.9
nemo-gpt-1.3B(nv开源) | 8.4 | 55.8 | 87.5 | 80.6| 19.6 | 44.2
nemo-gpt-5B(nv开源) |4.4 | 67.0 | 91.2 | 85.9 | 13.2 | 62.0
llama-7B | 3.5 | 73.8 | 91.4 | 86.5 | 10.2 | 77.1
bloom-7B1 | 6.6 | 57.5 | 83.2 | 86.9 | 16.1 | 59.6
bloomz-7B1 | 6.7 | 55.8 | 87.8| 79.9 | 19.3 | 63.0
falcon-7B | 3.3|74.6|84.6|89.7|10.9|76.3
falcon-40B | 2.8|77.2 |88.0|92.6|8.4|82.8

#### 多轮问答
Setting | coqa<br>(f1)| coqa<br>(em) |
---|---|---|
gpt2-medium | 42.7 | 32.7
nemo-gpt-345m(nv开源) | 38.7 | 29.3 
nemo-gpt-1.3B(nv开源) | 55.2 | 42.9
nemo-gpt-5B(nv开源)| 68.3 | 56.6 | 
llama-7B | 74.6 | 62.1
bloom-7B1 | 66.9 | 51.3
bloomz-7B1 | 58.5 | 48.9
falcon-7B |72.5|59.8
falcon-40B|80.0|68.1|

#### 机器翻译

Setting | wmt14(en-fr)<br>(bleu) | wmt14(fr-en)<br>(blue) | 
---|---|---|
gpt2-medium | 0.61 | 0.80 | 
nemo-gpt-345m(nv开源) | 0.87 | 1.22 | 
nemo-gpt-1.3B(nv开源) | 3.8 | 20.8 | 
nemo-gpt-5B(nv开源) | 10.7 | 28.3
llama-7B | 17.1 | 28.9
bloom-7B1 | 7.8 | 26.7
bloomz-7B1 | 13.9 | 22.0
falcon-7B |18.8|30.6
falcon-40B|26.0|34.9|

#### 文章摘要

Setting | cnn_dailymail<br>(rouge1) | cnn_dailymail<br>(rouge2) | cnn_dailymail<br>(rougeL)|
---|---|---|---|
nemo-gpt-345m(nv开源) | 21.8 | 6.1 | 15.6
nemo-gpt-1.3B(nv开源) | 24.6 | 8.2 | 17.8
nemo-gpt-5B(nv开源) | 25.1 | 8.5 | 18.1
llama-7B | 24.7 | 8.4 | 17.9
bloom-7B1 | 17.2 | 5.6 | 12.5
falcon-7B|24.1|8.3|17.6

#### 中文
Setting | c3<br>(acc_norm)  | wiki_zh<br>(ppl)|
---|---|---|
llama-7B|44.6|20.4|
bloomz-7B1|73.2|7.7
falcon-7B|49.5|9.2|
falcon-40B|55.8|7.3|

## few-shot任务

- 默认few-shot设置K=5.

#### 常识推理
Setting | piqa<br>(acc_norm)<br>zero-shot/few-shot | arc_easy<br>(acc_norm)<br>zero-shot/few-shot| arc_challenge<br>(acc_norm)<br>zero-shot/few-shot
---|---|---|---|
gpt2-medium | 66.3 | 43.6 | 25.0 | 
nemo-gpt-345m | 66.8/66.3 | 42.4 / 48.4 | 23.8/ 24.7
nemo-gpt-1.3B | 68.6/68.1| 49.2 /57.2 | 25.7/ 26.7
nemo-gpt-5B |76.2/75.2| 61.2/69.4 | 33.7/ 37.8 |
llama-7B | 78.8/80.4| 71.9/78.7 | 44.4/49.3
bloom-7B1 | 73.7/73.4 | 57.3/68.7 | 33.4/39.2
bloomz-7B1 | 77.5 | 72.9 | 43.6
falcon-7B | 80.6 | 70.9 | 43.1
falcon-40B |83.2|79.5|54.6

#### 阅读理解
Setting | race_middle<br>(acc_norm)<br>zero-shot/few-shot | race_high<br>(acc_norm)<br>zero-shot/few-shot | coqa<br>(f1)<br>zero-shot/few-shot
---|---|---|---|
gpt2-medium | 38.1/56.6 | 32.7/ | 42.7
nemo-gpt-345m | 38.1/38.7 | 32.8/33.0 | 38.7/37.2
nemo-gpt-1.3B | 41.4/42.8 | 35.9/36.2 | 55.2/55.0
nemo-gpt-5B | 49.2/47.8 | 40.0/41.7 | 68.3/68.9
llama-7B | 52.5/54.7 | 44.9/44.9 | 74.6/74.2
bloom-7B1| 50.3/49.7 | 41.4/41.6 | 66.9/67.9
bloomz-7B1 | 61.0 | 51.7 | 58.5
falcon-7B | 51.9 | 43.7|72.5
falcon-40B |58.0|50.9|80.0|

#### 文本蕴含
Setting | rte<br>(acc)<br>zero-shot/few-shot |
---|---|
gpt2-medium | 52.7 |
nemo-gpt-345m | 53.8/ 49.5
nemo-gpt-1.3B| 54.5/57.8
nemo-gpt-5B | 54.8/59.2
llama-7B | 58.1/62.8
bloom-7B1 |51.2/53.4
bloomz-7B1 | 83.3
falcon-7B | 54.8
falcon-40B|60.2

#### 指代消解
Setting | wsc273<br>(acc)<br>zero-shot/few-shot | winogrande<br>(acc)<br>zero-shot/few-shot
---|---|---|
gpt2-medium | 62.6 | 52.8
nemo-gpt-345m | 69.2/71.8 | 52.4/53.0
nemo-gpt-1.3B | 67.8/71.8 | 53.4/54.6
nemo-gpt-5B | 82.8/82.4| 61.2/63.8
llama-7B | 88.3/90.1 | 70.2/71.3
bloom-7B1 |80.9/80.9 | 64.3/65.7
bloomz-7B1 | 77.6 | 65.2
falcon-7B | 84.6 |67.3
falcon-40B |89.3|76.7

#### 闭卷问答
Setting | triviaqa<br>(acc)<br>zero-shot/few-shot  |
---|---|
nemo-gpt-345m |2.8/7.2
nemo-gpt-1.3B | 4.3/8.5
nemo-gpt-5B | 13.9/22.4
llama-7B | 37.6/46.2
bloom-7B1 | 5.8/19.6
bloomz-7B1 | 26.1
falcon-7B | 34.1
falcon-40B | 48.2