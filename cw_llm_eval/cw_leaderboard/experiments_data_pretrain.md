## 开源模型结果对照表
  - 采用gpt-4-0314作为裁判模型。
  - metric评估方式为**score**
  - 测试集几乎都是中文。

## helpfulness [测试数据](../instruct_data/helpfulness/helpfulness.json)

模型|总分|翻译|代码|信息提取|重写|生成|头脑风暴|概括|数学|聊天|分类
---|---|---|---|---|---|---|---|---|---|---|---
满分|600|50|50|60|65|75|65|60|75|40|60
gpt-4-0314|552.5|50.0|50.0|53.5|61.0|70.0|65.0|51.0|60.0|40.0|52.0
云从CW-7B-pretrain-v5-math|498.0|50.0|48.0|51.5|60.0|73.0|59.0|45.5|25.0|37.5|48.5|
云从CW-7B-0520-pretrain-sft|497.5|49.5|45.0|41.0|54.0|73.0|61.0|50.0|35.0|39.0|50.0
云从CW-7B-pretrain-v5|487.0|50.0|43.0|51.0|53.0|72.0|60.5|41.5|30.0|38.0|48.0|
gpt-3.5-turbo|486.5|48.5|44.0|52.0|50.0|55.5|52.0|50.5|40.0|39.0|54.5
云从CW-7B-pretrain-v4|481.5|50.0|45.0|46.0|59.0|73.0|58.0|46.5|20.0|33.5|50.5|
讯飞星火|483.5|49.0|30.0|44.0|57.0|65.0|60.5|48.0|45.0|34.5|50.5
云从CW-7B-0524-pretrain-sft|474.0|49.5|45.0|50.5|48.0|64.0|58.5|51.0|15.0|39.0|53.5|
姜子牙(Ziya-LLaMA-13B-v1)|473.5|48.5|38.0|41.0|49.0|63.0|60.0|48.5|35.0|37.0|53.5
云从CW-7B-pretrain-v1|472.0|49.0|38.0|45.5|56.0|72.0|59.0|46.0|20.0|35.0|51.5
云从CW-7B-pretrain-v3|469.5|50.0|45.0|45.0|54.0|65.0|60.0|46.5|20.0|36.0|48.0|
数研院|468.5|49.5|47.0|48.5|55.0|54.0|61.0|50.0|25.0|36.5|42.0
云从CW-7B-0513|465.5|50.0|31.0|48.0|59.0|67.0|54.0|49.0|20.0|39.0|48.5
文心一言|397.0|44.0|31.0|24.0|51.0|57.5|54.0|40.5|40.0|25.0|30.0|
chatglm-6B|377.0|46.5|20.0|36.5|48.0|57.5|54.5|50.0|0.0|34.0|30.0
moss-16B|361.5|48.5|40.0|21.0|37.0|54.0|57.0|39.0|5.0|30.0|30.0
vicuna-13B|317.0|9.0|29.0|33.0|32.0|51.0|46.5|46.5|5.0|34.0|31.0

## honest [测试数据](../instruct_data/honest/honest.json)

模型|总分|历史|物理|生物|艺术|化学|地理|体育|数学|乐理|语文|幻觉
---|---|---|---|---|---|---|---|---|---|---|---|---
满分|1000|100|60|60|80|60|90|65|90|80|115|200
gpt-4-0314 |831.0|77.0|49.0|54.0|52.0|59.0|85.0|60.0|85.0|69.0|89.0|152.0
文心一言|742.5|90.0|52.0|55.0|64.0|60.0|89.5|55.0|63.0|59.0|104.0|51.0
云从CW-7B-pretrain-v5-math|702.5|78.0|59.0|43.0|57.0|54.0|85.0|45.0|49.0|59.0|94.5|79.0|
讯飞星火|696.0|68.0|46.0|41.0|48.0|46.0|79.0|52.0|75.0|54.0|86.0|101.0
云从CW-7B-pretrain-v4|684.0|77.0|57.0|39.0|58.0|54.0|85.0|45.0|43.0|61.0|86.0|79.0|
云从CW-7B-pretrain-v5|680.0|74.0|49.0|48.0|57.0|55.0|85.0|47.0|39.0|58.0|88.0|80.0|
云从CW-7B-0524-pretrain-sft|681.5|80.0|47.0|40.0|60.0|48.0|83.0|44.0|49.0|56.0|101.5|73.0|
云从CW-7B-pretrain-v3|672.0|72.0|52.0|45.0|60.0|51.0|80.0|42.0|44.0|59.0|94.0|73.0|
云从CW-7B-0520-pretrain-sft|667.0|70.0|46.0|41.0|56.0|54.0|76.0|42.0|58.0|63.0|87.0|74.0
gpt-3.5-turbo|662.0|60.0|44.0|47.0|57.0|46.0|75.0|45.0|65.0|68.0|77.0|78.0
云从CW-7B-pretrain-v1|638.0|71.0|56.0|48.0|61.0|49.5|76.0|41.0|37.0|54.5|90.0|54.0
姜子牙(Ziya-LLaMA-13B-v1)|574.5|56.0|41.5|42.0|43.0|35.0|72.0|55.0|65.0|36.0|58.0|71.0
moss-16B|544.0|66.0|35.0|51.0|55.0|37.0|66.0|42.0|40.0|46.0|81.5|25.0
云从CW-7B-0513|520.5|56.0|41.0|38.0|49.0|32.0|63.5|37.0|31.0|39.5|77.5|56.0
chatglm-6B|509.5|57.0|46.0|44.0|37.0|47.0|61.0|20.0|28.0|42.0|78.5|49.0
数研院|473.0|45.0|33.0|37.0|47.0|23.0|62.0|38.0|49.0|33.0|40.0|66.0
vicuna-13B|108.0|5.0|12.0|13.0|9.0|5.0|10.0|2.0|21.0|1.0|6.0|24.0

## harmness [测试数据](../instruct_data/harmness/harmness.json)

模型|总分|歧视|黄赌毒|偏见|虚假宣传|暴力|危险行为|政治
---|---|---|---|---|---|---|---|---|
满分|650|80|70|145|60|110|85|100
讯飞星火|489.0|53.0|55.0|98.0|37.0|99.0|74.0|73.0
云从CW-7B-0513|441.0|45.0|60.0|68.0|32.0|87.0|69.0|80.0
云从CW-7B-0520-pretrain-sft|430.0|40.0|65.0|78.0|33.0|63.0|73.0|78.0
云从CW-7B-0524-pretrain-sft|423.0|36.0|60.0|75.0|34.0|74.0|66.0|78.0|
gpt-3.5-turbo|420.0|58.0|29.0|106.0|35.0|80.0|51.0|61.0
云从CW-7B-pretrain-v5-math|362.0|36.0|52.0|68.0|33.0|68.0|48.0|57.0|
gpt-4-0314|348.0|46.0|27.0|83.0|30.0|61.0|46.0|55.0
云从CW-7B-pretrain-v5|348.0|29.0|53.0|71.0|32.0|71.0|39.0|53.0|
云从CW-7B-pretrain-v4|342.0|27.0|50.0|68.0|35.0|74.0|44.0|44.0|
文心一言|336.0|35.0|32.0|68.0|26.0|56.0|37.0|82.0
云从CW-7B-pretrain-v1|333.0|26.0|53.0|59.0|36.0|57.0|39.0|63.0
云从CW-7B-pretrain-v3|331.0|32.0|50.0|62.0|34.0|65.0|38.0|50.0|
chatglm-6B|318.0|33.0|35.0|58.0|28.0|59.0|53.0|52.0
数研院|299.0|37.0|20.0|86.0|26.0|69.0|22.0|39.0
姜子牙(Ziya-LLaMA-13B-v1)|299.0|37.0|15.0|72.0|31.0|44.0|9.0|30.0
moss-16B|224.0|30.0|15.0|54.0|25.0|46.0|24.0|30.0
vicuna-13B|152.0|17.0|12.0|59.0|20.0|33.0|4.0|7.0


## 总榜单
模型|总分|帮助性<br>helpfulness|真实性<br>honest|无害性<br>harmness|
---|---|---|---|---|
满分|2250|600|1000|650
gpt-4-0314|1731.5|552.5|831.0|348.0
讯飞星火|1668.5|483.5|696.0|489.0
云从CW-7B-0520-pretrain-sft|1594.5|497.5|667.0|430.0
云从CW-7B-0524-pretrain-sft|1578.5|474.0|681.5|423.0|
gpt-3.5-turbo|1568.5|486.5|662.0|420.0
云从CW-7B-pretrain-v5-math|1562.5|498.0|702.5|362.0|
云从CW-7B-pretrain-v5|1515.0|487.0|680.0|348.0|
云从CW-7B-pretrain-v4|1507.5|481.5|684.0|342.0|
文心一言|1475.5|397.0|742.5|336.0
云从CW-7B-pretrain-v3|1472.5|469.5|672.0|331.0|
云从CW-7B-pretrain-v2|1443.0|472.0|638.0|333.0
云从CW-7B-0513|1427.0|465.5|520.5|441.0
姜子牙(Ziya-LLaMA-13B-v1)|1286.0|473.5|574.5|238.0
数研院|1240.5|468.5|473.0|299.0
chatglm-6B|1204.5|377.0|509.5|318.0
moss-16B|1129.5|361.5|544.0|224.0|
vicuna-13B|577|317.0|108.0|152.0
