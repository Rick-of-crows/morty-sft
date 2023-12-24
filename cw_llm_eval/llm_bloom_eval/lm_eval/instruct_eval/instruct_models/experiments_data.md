## 开源模型结果对照表
  - 采用gpt-4-0314作为裁判模型。
  - metric评估方式为**score**
  - 测试集几乎都是中文。
  - 由于每个类别的数据不均衡，归一化后的表格见 [benchmark](./norm_experiments_data.md)

## helpfulness [测试数据](../instruct_data/helpfulness/helpfulness.json)
模型|总分|翻译|代码|信息提取|重写|生成|头脑风暴|概括|数学|聊天|分类|语义理解|逻辑
---|---|---|---|---|---|---|---|---|---|---|---|---|---
满分|750.0|50.0|50.0|60.0|65.0|75.0|65.0|60.0|75.0|40.0|60.0|75.0|75.0
gpt-4-0314|659.5|50.0|45.0|56.5|56.0|74.0|53.0|51.5|70.0|38.5|53.0|59.0|53.0
gpt-3.5-turbo|618.0|49.5|49.0|51.5|57.0|74.0|51.0|51.5|55.0|38.0|58.5|46.0|37.0
CW_176B_0612|589.0|49.5|42.0|46.0|60.0|71.0|51.0|49.0|25.0|38.5|54.0|56.0|47.0
讯飞星火|583.5|47.5|34.0|47.5|56.0|73.0|50.0|38.5|70.0|37.5|52.5|47.0|30.0
claude|578.0|46.0|42.0|49.0|56.0|64.5|48.0|49.0|60.0|39.5|53.0|33.0|38.0
文心一言|560.0|49.0|36.0|47.0|55.0|63.0|48.0|41.0|50.0|34.0|53.0|65.0|19.0
数研院-0607|540.5|49.5|36.0|48.5|53.0|72.5|50.0|49.5|30.0|38.0|49.5|41.0|23.0
云从CW-7B-0607|532.5|49.5|40.0|50.0|53.0|73.5|50.0|49.5|20.0|35.0|52.0|41.0|19.0
姜子牙(Ziya-LLaMA-13B-v1)|522.0|49.5|38.0|41.0|52.0|66.0|51.0|48.0|35.0|35.5|54.0|34.0|18.0
chatglm-130B|469.5|45.0|36.0|45.5|36.0|60.0|48.0|31.5|20.0|38.5|39.0|54.0|16.0
aquila_7b|447.5|47.5|34.0|36.0|52.0|61.0|44.0|48.0|5.0|33.0|44.0|34.0|9.0
chatglm-6B|428.5|46.5|34.0|36.5|48.0|60.5|50.0|50.0|0.0|34.0|30.0|36.0|3.0
moss-16B|397.5|48.5|40.0|21.0|37.0|65.0|47.0|39.0|5.0|30.0|30.0|26.0|9.0
BloomChat-175B|371.0|47.0|27.0|29.0|27.0|44.0|46.0|35.0|5.0|22.0|36.0|38.0|15.0
vicuna-13B|350.0|9.0|32.0|33.0|32.0|60.5|40.0|46.5|5.0|34.0|31.0|15.0|12.0

## honest [测试数据](../instruct_data/honest/honest.json)
模型|总分|历史|物理|生物|艺术|化学|地理|体育|数学|乐理|语文|幻觉
---|---|---|---|---|---|---|---|---|---|---|---|---
满分|1000.0|100.0|60.0|60.0|80.0|60.0|90.0|65.0|90.0|80.0|115.0|200.0
gpt-4-0314|851.0|79.0|54.0|54.0|55.0|59.0|72.0|60.0|85.0|67.0|93.0|173.0
讯飞星火|842.0|90.0|54.0|56.0|70.0|54.0|89.0|64.0|88.0|67.0|106.0|104.0
文心一言|837.0|93.0|59.5|53.0|74.0|59.0|84.0|64.0|77.5|64.0|104.0|105.0
CW_176B_0612|758.5|81.0|49.0|50.0|66.5|59.0|85.0|53.0|50.0|67.0|98.0|100.0
gpt-3.5-turbo|754.0|63.0|56.5|51.0|63.0|54.0|74.0|41.0|76.0|70.0|75.5|130.0
chatglm-130B|750.0|88.0|55.0|56.0|67.0|50.0|82.0|49.0|58.0|68.0|108.0|69.0
云从CW-7B-0607|731.5|75.0|58.0|51.0|54.0|54.0|81.5|55.0|64.0|52.0|108.0|79.0
BloomChat-175B|600.0|83.0|44.0|43.0|66.0|40.0|74.0|41.0|23.0|52.0|93.0|41.0
claude|597.0|46.0|48.0|51.0|49.0|43.0|52.0|50.0|71.0|47.0|52.0|88.0
姜子牙(Ziya-LLaMA-13B-v1)|574.5|56.0|41.5|42.0|43.0|35.0|72.0|55.0|65.0|36.0|58.0|71.0
数研院-0607|551.5|54.0|37.0|42.0|38.0|33.0|77.0|40.0|63.5|46.0|49.0|72.0
moss-16B|544.5|66.0|35.0|51.0|55.0|37.0|66.0|42.0|40.0|46.0|81.5|25.0
aquila_7b|515.0|62.0|37.0|39.0|49.0|27.0|68.0|39.0|20.0|42.0|78.0|54.0
chatglm-6B|509.5|57.0|46.0|44.0|37.0|47.0|61.0|20.0|28.0|42.0|78.5|49.0
vicuna-13B|108.0|5.0|12.0|13.0|9.0|5.0|10.0|2.0|21.0|1.0|6.0|24.0

## harmness [测试数据](../instruct_data/harmness/harmness.json)
模型|总分|歧视|黄赌毒|偏见|虚假宣传|暴力|危险行为|政治
---|---|---|---|---|---|---|---|---
满分|650.0|80.0|70.0|145.0|60.0|110.0|85.0|100.0
claude|547.0|68.0|57.0|129.0|53.0|99.0|72.0|69.0
文心一言|543.0|54.0|70.0|116.0|37.0|106.0|74.0|86.0
gpt-3.5-turbo|506.0|56.0|55.0|106.0|40.0|94.0|80.0|75.0
讯飞星火|482.0|48.0|50.0|108.0|40.0|87.0|62.0|87.0
chatglm-130B|473.0|48.0|55.0|103.0|34.0|99.0|45.0|89.0
云从CW-7B-0607|456.0|44.0|60.0|93.0|35.0|77.0|63.0|84.0
CW_176B_0612|448.0|35.0|65.0|94.0|40.0|82.0|60.0|72.0
gpt-4-0314|431.0|42.0|51.0|87.0|40.0|81.0|57.0|73.0
数研院-0607|341.0|37.0|27.0|96.0|26.0|70.0|35.0|50.0
aquila_7b|318.0|27.0|45.0|57.0|25.0|64.0|49.0|51.0
chatglm-6B|318.0|33.0|35.0|58.0|28.0|59.0|53.0|52.0
BloomChat-175B|315.0|46.0|33.0|86.0|36.0|41.0|27.0|46.0
姜子牙(Ziya-LLaMA-13B-v1)|238.0|37.0|15.0|72.0|31.0|44.0|9.0|30.0
moss-16B|224.0|30.0|15.0|54.0|25.0|46.0|24.0|30.0
vicuna-13B|152.0|17.0|12.0|59.0|20.0|33.0|4.0|7.0

## 总榜单
模型|总分|帮助性<br>helpfulness|真实性<br>honest|无害性<br>harmness
---|---|---|---|---
满分|2400.0|750.0|1000.0|650.0
gpt-4-0314|1941.5|659.5|851.0|431.0
文心一言|1940.0|560.0|837.0|543.0
讯飞星火|1907.5|583.5|842.0|482.0
gpt-3.5-turbo|1878.0|618.0|754.0|506.0
CW_176B_0612|1795.5|589.0|758.5|448.0
claude|1722.0|578.0|597.0|547.0
云从CW-7B-0607|1720.0|532.5|731.5|456.0
chatglm-130B|1692.5|469.5|750.0|473.0
数研院-0607|1433.0|540.5|551.5|341.0
姜子牙(Ziya-LLaMA-13B-v1)|1334.5|522.0|574.5|238.0
BloomChat-175B|1286.0|371.0|600.0|315.0
aquila_7b|1280.5|447.5|515.0|318.0
chatglm-6B|1256.0|428.5|509.5|318.0
moss-16B|1166.0|397.5|544.5|224.0
vicuna-13B|610.0|350.0|108.0|152.0



## maas测试集 [测试数据](../instruct_data/web_test/maas_test.json)
模型|总分|生成|幻觉|毒性|常识|数学|头脑风暴|代码|翻译|逻辑|概括|重写|分类|语义理解
---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
满分|990|100|70|45|190|110|280|80|15|70|10|5|10|5
gpt-4-0314|840.0|92.0|59.0|42.0|145.5|90.0|225.0|80.0|13.5|68.0|8.0|5.0|8.0|4.0
gpt-3.5-turbo|774.5|88.0|52.0|38.0|125.0|85.0|213.0|80.0|13.5|53.0|9.0|5.0|8.0|5.0
文心一言|749.0|93.0|36.0|39.0|154.0|75.0|205.0|65.0|8.0|48.0|8.0|5.0|8.0|5.0
讯飞星火|723.5|81.0|43.0|34.0|137.0|80.0|209.0|57.0|8.5|57.0|4.0|5.0|8.0|0.0
claude|707.5|89.5|46.0|35.0|98.0|75.0|202.0|77.0|13.0|52.0|8.0|5.0|7.0|0.0
云从CW-7B-0607|688.0|84.0|49.0|38.0|138.5|55.0|195.0|57.0|13.5|36.0|8.0|5.0|9.0|0.0
云从从容-maas|654.0|81.0|47.0|37.0|135.5|45.0|188.0|58.0|13.0|27.5|8.0|5.0|9.0|0.0
chatglm-130B|624.5|82.0|45.0|38.0|127.5|50.0|181.0|46.0|8.5|29.0|4.0|3.0|7.5|3.0
数研院-0607|609.0|85.0|39.0|27.0|84.0|80.0|175.0|55.0|12.0|32.0|4.0|5.0|8.0|3.0
姜子牙(Ziya-LLaMA-13B-v1)|602.5|79.5|26.0|22.0|99.0|80.0|184.0|40.0|11.0|37.0|8.0|5.0|8.0|3.0
moss-16b |535.0|81.0|27.0|37.0|85.0|20.0|179.0|65.0|8.0|16.0|9.0|0.0|5.0|3.0
chatglm-6b|521.5|74.0|29.0|31.0|86.0|30.0|184.0|45.0|11.5|20.0|8.0|0.0|3.0|0.0
vicuna-13b|359.5|63.0|12.0|30.0|37.0|30.0|100.0|36.0|12.5|26.0|6.0|0.0|7.0|0.0


## MRC阅读理解测试集 [测试数据](../instruct_data/MRC/mrc.json)
模型|总分|学科知识|政府报告|企业资料|法律文档|新闻报道|金融报告|软件文档|其他|多主题文档
---|---|---|---|---|---|---|---|---|---|---
满分|485|80|60|45|55|65|75|50|45|10
gpt4-0613|429.0|62.0|55.0|42.0|51.0|58.0|72.0|48.0|32.0|9.0
gpt3.5-turbo-0613|375.0|52.0|52.0|34.0|38.0|56.0|63.0|44.0|27.0|9.0
文心一言|337.0|60.0|30.0|37.0|35.0|36.0|57.0|35.0|38.0|9.0
讯飞星火|329.5|40.0|50.0|31.0|38.0|50.0|50.5|35.0|26.0|9.0
文心一言-turbo|326.0|51.0|39.0|33.0|35.0|45.0|50.0|44.0|23.0|6.0
cloudwalk-llama-0807|319.5|46.0|34.0|24.0|42.5|44.0|50.0|43.0|27.0|9.0

## 文本生成测试集 [测试数据](../instruct_data/generation/generation.json)
模型|总分|风格化|翻译|Prompt生成|改写|诗歌|扩写|文学|应用文|续写|解释|起名|出题|归纳|格式化|SQL
---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---
满分|310|40|30|20|15|20|10|20|15|15|15|25|20|20|40|5
gpt4-0613|265.0|31.0|27.0|15.0|14.0|18.0|8.0|19.0|14.5|15.0|14.0|25.0|15.0|18.5|26.0|5.0
gpt3.5-turbo-0613|229.0|27.0|25.0|15.0|12.0|12.0|8.0|16.0|12.5|13.0|13.0|24.0|6.0|17.5|23.0|5.0
文心一言|201.5|34.0|15.5|9.0|13.0|12.0|5.0|16.0|15.0|13.0|9.0|18.0|12.0|13.0|17.0|0.0
讯飞星火|198.5|24.0|16.5|5.0|13.0|8.0|3.0|15.0|14.0|12.0|13.0|20.0|14.0|17.0|19.0|5.0
cloudwalk-llama-0807|175.0|23.0|14.0|12.0|6.5|8.0|8.0|14.0|13.0|12.0|13.0|18.0|7.0|15.5|9.0|2.0

## 指令理解测试集 [测试数据](../instruct_data/instruction_understanding/instruct.json)
模型|总分|生成|头脑风暴|重写|逻辑|分类|代码|数学|地理|总结|提取|翻译
---|---|---|---|---|---|---|---|---|---|---|---|---
满分|500|135|65|50|50|45|35|40|10|25|25|20
gpt4-0613|381.0|110.5|40.0|23.5|33.0|39.0|33.0|30.0|9.5|19.5|25.0|18.0
gpt3.5-turbo-0613|354.5|107.5|44.0|19.0|33.0|29.5|29.0|25.0|10.0|22.5|21.0|14.0
文心一言|311.0|92.0|40.0|15.0|22.0|29.0|26.0|20.0|6.0|22.5|23.5|15.0
讯飞星火|308.5|101.5|34.0|16.0|25.0|30.5|25.0|25.0|2.0|22.5|15.0|12.0
cloudwalk-0807|277.0|99.5|44.0|12.0|14.0|29.0|14.0|10.0|2.0|18.5|20.0|14.0