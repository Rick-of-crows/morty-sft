import sys
import json

helpful_md = "模型|总分|翻译|代码|信息提取|重写|生成|头脑风暴|概括|数学|聊天|分类|语义理解|逻辑\n---|---|---|---|---|---|---|---|---|---|---|---|\n满分|600|50|50|60|65|75|65|60|75|40|60|75|75\n"
honest_md = "模型|总分|历史|物理|生物|艺术|化学|地理|体育|数学|乐理|语文|幻觉|\n---|---|---|---|---|---|---|---|---|---|---|---|---|\n满分|1000|100|60|60|80|60|90|65|90|80|115|200|\n"
harmness_md = "模型|总分|歧视|黄赌毒|偏见|虚假宣传|暴力|危险行为|政治|\n---|---|---|---|---|---|---|---|---|\n满分|650|80|70|145|60|110|85|100|\n"
total_md = "模型|总分|帮助性<br>helpfulness|真实性<br>honest|无害性<br>harmness|\n---|---|---|---|---|\n满分|2250|600|1000|650|\n"

helpful_data = json.load(open(sys.argv[1] + "/helpful/reviewer1_result.json", "r", encoding="utf-8"))
honest_data = json.load(open(sys.argv[1] + "/honest/reviewer1_result.json", "r", encoding="utf-8"))
harm_data = json.load(open(sys.argv[1] + "/harmness/reviewer1_result.json", "r", encoding="utf-8"))

model_name = sys.argv[1].split('_')[-1].strip()

harm_score = harmness_md + "%s|%.1f|%.1f|%.1f|%.1f|%.1f|%.1f|%.1f|%.1f|\n"%(model_name,\
							     harm_data['total_res']['player1_score'], \
                                                             harm_data['Hate']['player1_score'], \
                                                             harm_data['Toxicity']['player1_score'], \
                                                             harm_data['Bias']['player1_score'], \
                                                             harm_data['Control']['player1_score'], \
                                                             harm_data['Violence']['player1_score'], \
                                                             harm_data['Attack']['player1_score'], \
                                                             harm_data['Politics']['player1_score'])



honest_score = honest_md + "%s|%.1f|%.1f|%.1f|%.1f|%.1f|%.1f|%.1f|%.1f|%.1f|%.1f|%.1f|%.1f|\n"%(model_name, \
                                                                                                honest_data['total_res']['player1_score'], \
                                                                                                honest_data['History']['player1_score'], \
                                                                                                honest_data['Physics']['player1_score'], \
                                                                                                honest_data['Biology']['player1_score'], \
                                                                                                honest_data['Art']['player1_score'], \
                                                                                                honest_data['Chemical']['player1_score'], \
                                                                                                honest_data['Geography']['player1_score'], \
                                                                                                honest_data['Sports']['player1_score'], \
                                                                                                honest_data['Math_qa']['player1_score'], \
                                                                                                honest_data['Music']['player1_score'], \
                                                                                                honest_data['Chinese']['player1_score'], \
                                                                                                honest_data['Hallucination']['player1_score'])

helpful_score = helpful_md + "%s|%.1f|%.1f|%.1f|%.1f|%.1f|%.1f|%.1f|%.1f|%.1f|%.1f|%.1f|%.1f|%.1f|\n"%(model_name, \
                                                                                             helpful_data['total_res']['player1_score'], \
                                                                                             helpful_data['Translation']['player1_score'], \
                                                                                             helpful_data['Code']['player1_score'], \
                                                                                             helpful_data['Extract']['player1_score'], \
                                                                                             helpful_data['Rewrite']['player1_score'], \
                                                                                             helpful_data['Generation']['player1_score'], \
                                                                                             helpful_data['Brainstorming']['player1_score'], \
                                                                                             helpful_data['Summarization']['player1_score'], \
                                                                                             helpful_data['Math']['player1_score'], \
                                                                                             helpful_data['Chat']['player1_score'], \
                                                                                             helpful_data['Classification']['player1_score'], \
                                                                                             helpful_data['Semantics']['player1_score'], \
                                                                                             helpful_data['Logic']['player1_score'])


total_score = total_md + "%s|%.1f|%.1f|%.1f|%.1f|\n"%(model_name, \
                                                       harm_data['total_res']['player1_score'] + honest_data['total_res']['player1_score'] + helpful_data['total_res']['player1_score'], \
                                                       helpful_data['total_res']['player1_score'], honest_data['total_res']['player1_score'], harm_data['total_res']['player1_score'])

with open(sys.argv[1] + '/' + model_name+"_result.md",'w', encoding="utf-8") as f:
    f.write(helpful_score +'\n\n')
    f.write(honest_score +'\n\n')
    f.write(harm_score +'\n\n')
    f.write(total_score + '\n\n')
