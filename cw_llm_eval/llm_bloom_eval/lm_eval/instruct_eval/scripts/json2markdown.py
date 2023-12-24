import json
import os, sys, argparse

#   翻译	代码	信息提取	重写	生成	头脑风暴	概括	数学	聊天	分类	语义理解	逻辑
helpness_map = {
    "总分":"total_res", "翻译":"Translation", "代码":"Code", "信息提取":"Extract", \
    "重写":"Rewrite", "生成":"Generation", "头脑风暴":"Brainstorming", "概括":"Summarization", \
    "数学":"Math", "聊天":"Chat", "分类":"Classification", "语义理解":"Semantics", "逻辑":"Logic"
}
#   历史	物理	生物	艺术	化学	地理	体育	数学	乐理	语文	幻觉
honest_map = {
    "总分":"total_res", "历史":"History", "物理":"Physics", "生物":"Biology", "艺术":"Art", \
    "化学":"Chemical", "地理":"Geography", "体育":"Sports", "数学":"Math_qa", "乐理":"Music", \
    "语文":"Chinese", "幻觉":"Hallucination"
}

#	歧视	黄赌毒	偏见	虚假宣传	暴力	危险行为	政治
harmness_map = {
    "总分":"total_res", "歧视":"Hate", "黄赌毒":"Toxicity", "偏见":"Bias", "虚假宣传":"Control", \
    "暴力":"Violence", "危险行为":"Attack", "政治":"Politics"
}
#	生成	幻觉	毒性	常识	数学	头脑风暴	代码	翻译	逻辑	概括	重写	分类	语义理解
maas_map = {
    "总分":"total_res", "生成":"Generation", "幻觉":"Hallucination", "毒性":"Toxicity", "常识":"OpenQA", \
    "数学":"Math", "头脑风暴":"Brainstorming", "代码":"Code", "翻译":"Translation", \
    "逻辑":"Logic", "概括":"Summarization", "重写":"Rewrite", "分类":"Classification", "语义理解":"Semantics"
}

# 学科知识  政府报告  企业资料  法律文档  新闻报道  金融报告  软件文档  其他  多主题文档
mrc_map = {
    "总分":"total_res", "学科知识":"mrc_subject", "政府报告":"mrc_goverment", "企业资料":"mrc_company", "法律文档":"mrc_laws", \
        "新闻报道":"mrc_news", "金融报告":"mrc_finance", "软件文档":"mrc_software", "其他":"mrc_other", "多主题文档":"mrc_topic"
}

# 风格化   翻译  Prompt生成   改写   诗歌   扩写   文学   应用文   续写   解释  起名   出题   归纳  格式化  SQL
generate_map = {
    "总分":"total_res", "风格化":"Style", "翻译":"Translation", "Prompt生成":"Prompt", "改写":"Rewrite", "诗歌":"gen_poetry", "扩写":"Expand", "文学":"gen_literature", \
        "应用文":"gen_application", "续写":"generate_story", "解释":"Semantics", "起名":"Slogan", "出题":"Exam", "归纳":"Summarization", "格式化":"Format", "SQL":"Code"
}

# 生成   头脑风暴   重写   逻辑   分类   代码   数学    地理    总结    提取    翻译
instruct_map = {
    "总分":"total_res", "生成":"Generation", "头脑风暴":"Brainstorming", "重写":"Rewrite", "逻辑":"Logic", \
        "分类":"Classification", "代码":"Code", "数学":"Math", "地理":"Geography", "总结":"Summarization", "提取":"Extract", "翻译":"Translation"
}

def get_type(test_type):
    if test_type == "help":
        return helpness_map
    elif test_type == "honest":
        return honest_map
    elif test_type == "harm":
        return harmness_map
    elif test_type == "maas":
        return maas_map
    elif test_type == "mrc":
        return mrc_map
    elif test_type == "generate":
        return generate_map
    elif test_type == "instruct":
        return instruct_map
    else:
        raise RuntimeError('Unknown test_type: %s'% test_type)

def score_parse(args):
    res = json.load(open(args.i,"r"))
    #print(harmness_map.values())
    #keys = ["模型"] + list(res.keys())
    do_map = get_type(args.type)
    keys = ["模型"] + list(do_map.keys())
    model_names = args.model_names
    total_result = [["满分"]]
    for name in model_names:
        total_result.append([name])
    print(res)
    # for k, v in res.items():
    #     assert len(v) == len(model_names) + 1
    #     count = 0
    #     for kk in v:
    #         total_result[count].append(str(v[kk]))
    #         count += 1
    for key in do_map:
        value = do_map[key]
        assert value in res, value
        v = res[value]
        count = 0
        for kk in v:
            total_result[count].append(str(v[kk]))
            count += 1

    #print(total_result)
    markdown = []
    #print("keys:", keys)
    sep = "|".join(["---"]*len(keys))
    keys = "|".join(keys)
    markdown.append(keys)
    markdown.append(sep)
    for result in total_result:
        result = "|".join(result)
        markdown.append(result)
    
    print("\n".join(markdown))

def rank_parse(args):
    res = json.load(open(args.i,"r"))
    keys = ["models"] + list(res.keys())
    values = args.model_names
    for k,v in res.items():
        values.append(":".join([str(v['a_win']),str(v['b_win']),str(v['tie'])]))
    sep = "|".join(["---"]*len(keys))
    keys = "|".join(keys)
    values = "|".join(values)
    
    print("\n".join([keys,sep,values]))

if __name__=='__main__':
    parser = argparse.ArgumentParser("json2markdown")
    parser.add_argument("-i", help="instruct result path", required=True)
    parser.add_argument("-metric", help="score or rank result", required=True)
    parser.add_argument("-type", help="markdown type, help harm honest maas mrc generate instruct", required=True)
    parser.add_argument("-model_names", help="model name list", nargs="+", required=True)
    args = parser.parse_args()
    if args.metric == "score":
        score_parse(args)
    elif args.metric == "rank":
        rank_parse(args)
    else:
        raise RuntimeError('Unknown metric: %s'% args.metric)