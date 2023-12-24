import instruct_models
import os, sys, argparse
from utils import *
from tqdm import tqdm

def player_func(args):
    data = json_load(args.data)
    model_config = yaml_load(args.config)
    assert args.player in model_config['player']
    val = model_config['player'][args.player]
    assert 'type' in val and 'config' in val
    type_name = val['type']
    config = val['config']
    bs = args.bs
    save_path = os.path.join(args.output, args.player + ".json")
    ### do model init
    model = instruct_models.get_model(type_name, config)
    ### generate
    batch_text = []
    items = []
    save_flg = True
    for k, item in enumerate(tqdm(data)):
        #instruct = item["instruction"]
        #if k < 76:
        #    continue
        #print(instruct)
        #instruct = item["instruction"]
        items.append(item)
        if ((k + 1)) % bs == 0 or k == len(data) - 1:
            ### pad for inference
            outs = model.base_generate(items)
            if outs is None:
                save_flg = False
                continue
            assert len(outs) == len(items)
            for i in range(len(items)):
                items[i]['output'] = outs[i]
                print(f"================= out_{i} ================\n{outs[i]}\n===============================")
            items = []
    if save_flg:
        json_dump(save_path, data, ensure_ascii=False)

if __name__=='__main__':
    parser = argparse.ArgumentParser("instruct eval")
    parser.add_argument("-data", help="instruct data path", required=True)
    parser.add_argument("-config", help="model config path", required=True)
    parser.add_argument("-player", help="which player do task", required=True)
    parser.add_argument("-output", help="result path", required=True)
    parser.add_argument("-bs", help="batch size", type=int, default=1, required=False)
    args = parser.parse_args()
    player_func(args)