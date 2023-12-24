
def get_model(model_type, config):
    if model_type == "bloom":
        from .bloom import BLOOM as LM
    elif model_type == "chatglm":
        from .chatglm import ChatGLM as LM
    elif model_type == "openai":
        from .openai_gpt import ChatGPT as LM
    elif model_type == "nemo":
        from .chat_nemo import ChatNeMo as LM
    elif model_type == "vicuna":
        from .vicuna import VICUNA as LM
    elif model_type == "llama_api":
        from .llama_api import LLamaApi as LM
    elif model_type == "wenxin":
        from .wenxin import WenXin as LM
    elif model_type == "moss":
        from .moss import MOSS as LM
    elif model_type == "shuyan":
        from .shuyan import ShuYan as LM
    elif model_type == "ziya":
        from .ziya import Ziya as LM
    elif model_type == "maas":
        from .maas import Maas as LM
    elif model_type == "maas_engine":
        from .maas_engine import MaasEngine as LM
    elif model_type == "cwchain":
        from .cwchain_model import CwChainModel as LM
    elif model_type == "aquila":
        from .chat_aquila import Aquila as LM
    elif model_type == "bloom_api":
        from .bloom_api import BloomApi as LM
    elif model_type == "xunfei":
        from .xunfei import XunFei as LM
    else:
        raise RuntimeError('Unknown model_type: %s'% model_type)

    model = LM(config)
    return model