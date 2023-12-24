import torch
import os
import shutil

def reshape_embedding_ckpt(path, path_save, resize_shape):
    ckpt_list = [i for i in os.listdir(path) if 'layer_' in i]
    ckpt_list.sort()
    assert ckpt_list != [], 'can not find any ckpt, ckpt_list is empty'
    
    for idx, ckpt in enumerate(ckpt_list):
        if idx == 0 or idx == len(ckpt_list) - 1:
            state_dict = torch.load(os.path.join(path, ckpt), map_location='cpu')
            weight_name = 'word_embeddings.weight' if idx == 0 else 'final_linear.weight'
            word_embedding = state_dict[weight_name]
            
            word_embeddings_avg = word_embedding[:32000].mean(dim=0, keepdim=True)
            new_word_embedding = torch.zeros(resize_shape, word_embedding.shape[1], dtype=word_embedding.dtype)
            new_word_embedding[:32000] = word_embedding
            new_word_embedding[32000:] = word_embeddings_avg
            
            state_dict[weight_name] = new_word_embedding
            torch.save(state_dict, os.path.join(path_save, ckpt))
        else:
            shutil.copy(os.path.join(path, ckpt), os.path.join(path_save, ckpt))
    
    mp_list = [i for i in os.listdir(path) if 'mp_rank' in i]
    assert mp_list != [], 'can not find any mp_rank file, mp_list is empty'
    for mp_rank in mp_list:
        shutil.copy(os.path.join(path, mp_rank), os.path.join(path_save, mp_rank))
    
    latest_file = '/'.join(path.split('/')[:-1])
    assert os.path.exists(os.path.join(latest_file, 'latest'))==True, 'latest file is not exists'
    shutil.copy(os.path.join(latest_file, 'latest'), os.path.join('/'.join(path_save.split('/')[:-1]), 'latest'))
    
    
if __name__=='__main__':
    path = 'checkpoints/llama-7b-deepspeed/global_step0'
    path_save = 'checkpoints/llama-7b-deepspeed-resize-embedding/global_step0'
    resize_shape = 32128
    assert os.path.exists(path)==True, 'llama deepspeed checkpoint path not exists'
    assert os.path.exists(path_save)==True, 'path to save resized checkpoints not exists'
    assert resize_shape % 128 == 0 and resize_shape > 32000, 'resize_shape need to be devisible by 128 and larger than origin llama embedding shape'
    
    reshape_embedding_ckpt(path, path_save, resize_shape)
