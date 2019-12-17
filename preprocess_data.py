# -*-coding:UTF-8 -*-
from transformers import BertTokenizer
import json
import random
import torch
from torch.utils.data import TensorDataset

# 讀取數據
def LoadJson(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        AllData = json.load(f)
    return AllData

def convert_data_to_feature(filepath):

    DRCD = LoadJson(filepath)
    tokenizer = BertTokenizer(vocab_file='bert-base-chinese-vocab.txt')

    context_tokens = []
    context_loss_tokens = []
    max_seq_len = 0

    # BertForMaskedLM的訓練需要特殊符號('[MASK]')以及被mask掉的詞的id
    for data in DRCD["data"]:
        for paragraph in data["paragraphs"]:
            context = paragraph["context"]
            do = True # 用於檢測做滑窗後context有沒有大於250以及沒有做滑窗則要做將context轉換成input_id和loss_id
            # 當context大於500時，將window size設為500，並且每250做一次滑窗
            while(len(context) >= 500):
                # 擷取500個字
                little_context = context[:500]
                # 將context轉換成input_id和loss_id
                max_seq_len = conversion_context(little_context, tokenizer, max_seq_len, context_tokens, context_loss_tokens)
                # 滑窗為250個字
                context = context[250:]
                if len(context) <= 250:
                    do = False # context小於等於250，就不用將context轉換成input_id和loss_id

            # 將context轉換成input_id和loss_id
            if do:
                max_seq_len = conversion_context(context, tokenizer, max_seq_len, context_tokens, context_loss_tokens)
    
    print("最長內容長度:",max_seq_len)
    print("有" + str(len(context_tokens)) + "筆資料")
    assert max_seq_len <= 512 # 小於BERT-base長度限制

    # 補齊長度
    for c in context_tokens:
        while len(c)<max_seq_len:
            c.append(0)

    for c_l in context_loss_tokens:
        while len(c_l)<max_seq_len:
            c_l.append(-1)
    
    # BERT input embedding
    input_ids = context_tokens
    loss_ids = context_loss_tokens
    assert len(input_ids) == len(loss_ids)
    data_features = {'input_ids':input_ids,
                    'loss_ids':loss_ids}

    return data_features

# 將context轉換成input_id和loss_id
def conversion_context(context, tokenizer, max_seq_len, context_tokens, context_loss_tokens):
    Mask_id_list = []
    new_word_piece_list = []
    # 內容一定要有做MASK
    while len(Mask_id_list) == 0:
        word_piece_list = tokenizer.tokenize(context)
        # 隨機的將詞給調換成mask或其他的詞
        random_change_word_piece(tokenizer, word_piece_list, Mask_id_list, new_word_piece_list)

    # new_word_piece_list經過id轉換再在頭尾添加特殊符號的id
    bert_ids = tokenizer.build_inputs_with_special_tokens(tokenizer.convert_tokens_to_ids(new_word_piece_list))
    # bert_loss_ids會紀錄兩種情況，一種是沒有被mask掉的詞，所以會記錄-1，另一種是會記錄被mask掉的詞的id
    bert_loss_ids = []
    bert_loss_ids.append(-1)     # [CLS]的-1
    Mask_id_count = 0
    for word_piece in new_word_piece_list:
        if word_piece == '[MASK]':
            try:
                bert_loss_ids.append(Mask_id_list[Mask_id_count])
                Mask_id_count = Mask_id_count + 1
            except :
                print(new_word_piece_list)
                print(Mask_id_count)
                print(Mask_id_list)
                assert Mask_id_count < len(Mask_id_list)
        else:
            bert_loss_ids.append(-1)

    bert_loss_ids.append(-1)     # [SEP]的-1

    context_tokens.append(bert_ids)
    context_loss_tokens.append(bert_loss_ids)
    assert len(bert_ids) == len(bert_loss_ids) # 檢查兩者長度是否相同

    # 判斷目前長度是否大於最大長度
    if(len(bert_ids)>max_seq_len):
        return len(bert_ids)

    Mask_id_list.clear()
    new_word_piece_list.clear()
    word_piece_list.clear()
    return max_seq_len

# 每個詞有15%的機率，去做80%的機率調換成mask，10%的機率調換成其他的詞，10%的機率不變
def random_change_word_piece(tokenizer, word_piece_list, Mask_id_list, new_word_piece_list):
    count = 0
    for word_piece in word_piece_list:
        if 0.15 >= random.random():
            change_probability = random.random()
            if 0.8 >= change_probability:
                # 紀錄被變成[MASK]前的字的id
                Mask_id_list.append(tokenizer.convert_tokens_to_ids(word_piece))
                new_word_piece_list.append('[MASK]')
            elif 0.9 >= change_probability:
                # 從vocab.txt中隨機選取一個字去替換原本的字
                vocab_index = random.randint(0, len(tokenizer.vocab)-1)
                new_word_piece_list.append(tokenizer.convert_ids_to_tokens(vocab_index))
            else: 
                new_word_piece_list.append(word_piece)
        else:
            new_word_piece_list.append(word_piece)

    # 檢查[MASK]的數量是否等於Mask_id_list的數量，沒有的話會清除Mask_id_list，接著再重做一次
    for new_word_piece in new_word_piece_list:
        if new_word_piece == '[MASK]':
            count = count + 1
    if count != len(Mask_id_list):
        Mask_id_list.clear()
        new_word_piece_list.clear()
        word_piece_list.clear()


def makeDataset(input_ids, loss_ids):
    all_input_ids = torch.tensor([input_id for input_id in input_ids], dtype=torch.long)
    all_loss_ids = torch.tensor([loss_id for loss_id in loss_ids], dtype=torch.long)
    return TensorDataset(all_input_ids, all_loss_ids)
        
if __name__ == "__main__":
    # tokenizer = BertTokenizer(vocab_file='bert-base-chinese-vocab.txt')
    # print(len(tokenizer.vocab))
    # print(tokenizer.convert_ids_to_tokens(21127))
    data_features = convert_data_to_feature('DRCD_test.json')
    Dataset = makeDataset(data_features['input_ids'], data_features['loss_ids'])
    # print(Dataset[0])

   
