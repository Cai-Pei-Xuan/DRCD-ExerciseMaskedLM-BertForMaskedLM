# -*-coding:UTF-8 -*-
import torch
from transformers import BertConfig, BertForMaskedLM, BertTokenizer
import torch.nn.functional as F     # 激励函数都在这

def toBertIds(question_input):
    return tokenizer.build_inputs_with_special_tokens(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(question_input)))

if __name__ == "__main__":

    # load and init
    tokenizer = BertTokenizer(vocab_file='bert-base-chinese-vocab.txt')
    
    bert_config, bert_class = (BertConfig, BertForMaskedLM)
    config = bert_config.from_pretrained('trained_model/0/config.json')
    model = bert_class.from_pretrained('trained_model/0/pytorch_model.bin', from_tf=bool('.ckpt' in 'bert-base-chinese'), config=config)
    model.eval()


    question_inputs = ["[CLS] 你 [MASK] 嗎 [SEP]"]
    for question_input in question_inputs:
        tokenized_text = tokenizer.tokenize(question_input)
        maskpos = tokenized_text.index('[MASK]')
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
        tokens_tensor = torch.tensor([indexed_tokens])

        outputs = model(tokens_tensor)
        predictions = outputs[0]

        logit_prob = F.softmax(predictions[0, maskpos]).data.tolist()
        predicted_index = torch.argmax(predictions[0, maskpos]).item()
        predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
        print(predicted_token,logit_prob[predicted_index])