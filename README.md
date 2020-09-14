# DRCD-ExerciseMaskedLM-BertForMaskedLM
使用[台達電的資料集](https://github.com/DRCKnowledgeTeam/DRCD)練習MaskLM的做法並fine-turing BertForMaskedLM

## 注意:本人所練習的MaskLM的做法，與Bert中的作法有些許差異。
### 底下是Bert對MaskLM任務的描述:
```
chooses 15% of the token positions at random for prediction
If the token is chosen, replace the token with:
(1) the [MASK] token 80% of the time
(2) a random token 10% of the time
(3) the unchanged token 10% of the time
```
### 然而本人實際會做預測的只有該[MASK]符號，其餘兩種情況並不考慮。

## 檔案說明
### Data
- preprocess_data.py : maskLM的前處理
- train.py : 模型訓練(BertForMaskedLM fine-tune)
- predict.py : maskLM的預測
- requestment.txt : 紀錄需要安裝的環境
## 環境需求
- python 3.6+
- pytorch 1.3+
- transformers 2.2+
- CUDA Version: 10.0
