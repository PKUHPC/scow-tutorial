# bge-reranker-v2-m3

> 作者: 黎颖; 龙汀汀
>
> 联系方式: yingliclaire@pku.edu.cn;   l.tingting@pku.edu.cn

reranker 使用 cross similarity 来计算两个句子的相似度，把两个句子共同作为模型的输入，与 embedding 方法相比更耗费计算资源，但是对语言有更好的理解。

使用的环境如下：

创建conda环境

```bash
conda create -n tutorial python=3.9
conda activate tutorial
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
```

注：使用的 torch 版本需要与 cuda 版本匹配，请查询 cuda 版本。

其中， requirements.txt 的内容是：
```
numpy==1.26.4
matplotlib==3.8.4
ipykernel==6.29.5
transformers==4.42.4
```

## 1. 下载模型

在联网的机器上下载模型：

```bash
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download --resume-download BAAI/bge-reranker-v2-m3
mv ~/.cache/huggingface/hub/models--BAAI--bge-reranker-v2-m3 models/ # 移动模型到自定义的路径下
```

## 2. 模型使用

[[参考链接]](https://huggingface.co/BAAI/bge-reranker-v2-m3)

使用 FlagEmbedding， 计算 cross similarity

安装 FlagEmbedding 包：

```bash
pip install -U FlagEmbedding
```

运行 python 脚本：
```python
from FlagEmbedding import FlagReranker

# 填写模型路径
reranker = FlagReranker('models/models--BAAI--bge-reranker-v2-m3/snapshots/953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e', use_fp16=True) 

# 计算相似度
score = reranker.compute_score(['query', 'passage'], normalize=True)
print(score) # 0.003497010252573502

scores = reranker.compute_score([['what is panda?', 'hi'], ['what is panda?', 'The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China.']], normalize=True)
print(scores) # [0.00027803096387751553, 0.9948403768236574]

```

或者使用 Huggingface Transformer 也可以调用模型：

```python
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# 填写模型路径
modle_path = 'models/models--BAAI--bge-reranker-v2-m3/snapshots/953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e'

# 加载模型和分词器
tokenizer = AutoTokenizer.from_pretrained(modle_path)
model = AutoModelForSequenceClassification.from_pretrained(modle_path)
model.eval()

# 把模型移动到显卡
print(torch.cuda.is_available())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)


# 计算相似度
pairs = [['what is panda?', 'hi'], ['what is panda?', 'The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China.']]
with torch.no_grad():
    inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
    
    # 将输入数据移动到GPU
    inputs = {key: value.to(device) for key, value in inputs.items()}
    
    scores = model(**inputs, return_dict=True).logits.view(-1, ).float()
    print(scores)
```

