# Bge Embedding

> 作者: 黎颖; 龙汀汀
>
> 联系方式: yingliclaire@pku.edu.cn;   l.tingting@pku.edu.cn

Embedding 分别计算两个句子的向量表示，使用两个向量的余弦相似度衡量句子的相似度。

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
huggingface-cli download --resume-download BAAI/bge-m3
mv ~/.cache/huggingface/hub/models--BAAI--bge-m3 models/ # 把模型移动到自定义的路径下
```

## 2. 模型使用

[[参考链接]](https://huggingface.co/BAAI/bge-m3)

首先，安装 FlagEmbedding 包：

```bash
pip install -U FlagEmbedding
```

稠密检索使用低维、密集的向量表示文本数据，将文本嵌入到连续的向量空间中，能够捕捉语义相似性，适合处理自然语言处理（NLP）任务中的模糊查询和复杂语义关系。

```python
from FlagEmbedding import BGEM3FlagModel

# 填写模型路径
model = BGEM3FlagModel('models/models--BAAI--bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181',  
                       use_fp16=True)

# 待计算的句子
sentences_1 = ["What is BGE M3?", "Defination of BM25"]
sentences_2 = ["BGE M3 is an embedding model supporting dense retrieval, lexical matching and multi-vector interaction.", 
               "BM25 is a bag-of-words retrieval function that ranks a set of documents based on the query terms appearing in each document"]

# 计算 Embedding
embeddings_1 = model.encode(sentences_1, 
                            batch_size=12, 
                            max_length=8192, 
                            )['dense_vecs']
embeddings_2 = model.encode(sentences_2)['dense_vecs']

# 计算相似度
similarity = embeddings_1 @ embeddings_2.T
print(similarity)
# 结果应该是：
# [[0.6265, 0.3477], [0.3499, 0.678 ]]
```



稀疏检索使用高维、稀疏的向量表示文本，其中大部分特征值为零，其计算效率高，易于解释，适合处理短文本和关键词匹配。

稀疏检索
```python
from FlagEmbedding import BGEM3FlagModel

# 填写模型路径
model = BGEM3FlagModel('models/models--BAAI--bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181',  use_fp16=True) 

# 待计算的句子
sentences_1 = ["What is BGE M3?", "Defination of BM25"]
sentences_2 = ["BGE M3 is an embedding model supporting dense retrieval, lexical matching and multi-vector interaction.", 
               "BM25 is a bag-of-words retrieval function that ranks a set of documents based on the query terms appearing in each document"]

# 通过 lexical mathcing 计算相似度
output_1 = model.encode(sentences_1, return_dense=True, return_sparse=True, return_colbert_vecs=False)
output_2 = model.encode(sentences_2, return_dense=True, return_sparse=True, return_colbert_vecs=False)

lexical_scores = model.compute_lexical_matching_score(output_1['lexical_weights'][0], output_2['lexical_weights'][0])
print(lexical_scores)
# 0.19554901123046875
print(model.compute_lexical_matching_score(output_1['lexical_weights'][0], output_1['lexical_weights'][1]))
# 0.0

# 查看每个 token 的 weight：
print(model.convert_id_to_token(output_1['lexical_weights']))
# [{'What': 0.08356, 'is': 0.0814, 'B': 0.1296, 'GE': 0.252, 'M': 0.1702, '3': 0.2695, '?': 0.04092}, 
#  {'De': 0.05005, 'fin': 0.1368, 'ation': 0.04498, 'of': 0.0633, 'BM': 0.2515, '25': 0.3335}]
```

多向量检索是一种混合方法，结合了稠密和稀疏检索的优点，使用多个向量来表示一个文档或查询。

```python
from FlagEmbedding import BGEM3FlagModel

# 填写模型路径
model = BGEM3FlagModel('models/models--BAAI--bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181',  use_fp16=True) 

# 待计算的句子
sentences_1 = ["What is BGE M3?", "Defination of BM25"]
sentences_2 = ["BGE M3 is an embedding model supporting dense retrieval, lexical matching and multi-vector interaction.", 
               "BM25 is a bag-of-words retrieval function that ranks a set of documents based on the query terms appearing in each document"]

# 通过 colbert 计算相似度
output_1 = model.encode(sentences_1, return_dense=True, return_sparse=True, return_colbert_vecs=True)
output_2 = model.encode(sentences_2, return_dense=True, return_sparse=True, return_colbert_vecs=True)

print(model.colbert_score(output_1['colbert_vecs'][0], output_2['colbert_vecs'][0]))
print(model.colbert_score(output_1['colbert_vecs'][0], output_2['colbert_vecs'][1]))
# 0.7797
# 0.4620
```

计算三种检索的加权平均值：

```python
from FlagEmbedding import BGEM3FlagModel

# 填写模型路径
model = BGEM3FlagModel('models/models--BAAI--bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181',  use_fp16=True) 

# 待计算的句子
sentences_1 = ["What is BGE M3?", "Defination of BM25"]
sentences_2 = ["BGE M3 is an embedding model supporting dense retrieval, lexical matching and multi-vector interaction.", 
               "BM25 is a bag-of-words retrieval function that ranks a set of documents based on the query terms appearing in each document"]

sentence_pairs = [[i,j] for i in sentences_1 for j in sentences_2]

# 计算混合相似度
# w[0]*dense_score + w[1]*sparse_score + w[2]*colbert_score
print(model.compute_score(sentence_pairs, 
                          max_passage_length=128, 
                          weights_for_different_modes=[0.4, 0.2, 0.4])) 
```