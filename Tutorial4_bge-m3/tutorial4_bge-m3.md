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

## 1. 下载模型

在联网的机器上下载模型：

```bash
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download --resume-download BAAI/bge-m3
mv ~/.cache/huggingface/hub/models--BAAI--bge-m3 models/
```

## 2. 模型使用

[[参考链接]](https://huggingface.co/BAAI/bge-m3)

```bash
pip install -U FlagEmbedding
```

稠密检索使用低维、密集的向量表示文本数据，将文本嵌入到连续的向量空间中，能够捕捉语义相似性，适合处理自然语言处理（NLP）任务中的模糊查询和复杂语义关系。运行本目录下的 dense.py 脚本。
```bash
python dense.py
```


稀疏检索使用高维、稀疏的向量表示文本，其中大部分特征值为零，其计算效率高，易于解释，适合处理短文本和关键词匹配。运行本目录下的 sparse.py 脚本。
```bash
python sparse.py
```


多向量检索是一种混合方法，结合了稠密和稀疏检索的优点，使用多个向量来表示一个文档或查询。运行本目录下的 multi-vector.py 脚本。
```bash
python multi-vector.py
```

计算三种检索的加权平均值：
```bash
python text_pairs.py
```