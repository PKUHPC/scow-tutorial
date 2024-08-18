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

稠密检索
```bash
python dense.py
```


稀疏检索
```bash
python sparse.py
```


多向量检索
```bash
python multi-vector.py
```

计算句子对的分数
```bash
python text_pairs.py
```