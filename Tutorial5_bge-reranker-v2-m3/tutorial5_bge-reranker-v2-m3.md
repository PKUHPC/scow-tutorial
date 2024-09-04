# bge-reranker-v2-m3

> 作者: 黎颖; 龙汀汀
>
> 联系方式: yingliclaire@pku.edu.cn;   l.tingting@pku.edu.cn

reranker 使用 cross similarity 来计算两个句子的相似度，把两个句子共同作为模型的输入，与 embedding 方法相比更耗费计算资源，但是对语言有更好的理解。

使用的环境如下：

创建conda环境

```bash
conda create -n tutorial5 python=3.9
conda activate tutorial5
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install numpy==1.26.4 matplotlib==3.8.4 ipykernel==6.29.5 transformers==4.42.4
```

注：使用的 torch 版本需要与 cuda 版本匹配，请查询 cuda 版本。

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
```bash
python cross_similarity.py
```

或者使用 Huggingface Transformer 也可以调用模型：

```bash
python cross_similarity2.py
```

