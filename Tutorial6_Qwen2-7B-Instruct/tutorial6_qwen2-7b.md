# qwen2-7b 推理和微调 

> 作者: 黎颖; 龙汀汀
>
> 联系方式: yingliclaire@pku.edu.cn;   l.tingting@pku.edu.cn


Qwen 系列模型是由阿里巴巴开发的。Qwen 模型系列包括不同规模的模型，参数范围从 0.5 到 720 亿，适用于各种应用场景，如文本生成、翻译、问答等。

Qwen2-7B-Instruct 支持高达 131,072 个 token 的上下文长度，能够处理大量输入。本 tutorial 旨在使用 Qwen2-7B-Instruct 模型展示模型对话、微调训练的过程。

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
huggingface-cli download --resume-download Qwen/Qwen2-7B-Instruct
mv ~/.cache/huggingface/hub/models--Qwen--Qwen2-7B-Instruct models/ # 把模型移动到自定义的目录下
```

## 2. 模型推理

<!-- TODO 模型路径需要修改，除此以外，其余部分的模型路径均需要检查 -->

[[参考链接]](https://huggingface.co/Qwen/Qwen2-7B-Instruct)

运行 python 脚本进行简单对话

```bash
python qwen2_7b_test.py
```

## 3. 模型微调

使用 LLaMA Factory 进行微调。首先需要安装 LLaMA Factory，在外部目录下安装，我们安装的目录是：/tutorial-xscow/LLaMA-Factory

关于 LLaMA Factory 项目的详细说明参考： [[参考链接]](https://github.com/hiyouga/LLaMA-Factory)

```bash
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics,deepspeed,qwen]"
```

使用 LLaMA Factory 微调：
在当前目录下的data目录下准备好微调数据集，运行下面的命令进行微调

```bash
llamafactory-cli train qwen_finetune.yaml
```

合并 Lora 参数：

```bash
llamafactory-cli export qwen_merge.yaml
```

合并后的模型在 outputs/Qwen-7B-Finetuned 目录下。使用下面命令可以进行加载：

```bash
llamafactory-cli chat qwen-chat.yaml
```

运行过程中使用 nvidia-smi 命令可以查看 GPU 运行情况。


