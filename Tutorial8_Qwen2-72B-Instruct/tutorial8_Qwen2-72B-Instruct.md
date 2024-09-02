# Qwen2-72B-Instruct 单机多卡

> 作者: 黎颖; 龙汀汀
>
> 联系方式: yingliclaire@pku.edu.cn;   l.tingting@pku.edu.cn


Qwen 系列模型是由阿里巴巴开发的。Qwen 模型系列包括不同规模的模型，参数范围从 0.5 到 720 亿，适用于各种应用场景，如文本生成、翻译、问答等。

Qwen2-72B-Instruct 支持高达 131,072 个 token 的上下文长度，能够处理大量输入。本 tutorial 展示如何在 scow 平台上使用单机多卡跑通 Qwen2-72B-Instruct 推理。首先，登陆 scow 平台申请单机4卡A100，连接后运行下面内容。


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
huggingface-cli download --resume-download Qwen/Qwen2-72B-Instruct
mv ~/.cache/huggingface/hub/models--Qwen--Qwen2-72B-Instruct models/ # 移动到自定义目录
```

## 2. 模型推理

[[参考链接]](https://huggingface.co/Qwen/Qwen2-72B-Instruct-GPTQ-Int4)

运行 python 进行简单对话：

```bash
python qwen_test.py
```

推理过程中使用 nvidia-smi 命令可以查看 GPU 运行情况。

## 3. 连续对话

使用 LLaMA Factory 可以非常方便地实现多卡并行连续对话。首先需要安装 LLaMA Factory，在外部目录下安装，我们安装的目录是：/tutorial-xscow/LLaMA-Factory

关于 LLaMA Factory 项目的详细说明参考： [[参考链接]](https://github.com/hiyouga/LLaMA-Factory)

```bash
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics,deepspeed,qwen]"
```

安装好后，使用下面命令可以进行加载：

```bash
llamafactory-cli chat qwen-chat.yaml
```

输入 exit 结束对话。

运行过程中使用 nvidia-smi 命令可以查看 GPU 运行情况。



