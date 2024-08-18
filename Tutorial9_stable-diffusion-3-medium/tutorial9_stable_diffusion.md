# Stable_diffusion

> 作者: 黎颖; 龙汀汀
>
> 联系方式: yingliclaire@pku.edu.cn;   l.tingting@pku.edu.cn


Stable Diffusion 是由 Stability AI 开发的一个开源的深度学习模型，用于生成高质量图像。



使用的环境如下：

创建conda环境

```bash
conda create -n tutorial python=3.9
conda activate tutorial
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
pip install -U diffusers
```

注：使用的 torch 版本需要与 cuda 版本匹配，请查询 cuda 版本。

## 1. 下载模型

在联网的机器上下载模型：

```bash
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download --token hf_*** --resume-download stabilityai/stable-diffusion-3-medium
mv ~/.cache/huggingface/hub/models--stabilityai--stable-diffusion-3-medium-diffusers models/
```

其中 "hf_***" 是 huggingface 官网为每个用户提供的 token 序列。

## 2. 文生图

[[参考链接]](https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers/tree/main)

运行下面命令进行从文字生成图像：

```bash
python test1.py
```

生成的图像在本地目录下，可点击或下载查看。

