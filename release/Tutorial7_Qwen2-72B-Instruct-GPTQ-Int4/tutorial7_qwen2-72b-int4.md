# qwen2-72b-int4 推理

> 作者: 黎颖; 龙汀汀
>
> 联系方式: yingliclaire@pku.edu.cn;   l.tingting@pku.edu.cn


Qwen 系列模型是由阿里巴巴开发的。Qwen 模型系列包括不同规模的模型，参数范围从 0.5 到 720 亿，适用于各种应用场景，如文本生成、翻译、问答等。

Qwen2-72B-Instruct-GPTQ-Int4 支持高达 131,072 个 token 的上下文长度，能够处理大量输入。本 tutorial 展示如何在 scow 平台上使用单机单卡跑通 Qwen2-72B-Instruct-GPTQ-Int4 做推理。


使用的环境如下：

创建conda环境

```bash
conda create -n tutorial python=3.9
conda activate tutorial
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
pip install --upgrade pyarrow
```

注：使用的 torch 版本需要与 cuda 版本匹配，请查询 cuda。

其中， requirements.txt 的内容是：
```
numpy==1.26.4
matplotlib==3.8.4
ipykernel==6.29.5
optimum>1.13.2
auto-gptq>0.4.2
transformers>=4.32.0,<4.38.0
accelerate
tiktoken
einops
transformers_stream_generator==0.0.4
scipy
```

## 1. 下载模型

在联网的机器上下载模型：

```bash
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download --resume-download Qwen/Qwen2-72B-Instruct-GPTQ-Int4
mv ~/.cache/huggingface/hub/models--Qwen--Qwen2-72B-Instruct-GPTQ-Int4 models/ # 移动到自定义目录下
```

## 2. 模型推理

[[参考链接]](https://huggingface.co/Qwen/Qwen2-72B-Instruct-GPTQ-Int4)

运行以下 python 脚本进行简单对话:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# 使用 GPU
device = "cuda"

# 模型路径
model_path = "models/models--Qwen--Qwen2-72B-Instruct-GPTQ-Int4/snapshots/6b82a333287651211b1cae443ff2d2a6802597b9"

# 加载模型和分词器
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# prompt
prompt = "什么是大语言模型"
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt}
]

# 生成回答
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(device)

generated_ids = model.generate(
    model_inputs.input_ids,
    max_new_tokens=512
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

print(response)
```

推理过程中使用 nvidia-smi 命令可以查看 GPU 运行情况。
