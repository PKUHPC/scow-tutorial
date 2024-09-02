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