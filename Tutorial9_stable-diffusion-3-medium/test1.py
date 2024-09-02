import torch
from diffusers import StableDiffusion3Pipeline

# 加载模型
pipe = StableDiffusion3Pipeline.from_pretrained("../models/models--stabilityai--stable-diffusion-3-medium-diffusers/snapshots/ea42f8cef0f178587cf766dc8129abd379c90671", torch_dtype=torch.float16)

# 使用 GPU
pipe = pipe.to("cuda")

# promt 内容，可以使用多个 prompt
# prompt2 = "Photorealistic"
prompt = "Albert Einstein leans forward, holds a Qing dynasty fan. A butterfly lands on the blooming peonies in the garden. The fan is positioned above the butterfly. "

# 根据 prompt 生成多张图片
for i in range(10):
    image = pipe(
            prompt=prompt,
            # prompt_2=prompt2,
            negative_prompt="ugly, deformed, disfigured, poor details, bad anatomy",
            num_inference_steps=70,
            guidance_scale=7,
            height=1024,
            width=1024,
        ).images[0]

    image.save(f"{i}.png")