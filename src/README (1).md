---
license: gemma
library_name: transformers
base_model: google/gemma-3-4b-it
tags:
- gemma3
- vision
- image-to-text
- multimodal
- ocr
- arabic
- vllm
- lora
- llama-factory
datasets:
- custom
language:
- ar
- en
pipeline_tag: image-text-to-text
extra_gated_heading: Access Gemma 3
extra_gated_prompt: >-
  By clicking "Acknowledge license" below, you agree to the Gemma Terms of Use.
---


# Arabic Legal Documents OCR 1.0 (VLM Finetuned)

**[Watch the Full 3.5-Hour Masterclass on YouTube](https://youtu.be/OGk1N3YwEHI)**

This model is a finetuned version of **Gemma-3-4B-IT**, optimized for extracting structured data from low-quality, scanned Arabic legal documents using Vision Language Model reasoning.

## 🛠 Installation

Depending on your usage (Local Inference vs. Production Serving), install the required packages:

### For Transformers (Local Inference)

```bash
pip install transformers==4.57.6 optimum==1.26.0 accelerate==1.8.0 peft==0.17.0 json-repair PIL

```

### For vLLM (High-Performance Serving)

```bash
!pip install -q transformers==4.57.6
!pip install -q optimum==1.26.0
!pip install -q datasets==4.4.0

!pip install -q torch==2.8.0
!pip install -q torchvision==0.23
!pip install -q torchaudio==2.8.0

!pip install -q vllm==0.15.0
!pip install json-repair

```

---

## 🖼 Mandatory Image Preprocessing

To achieve the best OCR results, images **must** be preprocessed (resized and converted to grayscale) before being sent to the model. Below are the utility functions for both standard PIL usage and Base64 (vLLM/OpenAI API).

```python
import base64
from io import BytesIO
from PIL import Image, ImageEnhance

def preprocess_image(image_path, max_width=1024, do_enhance=True, return_base64=False):
    image = Image.open(image_path)
    
    # 1. Convert to grayscale
    gray_image = image.convert('L')
    
    # 2. Resize maintaining aspect ratio
    if gray_image.width > max_width:
        ratio = max_width / float(gray_image.width)
        new_height = int(gray_image.height * ratio)
        gray_image = gray_image.resize((max_width, new_height), Image.LANCZOS)

    # 3. Enhance contrast
    if do_enhance:
        enhancer = ImageEnhance.Contrast(gray_image)
        gray_image = enhancer.enhance(1.5)

    if return_base64:
        buffered = BytesIO()
        gray_image.save(buffered, format="JPEG", optimize=True, quality=95)
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return f"data:image/jpeg;base64,{img_str}"
    
    return gray_image

```

---

## 🚀 Usage Examples

### 1. Using Transformers & json-repair

```python
import json_repair
from transformers import AutoProcessor, Gemma3ForConditionalGeneration

model_id = "bakrianoo/arabic-legal-documents-ocr-1.0"
model = Gemma3ForConditionalGeneration.from_pretrained(model_id, device_map="auto", torch_dtype=torch.bfloat16)
processor = AutoProcessor.from_pretrained(model_id)

# Preprocess image first
processed_img = preprocess_image("document.jpg", return_base64=False)

messages = [
    {"role": "user", "content": [{"type": "image", "image": processed_img}, {"type": "text", "text": "Extract details to JSON."}]}
]

inputs = processor.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
output = model.generate(**inputs, max_new_tokens=2048)
raw_text = processor.decode(output[0], skip_special_tokens=True)

# Fix and parse JSON output
json_data = json_repair.loads(raw_text)
print(json_data)

```

### 2. Using vLLM API

Run vLLM server

```bash
vllm serve "bakrianoo/arabic-legal-documents-ocr-1.0" \
--dtype bfloat16 --gpu_memory_utilization 0.8 \
--enable-chunked-prefill \
--allowed-local-media-path "/workspace/"
```

Inference

```python
from openai import OpenAI
import json_repair

client = OpenAI(api_key="any", base_url="http://localhost:8000/v1")

# Preprocess to Base64
b64_image = preprocess_image("document.jpg", return_base64=True)

response = client.chat.completions.create(
    model="bakrianoo/arabic-legal-documents-ocr-1.0",
    messages=[{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": b64_image}},
        {"type": "text", "text": "Extract details to JSON."}
    ]}]
)

# Robust parsing
structured_output = json_repair.loads(response.choices[0].message.content)

```

---

## 📺 Full Tutorial

Watch the detailed walkthrough on YouTube to understand the training pipeline:
[VLM Finetuning for OCR Tasks](https://youtu.be/OGk1N3YwEHI)

## Resource

LoRA Adapter: https://huggingface.co/bakrianoo/arabic-legal-documents-ocr-1.0/tree/main/checkpoints

Data: https://huggingface.co/bakrianoo/arabic-legal-documents-ocr-1.0/tree/main/data

Scripts: https://huggingface.co/bakrianoo/arabic-legal-documents-ocr-1.0/tree/main/scripts

