# !pip install json-repair peft
# !pip install accelerate

import json
import os
from os.path import join
import json_repair
import torch
from transformers import AutoProcessor, Gemma3ForConditionalGeneration

data_dir = "/workspace"
base_model_id = "google/gemma-3-4b-it"
finetuned_model_id = f"{data_dir}/ocr-models-gemma-3-4b-it/checkpoint-400/"
sample_image = f"{data_dir}/pdf_images/0011/page_010.jpg"

task_1_message = "\n".join([
    "You are a professional OCR Details Extractor.",
    "Your rule to extract: the page markdown content in addition to the structural_elements of the document.",
    "Extract the final output into a json format.",
    "Do not generate any introduction or conclusion."
])

task_2_message = "\n".join([
    "You are a professional OCR Details Extractor.",
    "Your rule to extract the: document_classification, source, physical_properties, official_marks, signatures_authorization, routing_distribution, attachments_references, condition_notes and confidence_quality of the document.",
    "Extract the final output into a json format.",
    "Do not generate any introduction or conclusion."
])

device = "cuda"
torch_dtype = None

def parse_json(text):
    try:
        return json_repair.loads(text)
    except:
        return None


# default: Load the model on the available device(s)
model = Gemma3ForConditionalGeneration.from_pretrained(
    base_model_id, dtype="auto", device_map="auto"
)

processor = AutoProcessor.from_pretrained(base_model_id)
model.load_adapter(finetuned_model_id)

messages = [
    {
        "role": "system",
        "content": [{"type": "text", "text": "You are a helpful assistant."}]
    },
    {
        "role": "user",
        "content": [
            {"type": "image", "image": sample_image},
            {"type": "text", "text": task_1_message}
        ]
    }
]

# Preparation for inference
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt",
    # enable_thinking=False,
)

inputs = inputs.to(model.device)

# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=1024)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)

parse_json(output_text[0])
