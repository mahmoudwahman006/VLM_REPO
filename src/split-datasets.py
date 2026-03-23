import json
import os
from os.path import join
import random
from tqdm.auto import tqdm
import requests

data_dir = "/workspace/"
sft_data_path = join(data_dir,"ocr-images-sft.jsonl")

os.makedirs(join(data_dir, "datasets", "llamafactory-ocr-finetune-data"), exist_ok=True)

llm_finetunning_data = []

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

image_paths_set = set()
val_pdf_files = ['0012.pdf', '0005.pdf', '0011.pdf']

train_ds = []
val_ds = []

for line in open(sft_data_path):
    if line.strip() == "":
        continue

    rec = json.loads(line.strip())

    if rec['image_path'] in image_paths_set:
        continue

    image_paths_set.add(rec['image_path'])

    # fix gdrive static path
    rec['image_path'] = rec['image_path'].replace("/gdrive/MyDrive/youtube-resources/temp/image-ocr-finetune/assets/", data_dir)

    try:
        ft_output = json.loads(rec['output'])
    except:
        continue
    
    task_1_rec_data = {
        "conversations": [
                {
                    "value": "<image>"+task_1_message,
                    "from": "human"
                },
                {
                    "value": json.dumps({
                        'output': ft_output['content'],
                        'structural_elements': ft_output['structural_elements'],
                    }, ensure_ascii=False, default=str),
                    "from": "gpt"
                }
            ],
        "images": [
            rec['image_path']
        ]
    }

    del ft_output['content']
    del ft_output['structural_elements']

    task_2_rec_data = {
        "conversations": [
                {
                    "value": "<image>"+task_2_message,
                    "from": "human"
                },
                {
                    "value": json.dumps(ft_output, ensure_ascii=False, default=str),
                    "from": "gpt"
                }
            ],
        "images": [
            rec['image_path']
        ]
    }
    
    if rec['pdf_name'] in val_pdf_files:
        val_ds.append(task_1_rec_data)
        val_ds.append(task_2_rec_data)
    else:
        train_ds.append(task_1_rec_data)
        train_ds.append(task_2_rec_data)


random.Random(101).shuffle(train_ds)
random.Random(101).shuffle(val_ds)



with open(join(data_dir, "datasets", "llamafactory-ocr-finetune-data", "train.json"), "w") as dest:
    json.dump(train_ds, dest, ensure_ascii=False, default=str)

with open(join(data_dir, "datasets", "llamafactory-ocr-finetune-data", "val.json"), "w", encoding="utf8") as dest:
    json.dump(val_ds, dest, ensure_ascii=False, default=str)
