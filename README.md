
### Project : FineTune(VLMs) Complex Arabic OCR 

#### Image Preprocessing Module

This notebook implements a preprocessing pipeline for preparing PDF documents for OCR. It converts PDFs into images, resizes them while maintaining aspect ratio, optimizes image quality, and saves the processed outputs in organized directories. The workflow supports batch processing and is designed for efficient large-scale document digitization and text extraction tasks.

#### Models that used arabic language most : 

qwen3-VL : https://huggingface.co/collections/Qwen/qwen3-vl  
1) Choose Version 2B Instruct
2) small model in or complex usecase after the finetune may case Hallucination (add more data and gissing the data) 
3) choose version 8B Instruct (more downloads than 2B)
4) the structions of training is to hallucination more than iffecient  

LLaVa-NeXT : https://huggingface.co/docs/transformers/en/model_doc/llava_next 
1) incorporating higher image resolutions and more reasoning/OCR datasets // like Gemma 3
2) faster than gemma 3
3) in a small page (contain 2-3 lines) is accurate 
4) in complete pages not accurate it may be becuase of small pages that trained on 

Gemma 3 : https://huggingface.co/google/gemma-3-27b-it 
1) no Hallucination found 
2) choose gemma-3-4B-it 
3) make the authentication for the model 
4) create a token access

#### Document Extraction JSON Field Refrence : 
#### From trying many models for the arabic language is Gemini 3
#### Synthetic Data is a termination means creating a data from a powerfull model giving an accebtable respone then training this responses to a local week model trying to make it powerfull like the first one ( knowledge Distillation).
#### LoRA Adapter : termination for frozing the layers of a pretrained model and jut adding a new layers with new data (fine-tune)
#### 
---

### 📫 Contact Me:
<a href="https://www.linkedin.com/in/mahmoud-wahman-a41848217" target="_blank"><img src="https://img.shields.io/badge/-Mahmoud%20Wahman-0077B5?style=for-the-badge&logo=Linkedin&logoColor=white"/></a>
<a href="https://wa.me/+201125442586" target="_blank"><img src="https://img.shields.io/badge/-Mahmoud%20Wahman-25D366?style=for-the-badge&logo=WhatsApp&logoColor=white"/></a>
<a href="mailto:mahmud962002@gmail.com" target="_blank"><img src="https://img.shields.io/badge/-Mahmoud%20Wahman-EA2328?style=for-the-badge&logo=Gmail&logoColor=white"/></a>

