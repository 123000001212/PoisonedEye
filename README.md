## PoisonedEye: Knowledge Poisoning Attack on Retrieval-Augmented Generation based Large Vision-Language Models

## What is this repo?
This repo includes an implementation of the knowledge poisoning attack on Vision-Language Retrieval-Augmented Generation (VLRAG) systems. See our paper "PoisonedEye: Knowledge Poisoning Attack on Retrieval-Augmented Generation based Large Vision-Language Models" for details.

## Description of Files and Folders
- ``data/`` : Contains the definition and structure of the M-BEIR dataset.
- ``constant.py`` : Holds constant values.
- ``database.py`` : Defines the structure and functionality of the retrieval database.
- ``llava_inference_rag_poison_final_class.py`` : Performs class-wise evaluation of the attack.
- ``llava_inference_rag_poison_final.py`` : Performs sample-wise evaluation of the attack.
- ``mbeir_dataset_imageonly_webqa.py`` : Constructs a FAISS index for WebQA images.
- ``mbeir_dataset.py`` : Constructs a FAISS index for the OVEN-Wiki dataset.
- ``utils.py`` : Provides various utility functions.

## Preparation before running the code
You'll need several steps to make sure the code run properly.
### Hardware requirements
We recommend a GPU with at least 48 GB GPU-memory.

Make sure you have at least 500 GB free disk space to download datasets and store databases.

### Environmental setup
Make sure you have a pytorch environment with transformers and all packages in `requirements.txt`.

Or just create a new environment and run `pip install -r requirements.txt`.

The recommended python version is 3.10.16
### Download datasets
Download M-BEIR 

```
# download from huggingface
huggingface-cli download --repo-type dataset --resume-download TIGER-Lab/M-BEIR --local-dir M-BEIR

# Navigate to the M-BEIR directory
cd ./M-BEIR

# Combine the split tar.gz files into one
sh -c 'cat mbeir_images.tar.gz.part-00 mbeir_images.tar.gz.part-01 mbeir_images.tar.gz.part-02 mbeir_images.tar.gz.part-03 > mbeir_images.tar.gz'

# Extract the images from the tar.gz file
tar -xzf mbeir_images.tar.gz
```
Add full OVEN-Wiki cand_pool to M-BEIR
```
# Navigate to the M-BEIR cand_pool directory "./M-BEIR/cand_pool/local"
cd ./cand_pool/local

# Download json.zip from "https://drive.google.com/file/d/1wQBGk4Ha_rvYEA0X-8ECX-lwce4wHCBa/view?usp=sharing"
gdown https://drive.google.com/uc?id=1wQBGk4Ha_rvYEA0X-8ECX-lwce4wHCBa

# Extract the file
unzip mbeir_oven_task8_2m_cand_pool.zip
```
(Optional) Download ImageNet-1k, Places-365 (only for class-wise evalutaion)
```
# go back to root dir of the repo
cd ../../..

# Imagenet-1k
huggingface-cli download --repo-type dataset --resume-download ILSVRC/imagenet-1k --include "data/val_images.tar.gz" --local-dir imagenet-1k --token hf_***

cd ./imagenet-1k/data

tar -xzf val_images.tar.gz

# Places-365
huggingface-cli download --repo-type dataset --resume-download haideraltahan/wds_places365 --local-dir places365
```
### Download models
```
huggingface-cli download --resume-download google/siglip-so400m-patch14-384 --local-dir siglip-so400m-patch14-384
huggingface-cli download --resume-download laion/CLIP-ViT-H-14-laion2B-s32B-b79K --local-dir CLIP-ViT-H-14-laion2B-s32B-b79K
huggingface-cli download --resume-download liuhaotian/llava-v1.6-mistral-7b --local-dir llava-v1.6-mistral-7b
huggingface-cli download --resume-download Qwen/Qwen2-VL-7B-Instruct --local-dir Qwen2-VL-7B-Instruct
```

## Create databases
Build retrieval database with faiss index.
```
python mbeir_dataset.py --model_path="siglip-so400m-patch14-384" --dim=1152 --beir_cand_pool_path="cand_pool/local/mbeir_oven_task8_2m_cand_pool.jsonl" --save_path="siglip_mbeir_oven_task8_2m_cand_pool.bin"

python mbeir_dataset_imageonly_webqa.py --model_path="siglip-so400m-patch14-384" --dim=1152 --save_path="siglip_mbeir_webqa_task2_cand_pool.bin"
```

## Start Poisoning
Sample-wise evaluation. poison_type $\in$ {text-only, poison-sample}
```
python llava_inference_rag_poison_final.py \
    --poison_type=poison-sample \
    --retrieval_encoder_path="siglip-so400m-patch14-384" \
    --retrieval_database_path="siglip_mbeir_oven_task8_2m_cand_pool.bin" \
    --mbeir_subset_name=infoseek \
    --eval_number=1000 \
    --disable_tqdm=False
```

Class-wise evaluation. poison_type $\in$ {text-only, poison-sample, poison-class}
```
python llava_inference_rag_poison_final_class.py \
    --poison_type=poison-class \
    --retrieval_encoder_path="siglip-so400m-patch14-384" \
    --retrieval_database_path="siglip_mbeir_oven_task8_2m_cand_pool.bin" \
    --img_database_path="siglip_mbeir_webqa_task2_cand_pool.bin" \
    --eval_dataset=places-365 \
    --eval_dataset_path=places365 \
    --disable_tqdm=False
```
## Citation 
To be updated...