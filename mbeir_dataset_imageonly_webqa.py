# Create retrieval database for webqa

import torch
import faiss
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from data.mbeir_dataset import (
    MBEIRMainDataset,
    MBEIRCandidatePoolDataset,
    MBEIRMainCollator,
    MBEIRCandidatePoolCollator,
    Mode,
)
from transformers import AutoProcessor, AutoModel
import argparse

parser = argparse.ArgumentParser(description='Create a faiss database on beir dataset\'s candidate pool')
parser.add_argument('--model_path', type=str, default='./siglip-so400m-patch14-384', help='Model path to encode texts and images')
parser.add_argument('--beir_path', type=str, default='./M-BEIR', help='Path to beir dataset')
parser.add_argument('--beir_cand_pool_path', type=str, default='cand_pool/local/mbeir_webqa_task2_cand_pool.jsonl', help='Path to beir dataset candidate pool')
parser.add_argument('--save_path', type=str, default='siglip_mbeir_webqa_task2_cand_pool.bin', help='Path to save faiss database')
parser.add_argument('--batch_size', type=int, default=512, help='Batch size')
parser.add_argument('--dim', type=int, default=1152, help='Dimension of the embedding')

args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

processor = AutoProcessor.from_pretrained(args.model_path)
model = AutoModel.from_pretrained(args.model_path).to(device)
model = model.eval()
# img_preprocess_fn = lambda img: processor(images=img, return_tensors="pt")
# tokenizer = lambda text: processor(text=text, return_tensors="pt", padding=True, truncation=True)

cand_pool = MBEIRCandidatePoolDataset(mbeir_data_dir=args.beir_path,
                                    cand_pool_data_path=args.beir_cand_pool_path,
                                    img_preprocess_fn=None,)

faiss_index = faiss.IndexFlatL2(args.dim)

def cand_pool_collator(batch):
    images = []
    for sample in batch:
        images.append(sample['img'])
    image_output = processor(images=images, return_tensors="pt")
    return image_output

dataloader = DataLoader(cand_pool, batch_size=args.batch_size, shuffle=False, collate_fn=cand_pool_collator, drop_last=False)
with torch.no_grad():
    for batch in tqdm(dataloader):
        image = batch.to(device)
        image_outputs = model.get_image_features(**image)
        
        # normalize model output embeddings
        image_outputs = image_outputs / image_outputs.norm(p=2, dim=-1, keepdim=True)

        # normalize the average vertors
        vectors = image_outputs
        vectors = vectors.detach().cpu().numpy() # convert to numpy

        faiss_index.add(vectors)


faiss.write_index(faiss_index, args.save_path)