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
from datasets import load_dataset
from utils import Imagenet1k
import argparse

parser = argparse.ArgumentParser(description='Create a faiss database on imagenet1k dataset\'s candidate pool')
parser.add_argument('--model_path', type=str, default='./siglip-so400m-patch14-384', help='Model path to encode texts and images')
parser.add_argument('--data_path', type=str, default='./imagenet-1k/data', help='Path imagenet-1k dataset')
parser.add_argument('--save_path', type=str, default='siglip_imagenet1k_val_cand_pool.bin', help='Path to save faiss database')
parser.add_argument('--batch_size', type=int, default=512, help='Batch size')
parser.add_argument('--dim', type=int, default=1152, help='Dimension of the embedding')

args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

processor = AutoProcessor.from_pretrained(args.model_path)
model = AutoModel.from_pretrained(args.model_path).to(device)
model = model.eval()

cand_pool = Imagenet1k(data_path=args.data_path)

faiss_index = faiss.IndexFlatL2(args.dim)

def cand_pool_collator(batch):
    images = []
    for sample in batch:
        images.append(sample)
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