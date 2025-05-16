# Database class for retrieval 

from transformers import AutoModel, AutoProcessor
from torchvision import transforms
import faiss
import torch
from PIL import Image
import numpy as np
from torchvision.utils import save_image
from utils import tensor_to_pil, get_weighted_sum, get_cluster_center, get_cluster_center_AP, \
                    pil_to_tensor, resize_image_tensor, normalize_image, denormalize_image, center_crop_tensor

class Database():
    def __init__(self, database_path, encode_model_path, device):
        self.database = faiss.read_index(database_path) # load index from disk
        self.model  = AutoModel.from_pretrained(encode_model_path).to(device) # load model
        self.processor = AutoProcessor.from_pretrained(encode_model_path) # load processer
        self.device = device
    
    def get_image_vector(self, image, encoder=None, processor=None):
        if encoder is None and processor is None:
            encoder = self.model
            processor = self.processor
        image_input = processor(images=image, return_tensors="pt", padding=True).to(self.device)
        image_outputs = encoder.get_image_features(**image_input)
        image_outputs = image_outputs / image_outputs.norm(p=2, dim=-1, keepdim=True)
        return image_outputs
    
    def get_image_vector_batch(self, image, encoder=None, processor=None):
        if encoder is None and processor is None:
            encoder = self.model
            processor = self.processor
        res=[]
        with torch.no_grad():
            for i in range(len(image)):
                img = [image[i]]
                image_input = processor(images=img, return_tensors="pt", padding=True).to(self.device)
                image_outputs = encoder.get_image_features(**image_input)
                image_outputs = image_outputs / image_outputs.norm(p=2, dim=-1, keepdim=True)
                res.append(image_outputs.squeeze(0))
        return torch.stack(res)
    
    def get_text_vector(self, text, encoder=None, processor=None):
        if not isinstance(text, list):
            text = [text]
        if encoder is None and processor is None:
            encoder = self.model
            processor = self.processor
        query_text = processor(text=text, return_tensors="pt", padding=True, truncation=True).to(self.device)
        text_outputs = encoder.get_text_features(**query_text)
        text_outputs = text_outputs / text_outputs.norm(p=2, dim=-1, keepdim=True)
        return text_outputs

    def get_vector(self, text, image):
        text_outputs = self.get_text_vector(text)
        image_outputs = self.get_image_vector(image)

        vectors = image_outputs + text_outputs
        vectors = vectors / vectors.norm(p=2, dim=-1, keepdim=True)
        vectors = vectors.detach().cpu().numpy() # convert to numpy
        return vectors
    
    def query(self, text: str, image, k=5, search_range=None):
        if k==0:
            return [[]],[[]]
        if text is None:
            vectors = self.get_image_vector(image).detach().cpu().numpy()
        elif image is None:
            vectors = self.get_text_vector(text).detach().cpu().numpy()
        else:
            vectors = self.get_vector(text, image)
        if search_range:
            lims, distances, indices = self.database.range_search(vectors, search_range) # neighbors within search_range
        else:
            distances, indices = self.database.search(vectors, k) # k closest neighbors
        return distances, indices

    def add(self, text, image):
        vectors = self.get_vector(text, image)
        self.database.add(vectors)

    def remove_last(self):
        length = self.database.ntotal
        index_to_remove = np.array([length-1], dtype=np.int64)
        return self.database.remove_ids(index_to_remove)

    def get_pixel_value(self, image):
        image_pixel_value = self.processor(images=image, return_tensors="pt", padding=True).to(self.device)['pixel_values']
        return image_pixel_value
    
    def encode_image(self, image, processor=None):
        if processor is None:
            processor = self.processor
        return processor(images=image, return_tensors="pt", padding=True).to(self.device)
    
    
    def create_poison(self, poison_image, poison_text, target_image, target_text, steps = 1000):
        image_ref = self.encode_image(poison_image)
        image_input = self.encode_image(poison_image)
        image_pert = torch.zeros_like(image_input['pixel_values']).requires_grad_()

        target_text_vector = self.get_text_vector(target_text).mean(dim=0)
        poison_text_vector = self.get_text_vector(poison_text)
        target_image_vector = self.get_image_vector_batch(target_image).mean(dim=0)
        target_vector = target_text_vector + target_image_vector
        target_vector = target_vector / target_vector.norm(p=2, dim=-1, keepdim=True)

        best_step = 0
        best_sim = 0
        best_pert = None

        for step in range(steps):
            image_input['pixel_values'] = image_ref['pixel_values'] + image_pert
            image_outputs = self.model.get_image_features(**image_input)
            poison_image_vector = image_outputs / image_outputs.norm(p=2, dim=-1, keepdim=True)

            poison_vector = poison_image_vector + poison_text_vector
            poison_vector = poison_vector / poison_vector.norm(p=2, dim=-1, keepdim=True)

            sim = torch.dot(poison_vector.squeeze(0), target_vector)
            if sim > best_sim:
                best_step = step
                best_pert = image_pert
                best_sim = sim
            # if step==0 or step%100==99:
            #     print(f"Step {step} | Sim = {sim}")

            gradient = torch.autograd.grad(outputs=-sim, inputs=image_pert)[0]

            image_pert = image_pert - 0.01 * gradient.sign()
            image_pert = image_pert.clamp(max=0.0625, min=-0.0625)

        # print(f"Best step {best_step} | Sim = {best_sim}")
        best_image_pixel = (image_ref['pixel_values'] + best_pert).squeeze(0)


        img_mean = torch.tensor(self.processor.image_processor.image_mean, dtype=torch.float32)
        img_std = torch.tensor(self.processor.image_processor.image_std, dtype=torch.float32)
        pil_image = tensor_to_pil(best_image_pixel, img_mean, img_std=img_std)
        pil_image.save('poison.png')
        
        return pil_image
    
    def compute_distance(self, image1, text1, image2, text2):
        image_vector1 = self.get_image_vector(image1)
        image_vector2 = self.get_image_vector(image2)
        text_vector1 = self.get_text_vector(text1)
        text_vector2 = self.get_text_vector(text2)
        vector1 = self.get_vector(text1, image1)
        vector2 = self.get_vector(text2, image2)
        # compute l2_distance
        diff = vector1[0] - vector2[0]
        distance = np.linalg.norm(diff, ord=2)
        # compute image_l2_distance
        diff = image_vector1[0] - image_vector2[0]
        image_distance = diff.norm(p=2).item()
        # compute text_l2_distance
        diff = text_vector1[0] - text_vector2[0]
        text_distance = diff.norm(p=2).item()
        # compute cos_sim  (since vectors have been normalized, |vector1|=|vector2|=1)
        sim = np.dot(vector1[0], vector2[0])
        
        res={"l2_distance": distance,
             "image_l2_distance": image_distance,
             "text_l2_distance": text_distance,
             "cos_sim": sim,}
        return res