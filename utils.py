# some util functions and classes

from torch.utils.data import Dataset
import torch
import torch.nn.functional as F
import math
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import io
import base64
from transformers import CLIPProcessor, CLIPModel
from sklearn.cluster import DBSCAN, KMeans, AffinityPropagation
import faiss
import random
from collections import defaultdict
from typing import Union, Tuple

def create_conversation(query_txt, query_img, cand_pool, cand_indices, cand_txt_col_name='txt', cand_img_col_name='img'):
    conversation = [{
        "role": "system",
        "content": [
            {"type": "text", "text": "Answer the question based on multiple text-image pairs as information."},
            ],
        },]
    image_list = []

    for indice in cand_indices:
        cand_data = cand_pool[indice]
        conversation.append({
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": f"{cand_data[cand_txt_col_name]}"},
                ],
            })
        conversation.append({
            "role": "assistant",
            "content": [
                {"type": "text", "text": "OK"},
                ],
            })
        image_list.append(cand_data[cand_img_col_name])

        # cand_data[cand_img_col_name].save(f'context {indice}.png')

    conversation.append({
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": f"Question: {query_txt}"},
                ],
            })
    image_list.append(query_img)

    query_img.save('query.png')

    return conversation, image_list

def create_conversation_qwen(query_txt, query_img, cand_pool, cand_indices, cand_txt_col_name='txt', cand_img_col_name='img'):
    conversation = [{
        "role": "system",
        "content": [
            {"type": "text", "text": "Answer the question based on multiple text-image pairs as the context."},
            ],
        },]
    image_list = []
    content_list = [{"type": "text", "text": f"Context: "}]
    for indice in cand_indices:
        cand_data = cand_pool[indice]
        image_list.append(cand_data[cand_img_col_name])
        # content_list.append({"type": "image", "image": f"{img_to_base64_str(cand_data[cand_img_col_name])}"})
        content_list.append({"type": "image"})
        content_list.append({"type": "text", "text": f"{cand_data[cand_txt_col_name]}\n"})
    content_list.append({"type": "text", "text": f"Question: "})
    # content_list.append({"type": "image", "image": f"{img_to_base64_str(query_img)}"})
    content_list.append({"type": "image"})
    content_list.append({"type": "text", "text": f"{query_txt}"})
    conversation.append({
            "role": "user",
            "content": content_list,
            })
    image_list.append(query_img)
    return conversation, image_list

def img_to_base64_str(img, format='png'):
    buffered = io.BytesIO()
    img.save(buffered, format=format)
    img_str = base64.b64encode(buffered.getvalue()).decode()
    data_url = f"data:image/{format.lower()};base64,{img_str}"
    return data_url

class ExtendDataset():
    # add extend datas into a dataset without modifying the original dataset
    def __init__(self, dataset, extend_data) -> None:
        self.dataset = dataset
        self.extend_data = extend_data
        self.length = len(dataset)
    
    def __getitem__(self, i):
        if i < self.length:
            return self.dataset[i]
        else:
            return self.extend_data[i-self.length]
        
    def __len__(self):
        return self.length + len(self.extend_data)
    

class ShuffleDataset():
    # shuffle a dataset
    def __init__(self, dataset) -> None:
        self.dataset = dataset
        self.indices = np.arange(len(self.dataset))
        np.random.shuffle(self.indices)

    def __getitem__(self, i):
        return self.dataset[self.indices[i]]

    def __len__(self):
        return len(self.dataset)
    

class ClassDatasetImageNet1k():
    # randomly select one sample from each class of a dataset 
    def __init__(self, dataset) -> None:
        self.dataset = dataset
        self.label_list = []
        self.img_list = []

        # Create a dictionary to store the indices of images for each label
        label_to_indices = defaultdict(list)
        self.label_to_indices = label_to_indices
        
        # Iterate through the entire dataset labels and fill the label_to_indices dictionary
        for idx, label in enumerate(dataset.data_label):
            label_to_indices[label].append(idx)

        # Randomly select one sample from each class's index list
        for label, indices in label_to_indices.items():
            selected_idx = random.choice(indices)
            img, _ = dataset[selected_idx]
            self.img_list.append(img)
            self.label_list.append(label)

    def __getitem__(self, i):
        return self.img_list[i], self.label_list[i]

    def __len__(self):
        return len(self.img_list)
    
class ClassDatasetCaltech101():
    # randomly select one sample from each class of a dataset 
    def __init__(self, dataset) -> None:
        self.dataset = dataset
        self.label_list = []
        self.img_list = []

        # Create a dictionary to store the indices of images for each label
        label_to_indices = defaultdict(list)
        self.label_to_indices = label_to_indices
        
        # Iterate through the entire dataset labels and fill the label_to_indices dictionary
        for idx, data in enumerate(dataset):
            filename = data['filename']
            label = filename.split('/')[0]
            label_to_indices[label].append(idx)

        # Randomly select one sample from each class's index list
        for label, indices in label_to_indices.items():
            selected_idx = random.choice(indices)
            data = dataset[selected_idx]
            img = data['image'].convert('RGB')
            self.img_list.append(img)
            self.label_list.append(label)

    def __getitem__(self, i):
        return self.img_list[i], self.label_list[i]

    def __len__(self):
        return len(self.img_list)
    
class ClassDatasetCountry211():
    # randomly select one sample from each class of a dataset 
    def __init__(self, dataset) -> None:
        self.dataset = dataset
        self.label_list = []
        self.img_list = []

        # Create a dictionary to store the indices of images for each label
        label_to_indices = defaultdict(list)
        self.label_to_indices = label_to_indices
        
        # Iterate through the entire dataset labels and fill the label_to_indices dictionary
        all_label_list = dataset['label']
        for idx, data in enumerate(all_label_list):
            label = str(data)
            label_to_indices[label].append(idx)

        # Randomly select one sample from each class's index list
        for label, indices in label_to_indices.items():
            selected_idx = random.choice(indices)
            data = dataset[selected_idx]
            img = data['image'].convert('RGB')
            self.img_list.append(img)
            self.label_list.append(label)

    def __getitem__(self, i):
        return self.img_list[i], self.label_list[i]

    def __len__(self):
        return len(self.img_list)

class ClassDatasetPlaces365():
    # randomly select one sample from each class of a dataset 
    def __init__(self, dataset) -> None:
        self.dataset = dataset
        self.label_list = []
        self.img_list = []

        # Create a dictionary to store the indices of images for each label
        label_to_indices = defaultdict(list)
        self.label_to_indices = label_to_indices
        
        # Iterate through the entire dataset labels and fill the label_to_indices dictionary
        all_label_list = dataset['cls']
        for idx, data in enumerate(all_label_list):
            label = str(data)
            label_to_indices[label].append(idx)

        # Randomly select one sample from each class's index list
        for label, indices in label_to_indices.items():
            selected_idx = random.choice(indices)
            data = dataset[selected_idx]
            img = data['0.webp'].convert('RGB')
            self.img_list.append(img)
            self.label_list.append(label)

    def __getitem__(self, i):
        return self.img_list[i], self.label_list[i]

    def __len__(self):
        return len(self.img_list)
    
class ClassDatasetSUN397():
    # randomly select one sample from each class of a dataset 
    def __init__(self, dataset) -> None:
        self.dataset = dataset
        self.label_list = []
        self.img_list = []

        # Create a dictionary to store the indices of images for each label
        label_to_indices = defaultdict(list)
        self.label_to_indices = label_to_indices
        
        # Iterate through the entire dataset labels and fill the label_to_indices dictionary
        all_label_list = dataset['cls']
        for idx, data in enumerate(all_label_list):
            label = str(data)
            label_to_indices[label].append(idx)

        # Randomly select one sample from each class's index list
        for label, indices in label_to_indices.items():
            selected_idx = random.choice(indices)
            data = dataset[selected_idx]
            img = data['webp'].convert('RGB')
            self.img_list.append(img)
            self.label_list.append(label)

    def __getitem__(self, i):
        return self.img_list[i], self.label_list[i]

    def __len__(self):
        return len(self.img_list)
    

class ClassDatasetRESISC45():
    # randomly select one sample from each class of a dataset 
    def __init__(self, dataset) -> None:
        self.dataset = dataset
        self.label_list = []
        self.img_list = []

        # Create a dictionary to store the indices of images for each label
        label_to_indices = defaultdict(list)
        self.label_to_indices = label_to_indices
        
        # Iterate through the entire dataset labels and fill the label_to_indices dictionary
        all_label_list = dataset['label']
        for idx, data in enumerate(all_label_list):
            label = str(data)
            label_to_indices[label].append(idx)

        # Randomly select one sample from each class's index list
        for label, indices in label_to_indices.items():
            selected_idx = random.choice(indices)
            data = dataset[selected_idx]
            img = data['image'].convert('RGB')
            self.img_list.append(img)
            self.label_list.append(label)

    def __getitem__(self, i):
        return self.img_list[i], self.label_list[i]

    def __len__(self):
        return len(self.img_list)
    
class ClassDatasetCountry211():
    # randomly select one sample from each class of a dataset 
    def __init__(self, dataset) -> None:
        self.dataset = dataset
        self.label_list = []
        self.img_list = []

        # Create a dictionary to store the indices of images for each label
        label_to_indices = defaultdict(list)
        self.label_to_indices = label_to_indices
        
        # Iterate through the entire dataset labels and fill the label_to_indices dictionary
        all_label_list = dataset['label']
        for idx, data in enumerate(all_label_list):
            label = str(data)
            label_to_indices[label].append(idx)

        # Randomly select one sample from each class's index list
        for label, indices in label_to_indices.items():
            selected_idx = random.choice(indices)
            data = dataset[selected_idx]
            img = data['image'].convert('RGB')
            self.img_list.append(img)
            self.label_list.append(label)

    def __getitem__(self, i):
        return self.img_list[i], self.label_list[i]

    def __len__(self):
        return len(self.img_list)
    
def get_target_probs(model, image_list, prompt, target_prefix, target, processor, device):
    with torch.no_grad():
        target_ids = processor.tokenizer(target,add_special_tokens=False, return_tensors='pt')['input_ids'].squeeze(0)
        target_length = target_ids.shape[-1]
        inputs = processor(images=image_list, text=prompt+target_prefix+target, return_tensors="pt").to(device)
        output = model(**inputs).logits[0,-target_length-1:-1].cpu()
        output.argmax(dim=-1)
        probs = torch.nn.functional.softmax(output, dim=-1)
        prob_list = []
        for i in range(target_length):
            prob_list.append(probs[i][target_ids[i]])
    return math.prod(prob_list)

def tensor_to_pil(img_tensor: torch.Tensor, img_mean=None, img_std=None):
    if img_mean is None and img_std is None:
        img_tensor = img_tensor.clamp(min=-1, max=1)
        img_tensor = (img_tensor.permute(1, 2, 0) + 1) / 2  # (C, H, W) -> (H, W, C), (-1,1) -> (0,1)
        img_numpy = img_tensor.detach().cpu().numpy()
        img_numpy = (img_numpy * 255).astype(np.uint8)
        pil_image = Image.fromarray(img_numpy)
    else:
        img_tensor = denormalize_image(img_tensor, img_mean, img_std)
        img_tensor = img_tensor.clamp(min=0, max=1)
        img_tensor = img_tensor.permute(1, 2, 0)
        img_numpy = img_tensor.detach().cpu().numpy()
        img_numpy = (img_numpy * 255).astype(np.uint8)
        pil_image = Image.fromarray(img_numpy)
    return pil_image


def resize_image_keep_ratio(img:Image, max_size=500):
    original_width, original_height = img.size
    scale = min(max_size / original_width, max_size / original_height, 1)
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)
    resized_img = img.resize((new_width, new_height), Image.BICUBIC)
    return resized_img

def resize_image_within_bounds(img, max_size=500, min_size=28):

    original_width, original_height = img.size
    scale_max = min(max_size / original_width, max_size / original_height, 1)
    scale_min = max(min_size / original_width, min_size / original_height, 1)

    if scale_max < scale_min:
        scale = scale_min
    else:
        scale = scale_max

    new_width = int(original_width * scale)
    new_height = int(original_height * scale)

    resized_img = img.resize((new_width, new_height), Image.BICUBIC)
    
    return resized_img

def deduplicate(dist, ind, k=5):
    res=[]
    dist_list=[]
    for i in range(len(dist)):
        if dist[i] not in dist_list:
            res.append(ind[i])
            dist_list.append(dist[i])
        k-=1
        if k==0:
            return res
    return res

class Imagenet1k():
    def __init__(self, data_path):
        self.root = data_path
        self.data_path = [f for r,d,f in os.walk(data_path)][0]
    def __len__(self):
        return len(self.data_path)
    def __getitem__(self,i):
        img = Image.open(os.path.join(self.root, self.data_path[i])).convert('RGB')
        return img

class Imagenet1k_withlabel():
    def __init__(self, data_path):
        self.root = data_path
        self.data_path = [f for r,d,f in os.walk(data_path)][0]
        self.data_label = [p.split(".")[0].split('_')[-1] for p in self.data_path]
    def __len__(self):
        return len(self.data_path)
    def __getitem__(self,i):
        img = Image.open(os.path.join(self.root, self.data_path[i])).convert('RGB')
        return img, self.data_label[i]


class CLIPClassifier():
    def __init__(self, model_path, class_list):
        self.model = CLIPModel.from_pretrained(model_path)
        self.processor = CLIPProcessor.from_pretrained(model_path)
        self.class_list = class_list

        databse_path = os.path.join(model_path, "imagenet1k_index.bin")
        self.database_path = databse_path
        if not os.path.exists(databse_path):
            self.database = faiss.IndexFlatL2(self.model.projection_dim)
            for c in class_list:
                text_input = self.processor(text=c, return_tensors='pt', padding=True)
                text_output = self.model.get_text_features(**text_input).detach().cpu().numpy()
                self.database.add(text_output)
            faiss.write_index(self.database, databse_path)

    def classify(self, img):
        self.database = faiss.read_index(self.database_path)
        image_input = self.processor(images=img, return_tensors='pt', padding=True)
        image_output = self.model.get_image_features(**image_input).detach().cpu().numpy()
        dis, ind = self.database.search(image_output, k=2)
        return self.class_list[ind[0][0]]
    

class LLaVAClassifier():
    def __init__(self, model, processor):
        self.model = model
        self.processor = processor
    def classify(self, img):
        conversation = [{
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "Analyze the main content of the image and respond with a single word or short phrase that classifies it, without adding any extra words or explanations."},
            ],
        },]
        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = self.processor(images=img, text=prompt, return_tensors="pt").to("cuda:0")
        output = self.model.generate(**inputs, max_new_tokens=20, pad_token_id=self.model.config.pad_token_id)
        result = self.processor.decode(output[0], skip_special_tokens=True).split('[/INST]')[-1].strip(' ')
        return result.lower()
    

def get_weighted_sum(image_vector: torch.Tensor):
    sim_list = []
    for vector in image_vector:
        sim_sum = -1 # Excluding vector itself, cosine_similarity(vector, vector)=1
        total = -1 # Excluding vector itself
        for vector2 in image_vector:
            sim_sum += torch.nn.functional.cosine_similarity(vector, vector2, dim=0)/2 + 0.5
            total += 1
        # print(sim_sum)
        sim_list.append(1 - sim_sum / total)
    weight_list = torch.tensor(sim_list, dtype=torch.float32, device=image_vector.device)
    weight_list = weight_list / weight_list.sum()
    # weight_sum = 0
    # for i,v in enumerate(image_vector):
    #     weight_sum += v * weight_list[i]
    weight_sum = torch.mv(image_vector.t(), weight_list)
    return weight_sum / weight_sum.norm(p=2, dim=-1, keepdim=True)

def get_cluster_center(image_vector: torch.Tensor, eps=0.7, min_samples=2):
    image_vector_np = image_vector.detach().cpu().numpy()
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(image_vector_np)
    labels = db.labels_
    # print(labels)
    # caluate cluster centers
    unique_labels = set(labels)
    cluster_centers = []

    for label in unique_labels:
        if label != -1:  # drop noise
            cluster_points = image_vector_np[labels == label]
            center = np.mean(cluster_points, axis=0)
            cluster_centers.append(center)

    if len(cluster_centers)!=0:
        cluster_centers = np.array(cluster_centers).mean(axis=0)
    else:
        cluster_centers = image_vector_np.mean(axis=0)

    cluster_centers = torch.tensor(cluster_centers, dtype=image_vector.dtype, device=image_vector.device)
    return cluster_centers / cluster_centers.norm(p=2, dim=-1, keepdim=True)

def get_cluster_center_AP(image_vector: torch.Tensor, eps=0.7, min_samples=2):
    image_vector_np = image_vector.detach().cpu().numpy()
    db = AffinityPropagation().fit(image_vector_np)
    labels = db.labels_
    # caluate cluster centers
    unique_labels = set(labels)
    cluster_centers = []

    for label in unique_labels:
        if label != -1:  # drop noise
            cluster_points = image_vector_np[labels == label]
            center = np.mean(cluster_points, axis=0)
            cluster_centers.append(center)

    if len(cluster_centers)!=0:
        cluster_centers = np.array(cluster_centers).mean(axis=0)
    else:
        cluster_centers = image_vector_np.mean(axis=0)

    cluster_centers = torch.tensor(cluster_centers, dtype=image_vector.dtype, device=image_vector.device)
    return cluster_centers / cluster_centers.norm(p=2, dim=-1, keepdim=True)



def normalize_image(img: torch.Tensor, img_mean: torch.Tensor, img_std: torch.Tensor) -> torch.Tensor:
    """
    Normalize the input image tensor using per-channel mean and standard deviation.
    
    Parameters:
        img (torch.Tensor): The image tensor to normalize. Expected to be in the format (C, H, W).
        img_mean (torch.Tensor): A 1-dimensional tensor of means for each channel.
        img_std (torch.Tensor): A 1-dimensional tensor of standard deviations for each channel.
        
    Returns:
        torch.Tensor: The normalized image tensor.
    """
    # Ensure the mean and std are tensors and on the same device as img
    img_mean = img_mean.to(img.device)
    img_std = img_std.to(img.device)

    # Normalize the image
    normalized_img = (img - img_mean[:, None, None]) / img_std[:, None, None]

    # ori_img = normalized_img * img_std[:, None, None] + img_mean[:, None, None]
    
    return normalized_img

def denormalize_image(normalized_img: torch.Tensor, img_mean: torch.Tensor, img_std: torch.Tensor) -> torch.Tensor:
    """
    Denormalize the input image tensor using per-channel mean and standard deviation.
    
    Parameters:
        normalized_img (torch.Tensor): The normalized image tensor to denormalize. Expected to be in the format (C, H, W).
        img_mean (torch.Tensor): A 1-dimensional tensor of means for each channel used during normalization.
        img_std (torch.Tensor): A 1-dimensional tensor of standard deviations for each channel used during normalization.
        
    Returns:
        torch.Tensor: The denormalized image tensor.
    """
    # Ensure the mean and std are tensors and on the same device as normalized_img
    img_mean = img_mean.to(normalized_img.device)
    img_std = img_std.to(normalized_img.device)

    # Denormalize the image
    denormalized_img = normalized_img * img_std[:, None, None] + img_mean[:, None, None]
    
    return denormalized_img

def resize_image_tensor(img_tensor: torch.Tensor, new_size=(384, 384)) -> torch.Tensor:
    """
    Resize the input image tensor to the specified size.
    
    Parameters:
        img_tensor (torch.Tensor): The image tensor to resize. Expected to be in the format (C, H, W).
        new_size (tuple): A tuple of two integers indicating the new height and width.
        
    Returns:
        torch.Tensor: The resized image tensor with shape (C, new_height, new_width).
    """
    # Ensure the input is a tensor and on the correct device
    if not isinstance(img_tensor, torch.Tensor):
        raise TypeError("Input should be a PyTorch Tensor.")
    
    # Check that the tensor has three dimensions (C, H, W)
    if img_tensor.dim() != 3:
        raise ValueError("Input tensor must have 3 dimensions (channels, height, width).")
    
    # Perform the resizing using interpolation
    resized_img_tensor = F.interpolate(
        img_tensor.unsqueeze(0),  # Add batch dimension
        size=new_size,           # Desired output size
        mode='bilinear',         # Interpolation mode for upsampling
        align_corners=False      # Prevents issues with bilinear interpolation
    ).squeeze(0)                 # Remove batch dimension
    
    return resized_img_tensor

def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    """
    Convert a PIL.Image to a PyTorch Tensor.
    
    Parameters:
        image (Image.Image): The PIL.Image to convert.
        
    Returns:
        torch.Tensor: The converted tensor with pixel values in the range [0, 1].
    """
    # Create a transform to convert PIL Image to Tensor
    transform = transforms.ToTensor()
    
    # Apply the transform to convert the image
    img_tensor = transform(image)
    
    return img_tensor


def center_crop_tensor(img_tensor: torch.Tensor, output_size: Union[int, Tuple[int, int]]) -> torch.Tensor:
    """
    Perform a center crop on the input image tensor. If the image is smaller than the crop size along any edge,
    it pads the image with 0 and then performs the center crop.
    
    Parameters:
        img_tensor (torch.Tensor): The image tensor to crop. Expected to be in the format (C, H, W).
        output_size (int or tuple): Desired output size. If an integer is provided, a square crop of size (output_size, output_size) is returned.
                                    If a tuple of two integers is provided, it should be in the form (height, width).
        
    Returns:
        torch.Tensor: The center-cropped image tensor after padding if necessary.
    """
    # Ensure the input is a tensor and has three dimensions (channels, height, width)
    if not isinstance(img_tensor, torch.Tensor):
        raise TypeError("Input should be a PyTorch Tensor.")
    if img_tensor.dim() != 3:
        raise ValueError("Input tensor must have 3 dimensions (channels, height, width).")

    c, h, w = img_tensor.size()

    # Determine the output size
    if isinstance(output_size, int):
        output_size = (output_size, output_size)
    elif len(output_size) != 2:
        raise ValueError("Output size should be an integer or a tuple of two integers.")

    crop_h, crop_w = output_size
    
    # Calculate padding if needed
    pad_top = max(0, (crop_h - h) // 2)
    pad_bottom = max(0, (crop_h - h + 1) // 2)
    pad_left = max(0, (crop_w - w) // 2)
    pad_right = max(0, (crop_w - w + 1) // 2)

    # Apply padding if the image is smaller than the crop size
    if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
        img_tensor = torch.nn.functional.pad(img_tensor, (pad_left, pad_right, pad_top, pad_bottom), "constant", 0)
        h, w = img_tensor.size(1), img_tensor.size(2)  # Update height and width after padding

    # Calculate the starting position for cropping
    start_x = (w - crop_w) // 2
    start_y = (h - crop_h) // 2

    # Ensure the crop does not exceed the image boundaries
    if start_x < 0 or start_y < 0 or start_x + crop_w > w or start_y + crop_h > h:
        raise ValueError("Crop size cannot be larger than the padded image dimensions.")

    # Perform the crop
    cropped_img_tensor = img_tensor[:, start_y:start_y+crop_h, start_x:start_x+crop_w]

    return cropped_img_tensor

