import os
import PIL
import json
import clip
import torch
import numpy as np

from tqdm import tqdm

from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader


images = '/storage/jecummin/projects/exp6_analysis/exp6_images/images'
images_to_ids = {}
for cls in os.listdir(images):
    for img in os.listdir(os.path.join(images, cls)):
        images_to_ids[img] = cls
with open('images_to_ids.json', 'w') as f:
    json.dump(images_to_ids, f)

with open('images_to_classes.json', 'r') as f:
    images_to_classes = json.load(f)

classes_to_ids = {}
for img in images_to_classes:
    classes_to_ids[images_to_classes[img]] = images_to_ids[img]

with open('classes_to_ids.json', 'w') as f:
    json.dump(classes_to_ids, f)



image_dir = '/storage/jecummin/projects/exp6_analysis/image_folders/all'

class ExperimentImages(Dataset):

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.classes = []
        self.fnames = []
        self.images_to_classes = {}
        self.classes_to_images = {}
        for cls in os.listdir(root_dir):
            cls_dir = os.path.join(root_dir, cls)
            if not os.listdir(cls_dir):
                continue
            self.classes.append(cls)
            self.classes_to_images[cls] = []
            for img in os.listdir(cls_dir):
                self.images_to_classes[img] = cls
                self.classes_to_images[cls].append(img)
                self.fnames.append(img)

        np.random.shuffle(self.fnames)
        self.classes = sorted(self.classes)

    def __len__(self):
        return len(self.fnames)
    
    def __getitem__(self, idx):

        fname = self.fnames[idx]
        cls = self.images_to_classes[fname]
        cls_idx = self.classes.index(cls)
        img = os.path.join(self.root_dir, cls, fname)
        image = PIL.Image.open(img)
        if self.transform:
            image = self.transform(image)

        return image, (cls, cls_idx), fname





device = "cuda:3" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)


dataset = ExperimentImages(image_dir, preprocess)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=32)


classes_to_embeddings = {c: [] for c in dataset.classes}
images_to_embeddings = {}
for i, (images, (cls, cls_idx), fnames) in tqdm(enumerate(dataloader)):
    images = images.to(device)
    with torch.no_grad():
        embeddings = model.encode_image(images)

    for j in range(len(fnames)):
        images_to_embeddings[fnames[j]] = embeddings[j].to('cpu').numpy()
        classes_to_embeddings[cls[j]].append(embeddings[j].cpu().numpy())


min_val = None
max_val = None

for img, emb in images_to_embeddings.items():
    if min_val is None:
        min_val = np.min(emb)
        max_val = np.max(emb)
        continue
    print(np.min(emb))
    foo
    if np.min(emb) < min_val:
        min_val = np.min(emb)
    if np.max(emb) > max_val:
        max_val = np.max(emb)

for img, emb in images_to_embeddings.items():
    emb = (emb - min_val) / (max_val - min_val)
    images_to_embeddings[img] = emb.tolist()
        
for cls, embs in classes_to_embeddings.items():
    new_embs = []
    for emb in embs:
        emb = (emb - min_val) / (max_val - min_val)
        new_embs.append(emb)
    classes_to_embeddings[cls] = new_embs
    
class_stats = {}
for cls in classes_to_embeddings:
    embeddings = np.stack(classes_to_embeddings[cls])
    means = np.mean(embeddings, axis=0).tolist()
    std = np.std(embeddings, axis=0).tolist()
    class_stats[cls] = {'mean': means, 'std': std}

with open('class_stats.json', 'w') as f:
    json.dump(class_stats, f)

with open('images_to_embeddings.json', 'w') as f:
    json.dump(images_to_embeddings, f)

    

