import os
import json
import numpy as np
import pandas as pd

from scipy import special, stats
from scipy.spatial import distance

from tqdm import tqdm


with open('class_stats.json', 'r') as f:
    class_stats = json.load(f)

with open('images_to_embeddings.json', 'r') as f:
    images_to_embeddings = json.load(f)

with open('ids_to_idx.json', 'r') as f:
    ids_to_idx = json.load(f)

with open('classes_to_ids.json', 'r') as f:
    classes_to_ids = json.load(f)

with open('images_to_ids.json', 'r') as f:
    images_to_ids = json.load(f)


human_results = pd.read_csv('human_responses.csv')
human_image_responses = {img: {17: {}, 50: {}, 150: {}, 10000:{}} for img in set(human_results['image'].tolist())}
for _, row in human_results.iterrows():
    img = row['image']
    dur = row['image_duration']
    response = row['response']
    

    idx = ids_to_idx[classes_to_ids[response]]
    if idx not in human_image_responses[img][dur]:
        human_image_responses[img][dur][idx] = 0
    human_image_responses[img][dur][idx] += 1


classes = sorted(list(class_stats.keys()))
means = np.array([np.array(class_stats[c]['mean']) for c in classes])


def distance_metric(mean, emb):
    #return sum([distance.euclidean(mean[c], emb[c]) for c in range(len(mean))])
    return distance.euclidean(mean, emb)
    #return distance.cosine(mean, emb)


distances = {}
for img, emb in tqdm(images_to_embeddings.items()):
    dist = sorted(list(range(len(classes))), key=lambda x: distance_metric(means[x],emb))
    distances[img] = dist

with open('image_distance_to_means.json', 'w') as f:
    json.dump(distances, f)

    
    
