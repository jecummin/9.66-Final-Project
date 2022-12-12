import os
import json
import numpy as np
import pandas as pd

from scipy import special, stats


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

with open('image_distance_to_means.json', 'r') as f:
    image_distances = json.load(f)


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


model_results_file = 'model_samples_warmup_20_og_std.json'
with open(model_results_file, 'r') as f:
    model_results = json.load(f)

# with open('exp_images.json', 'w') as f:
#     json.dump(list(model_results.keys()), f)
    
def KL(a, b):
    alpha = 1e-5
    return np.sum(np.where(a != 0, a * np.log(a / (b + alpha)), 0))
    
uniform_dist = np.ones(50) / 50

kl_uniform = {}
kl_avg = {}
for img, res in model_results.items():
    human_dists = human_image_responses[img]
    for step, dist in res.items():
        if step not in kl_avg:
            kl_avg[step] = {dur: 0 for dur in human_dists}
            kl_uniform[step] = {dur: 0 for dur in human_dists}
        for dur, human_dist in human_dists.items():
            x = np.zeros(50)
            # x = np.ones(50) * 0.0005
            for cls, num in dist.items():
                x[int(cls)]= num
            x /= x.sum()
            y = np.zeros(50)
            # y = np.ones(50) * 0.0005
            for cls, num in human_dist.items():
                y[int(cls)]= num
            y /= y.sum()

            kl_avg[step][dur] += KL(x, y)
            kl_uniform[step][dur] += KL(uniform_dist, y)
            #kl_avg[step][dur] += stats.wasserstein_distance(x, y)


kl_avg = {step: {dur: kl_avg[step][dur] / len(model_results) for dur in kl_avg[step]} for step in kl_avg}
print(model_results_file)
print(kl_avg)

kl_uniform = {step: {dur: kl_uniform[step][dur] / len(model_results) for dur in kl_uniform[step]} for step in kl_uniform}
print('Chance performance:')
print(kl_uniform)


accs = {'top1': {}, 'top-all': {}}
distance_accs = {'top1': {}, 'top-all': {}}
for img, res in model_results.items():
    human_dists = human_image_responses[img]
    distance_dist = image_distances[img]
    for step, dist in res.items():
        if step not in accs['top1']:
            accs['top1'][step] = {dur: 0 for dur in human_dists}
            accs['top-all'][step] = {dur: 0 for dur in human_dists}
            
            distance_accs['top1'][step] = {dur: 0 for dur in human_dists}
            distance_accs['top-all'][step] = {dur: 0 for dur in human_dists}
        for dur, human_dist in human_dists.items():
            sorted_human_dist = sorted(human_dist.keys(), key=lambda x: human_dist[x], reverse=True)
            sorted_model_dist = [int(c) for c in sorted(dist.keys(), key=lambda x: dist[x], reverse=True)][:len(sorted_human_dist)]
            sorted_distance_dist = distance_dist[:len(sorted_human_dist)]
            
            accs['top1'][step][dur] += 1 if sorted_human_dist[0] == sorted_model_dist[0] else 0
            acc_all = [1 if c in sorted_human_dist else 0 for c in sorted_model_dist]
            # acc_all = [1 if c == sorted_human_dist[0] else 0 for c in sorted_model_dist]
            accs['top-all'][step][dur] += 1 if any(acc_all) else 0

            distance_accs['top1'][step][dur] += 1 if sorted_human_dist[0] == sorted_distance_dist[0] else 0
            distance_acc_all = [1 if c in sorted_human_dist else 0 for c in sorted_distance_dist]
            # distance_acc_all = [1 if c == sorted_human_dist[0] else 0 for c in sorted_distance_dist]
            distance_accs['top-all'][step][dur] += 1 if any(distance_acc_all) else 0

            
accs = {_type: {step: {dur: accs[_type][step][dur] / len(model_results) for dur in accs[_type][step]} for step in accs[_type]} for _type in accs}
print('Accuracy')
print(accs)

distance_accs = {_type: {step: {dur: distance_accs[_type][step][dur] / len(model_results) for dur in distance_accs[_type][step]} for step in distance_accs[_type]} for _type in distance_accs}
print('Distance-based Accuracy')
print(distance_accs)
