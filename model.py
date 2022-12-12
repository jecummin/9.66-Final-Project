import json
import pyro
import torch
import numpyro
import numpy as np
import random as pyrandom
import jax.numpy as jnp
# import pyro.distributions as dist
# import pyro.distributions.constraints as constraints
# from pyro.distributions.util import scalar_like
# from pyro.infer import MCMC, NUTS, Predictive
# from pyro.infer.mcmc.util import initialize_model, summary
# from pyro.util import ignore_experimental_warning
from jax import random

import numpyro.distributions as dist
import numpyro.distributions.constraints as constraints
from numpyro.infer import MCMC, NUTS, Predictive, DiscreteHMCGibbs, HMC

import pandas as pd
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



#######
#######
#######
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
#######
#######
######
    

classes = sorted(list(class_stats.keys()))

ALL_DIMENSIONS = len(list(class_stats.values())[0]['mean'])
NUM_DIMS = ALL_DIMENSIONS
dims = np.random.choice(ALL_DIMENSIONS, NUM_DIMS)

means = jnp.array([np.array(class_stats[c]['mean'])[dims] for c in classes])
stds = jnp.array([np.array(class_stats[c]['std'])[dims] for c in classes])


#stds = jnp.ones((50, NUM_DIMS)) * jnp.mean(stds) # * jnp.mean(stds, axis=0)



#print(sorted(stds[:,dims[0]], reverse=True))
#exp_name = 'big_steps'
exp_name = 'warmup_20_og_std'
print(means.shape)

class_stats = {'mean': means, 'std': stds}
    
def model(stats,  obs):
    cls = numpyro.sample('cls', dist.Categorical(jnp.ones(len(classes)) / len(classes))).astype(int)
    means = stats['mean'][cls]
    cov = jnp.diag(stats['std'][cls])
    class_dist = dist.MultivariateNormal(means, cov)
    return numpyro.sample('obs', class_dist, obs=obs)

# def conditioned_model(model, stats, dims, emb):
#     return pyro.condition(model, data={"obs": emb})(stats, dims)

# for i in range(3):
#     obs = simple_model(class_stats, dims)
#     print(obs.shape)
# foo


# kernel = NUTS(conditioned_model)
# mcmc = MCMC(
#         kernel,
#         num_samples=5000,
#         warmup_steps=500,
#     )


images_and_embeddings = [(img, emb) for (img, emb) in list(images_to_embeddings.items()) if img in human_image_responses]
pyrandom.shuffle(images_and_embeddings)
sample_nums = [50, 150, 450, 1000]#, 150, 450, 1000]
#sample_nums = [2000, 5000]
model_samples = {}
num_images = 200
num_warmup = 20

with open('exp_images.json', 'r') as f:
    exp_images = json.load(f)

images_and_embeddings  = [m for m in images_and_embeddings if m[0] in exp_images]
for i, (img, emb) in enumerate(images_and_embeddings):
    hist = {}
    if i >= num_images:
        continue
    for j, num in enumerate(sample_nums):
    
        kernel = DiscreteHMCGibbs(NUTS(model), modified=True)

        mcmc = MCMC(
            kernel,
            num_samples=num,
            num_warmup=num_warmup,
        )

        obs = jnp.array(emb)[dims]
        mcmc.warmup(random.PRNGKey((3 + len(sample_nums)) * i + j), class_stats, obs)
        mcmc.run(random.PRNGKey((3 + len(sample_nums)) * i + j + 1), class_stats, obs)

        samples = mcmc.get_samples()
        sampled_classes = np.array(samples['cls'])
        hist[num] = {int(c): 0 for c in sampled_classes}
        for c in sampled_classes:
            hist[num][c] += 1

    print([(k, sorted(h.items(), key=lambda x: h[x[0]], reverse=True)) for k, h in hist.items()])
    print(human_image_responses[img])
    print(f'{i} / {num_images}')
    model_samples[img] = hist




    
with open(f'model_samples_{exp_name}.json', 'w') as f:
    json.dump(model_samples, f)

        
    #predictive = Predictive(model, posterior_samples=samples, infer_discrete=True)
    #y_pred = predictive(random.PRNGKey(3 * i + 2), class_stats, obs)
    #print(y_pred['cls'])
    #mcmc.print_summary()
    
    
    # print(mcmc.print_summary(prob=0.9))

