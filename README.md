# 9.66-Final-Project

This is the repo for Jesse Cummings' 9.66 project entitled Modeling Human Object Recognition Behavior as Bayesian Inference. The most important scripts are:

1. `preprocess_images.py` which runs images through the CLIP encoder to produce image embeddings and computes object class means and variances for use in the model.
2. `model.py` which builds the GMM for image classification and collects latent class samples conditioned on image embeddings as observations.
3. `compute_divergence.py` which evaluates the inferred class distributions of the model on their similarity to the human observations according to a collection of metrics presented in the paper.
