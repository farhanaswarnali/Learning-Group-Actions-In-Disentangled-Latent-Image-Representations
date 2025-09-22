# Learning-Group-Actions-In-Disentangled-Latent-Image-Representations

## Abstract
Modeling group actions on latent representations enables controllable transformations of high-dimensional image data. Prior works applying group-theoretic priors or modeling transformations typically operate in the high-dimensional data space, where group actions apply uniformly across the entire input, making it difficult to disentangle the subspace that varies under transformations. While latent-space methods offer greater flexibility, they still require manual partitioning of latent variables into equivariant and invariant subspaces, limiting the ability to robustly learn and operate group actions within the latent space. To address this, we introduce a novel end-to-end framework that for the first time learns group actions on latent image manifolds, automatically discovering transformation-relevant structures without manual intervention. Our method uses learnable binary masks with straight-through estimation to dynamically partition latent representations into transformation-sensitive and invariant components. We formulate this within a unified optimization framework that jointly learns latent disentanglement and group transformation mappings. The framework can be seamlessly integrated with any standard encoder-decoder architecture. We validate our approach on five 2D/3D image datasets, demonstrating its ability to automatically learn disentangled latent factors for group actions, while downstream classification tasks confirm the effectiveness of the learned representations.

 ![image alt](https://github.com/farhanaswarnali/Learning-Group-Actions-In-Disentangled-Latent-Image-Representations/blob/19bc0d8cfd87287f619b055e320182b322c4f870/Architecture.png)


## Requirements
To run this project, following Python packages are needed:
- torch
- torchvision
- matplotlib
- scikit-learn
- numpy

## Dataset
We use a variant of the original **MNIST** dataset, consisting of 70,000 grayscale handwritten digits across 10 classes (0–9), with each image of size 28×28.  
Here, we are providing the code for the **Rotated MNIST** dataset in this repo.

## Training Process

Our architecture employs an **encoder–decoder framework** with convolutional downsampling modules in the encoder and corresponding upsampling modules in the decoder.  

During training:
- We **randomly sample pairs of data points** from the training set.
- The model is optimized using the total loss:

![L_total](https://latex.codecogs.com/svg.latex?\mathcal{L}_{\text{total}}=\mathcal{L}_{\text{recon}}+\lambda_i\mathcal{L}_{\text{inv}}+\lambda_v\mathcal{L}_{\text{const}})

where:

- **Reconstruction loss**

![L_recon](https://latex.codecogs.com/svg.latex?\mathcal{L}_{\text{recon}}=||D_{\theta}(\Phi_g(E_{\phi}(x)))-T_g(x)||^2)

ensures that the decoder output of the transformed latent representation matches the ground truth transformed image.

- **Invariant loss**

![L_inv](https://latex.codecogs.com/svg.latex?\mathcal{L}_{\text{inv}}=||z_i^{(x)}-\mathrm{sg}[z_i^{(T_g(x))}]||^2)

constrains the invariant latent features to be identical for both the original and transformed inputs.

- **Consistency loss**

![L_const](https://latex.codecogs.com/svg.latex?\mathcal{L}_{\text{const}}=||\Phi_g^v(z_v^{(x)})-\mathrm{sg}[z_v^{(T_g(x))}]||^2)

ensures that transforming the variant factors in latent space produces the same result as those extracted from the actually transformed image.

Here, **sg[·]** (stop-gradient) prevents gradients from flowing through the encoder, focusing learning on latent-space transformations rather than feature extraction adjustments.

Hyperparameters λᵢ and λᵥ are set to **1**, and the threshold τ controls the sparsity of the learned latent partition.  
We **jointly optimize all network parameters**, including the Adaptive Latent Disentanglement (ALD) and group action modules, enabling automatic discovery of meaningful latent partitions and their transformations.  

Training uses the **Adam optimizer** (learning rate `1e-3`), batch size `64`, for `50` epochs, saving the model with the **lowest validation loss**.


## Usage

Run `rotated_mnist.py` to train the model on **Rotated MNIST**.  
The script takes care of data loading, model setup, loss computation, training, validation, and saves the best model based on validation loss.

