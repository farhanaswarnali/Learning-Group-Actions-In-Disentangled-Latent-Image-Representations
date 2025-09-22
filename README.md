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

\[
\mathcal{L}_{total} = \mathcal{L}_{recon} + \lambda_i \mathcal{L}_{inv} + \lambda_v \mathcal{L}_{const}
\]

where:
- **Reconstruction loss (L₍recon₎):** Mean Squared Error (MSE) between reconstructed and rotated images.  
- **Invariant loss (L₍inv₎):**  

\[
\mathcal{L}_{inv} = \|z_i(x) - \mathrm{sg}[z_i(T_g(x))]\|^2
\]

ensures that invariant latent features \(z_i\) are identical for the original and transformed images.  

- **Consistency loss (L₍const₎):**  

\[
\mathcal{L}_{const} = \|\Phi_g^v(z_v(x)) - \mathrm{sg}[z_v(T_g(x))]\|^2
\]

ensures that **transforming the variant factors in latent space produces the same result as extracting them from the actually transformed image**, leading to consistent and equivariant representation learning.  

Here, **sg[·]** (stop-gradient) prevents gradients from flowing through the encoder, so learning focuses on latent-space transformations rather than altering feature extraction.  

Hyperparameters λᵢ and λᵥ are set to **1**, and the **threshold τ** controls the sparsity of the learned latent partition.  
We **jointly optimize all network parameters**, including the Adaptive Latent Disentanglement (ALD) and group action modules, enabling automatic discovery of meaningful latent partitions and their transformations.  

Training uses **Adam optimizer** (learning rate `1e-3`), batch size `64`, for `50` epochs, saving the model with the **lowest validation loss**.

