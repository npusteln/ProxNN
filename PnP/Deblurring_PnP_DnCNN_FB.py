#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 09:24:27 2024
@author: Nelly Pustelnik
"""

#pip install git+https://github.com/deepinv/deepinv.git#egg=deepinv
import deepinv as dinv
import torch
from torchvision.io import read_image
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.io as sio
from deepinv.loss.metric import PSNR
perf_psnr = PSNR()

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'device is {device}')

# Download an image
image_path = "images/393035.jpg"
image_file = read_image(image_path)
x_true = image_file.unsqueeze(0).to(torch.float32).to(device)/255


# Define the Forward Operator: study case of deblurring + Gaussian noise
#-----------------------------------------------------------------------
# Load a forward operator $A$ and generate some (noisy) measurements. 
# The full list of operators is available here:  
# (https://deepinv.github.io/deepinv/deepinv.physics.html).

# Define linear operator
filter_0 = dinv.physics.blur.gaussian_blur(sigma=(2, 0.1), angle=0.0)
physics = dinv.physics.Blur(filter_0, device=device, padding='reflect')

# Define noise
physics.noise_model = dinv.physics.GaussianNoise(sigma=0.01)

# Display original and degraded image
y = physics(x_true)
back = physics.A_adjoint(y)
dinv.utils.plot([x_true, y, back], titles=['original','observation','backprojection'])



# Reconstruction PnP with Forward-Backward algorithm
#---------------------------------------------------


# Define data fidelity term
data_fidelity = dinv.optim.L2()

# Define prior
denoiser = dinv.models.DnCNN(in_channels=3, out_channels=3, pretrained="download_lipschitz", device=device) 
prior = dinv.optim.prior.PnP(denoiser=denoiser)

# Define regularization parameter
param_regul    = 0.01;

# Define algorithm parameters
random_tensor   = torch.randn(x_true.shape).to(device)
Anorm2          = physics.compute_norm(random_tensor)
param_gamma     = 0.9/ Anorm2         # Set the step-size
param_iter      = 200                 # number of iterations

# Iterations
xk = back.clone()
crit = 1e10*np.ones(param_iter)
psnr = 1e10*np.ones(param_iter)
with torch.no_grad():
    for k in range(param_iter):
        xk_prev = xk.clone()
        xk = xk - param_gamma*data_fidelity.grad(xk, y, physics)
        xk = denoiser(xk, param_gamma*param_regul)
        crit[k] = torch.linalg.norm(xk.flatten()-xk_prev.flatten())
        psnr[k] = perf_psnr(x_true,xk).item()
        if k % 10 == 0: print(f"crit[{k}]: {crit[k]}")


# Display results
dinv.utils.plot([x_true, y, xk], titles=['original','observation','restored'],figsize=[6,18])
fig, axs = plt.subplots(1, 2, figsize=(10, 4))  # 2 lignes, 1 colonne
axs[0].plot(crit)
axs[0].legend()
axs[0].set_title('Objective function w.r.t iterations')
axs[1].plot(psnr)
axs[1].legend()
axs[1].set_title('PSNR function w.r.t iterations')
plt.tight_layout()
plt.show()
