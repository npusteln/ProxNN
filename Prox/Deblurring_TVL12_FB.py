# %%
# #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 09:24:27 2024

@author: nellypustelnik
"""

#pip install git+https://github.com/deepinv/deepinv.git#egg=deepinv
import deepinv as dinv
import torch
from torchvision.io import read_image
import numpy as np
import matplotlib.pyplot as plt
from deepinv.loss.metric import PSNR
perf_psnr = PSNR()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


"""
Download an image
"""
image_path = "images/393035.jpg"
image_file = read_image(image_path)
x_true = image_file.unsqueeze(0).to(torch.float32).to(device)/255


"""
Define the Forward Operator: study case of deblurring
-----------------------------------------------------
Load a forward operator $A$ and generate some (noisy) measurements. 
See the [full list is of available 
operators](https://deepinv.github.io/deepinv/deepinv.physics.html).
"""

# Convolution forward model
filter_0 = dinv.physics.blur.gaussian_blur(sigma=(2, 0.9), angle=0.0)
physics = dinv.physics.Blur(filter_0, device=device, padding='reflect')

# Add noise in the forward model
physics.noise_model = dinv.physics.GaussianNoise(sigma=0.1)

y = physics(x_true)
back = physics.A_dagger(y)
dinv.utils.plot([x_true, y, back], titles=['original','observation','backprojection'])


tensor_shape = x_true.shape
random_tensor = torch.randn(tensor_shape).to(device)
Anorm2 = physics.compute_norm(random_tensor)


"""
Reconstruction TV-isotropic with Forward-Backward algorithm
------------------------------------------------------------
"""

# Define data fidelity term
data_fidelity = dinv.optim.L2()

# Define prior
prior     =  dinv.optim.TVPrior()

# Define regularization parameter
param_regul    = 0.03;

# Iterations
param_gamma    = 1.9/Anorm2         # Set the step-size
param_iter     = 30                 # number of iterations


xk = y
crit = 1e10*np.ones(param_iter)
psnr = 1e10*np.ones(param_iter)
for k in range(param_iter):

  x_prev = xk.clone()
  xk = xk - param_gamma*data_fidelity.grad(xk, y, physics)
  xk = prior.prox(xk,  gamma = param_gamma*param_regul)
  crit[k] = data_fidelity(xk, y, physics) + param_regul*prior.fn(xk)
  psnr[k] = perf_psnr(x_true,xk).item()


# %%

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



# %%
