# %%
# #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Feb. 10 09:24:27 2024

@author: nellypustelnik
"""

#pip install git+https://github.com/deepinv/deepinv.git#egg=deepinv
import deepinv as dinv
import torch
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.io as sio
mpl.rcParams.update(mpl.rcParamsDefault)
from PIL import Image
from deepinv.loss.metric import PSNR
perf_psnr = PSNR()
from torchvision.io import read_image



"""
Download an image
"""

device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"

img_size  = 100
device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"

url = (
    "https://culturezvous.com/wp-content/uploads/2017/10/chateau-azay-le-rideau.jpg?download=true"
)
x_true = dinv.utils.load_url_image(url=url, img_size=img_size).to(device)


#image_path = "images/393035.jpg"
#image_file = read_image(image_path)
#x_true = image_file.unsqueeze(0).to(torch.float32).to(device)/255

"""
Define the Forward Operator: study case of deblurring
-----------------------------------------------------
Load a forward operator $A$ and generate some (noisy) measurements. 
See the [full list is of available 
operators](https://deepinv.github.io/deepinv/deepinv.physics.html).
"""


# Define linear operator
filter_0 = dinv.physics.blur.gaussian_blur(sigma=(2, 0.1), angle=0.0)
physics = dinv.physics.Blur(filter_0, device=device, padding='reflect')

# Define noise
physics.noise_model = dinv.physics.GaussianNoise(sigma=0.1)

# Display original and degraded image
y = physics(x_true)
back = physics.A_adjoint(y)
dinv.utils.plot([x_true, y, back], titles=['original','observation','backprojection'])


"""
Define data-fidelity + prior for TV-isotropic
---------------------------------------------
"""
# Define data fidelity term
data_fidelity = dinv.optim.L2()

# Define prior
L         =  dinv.optim.TVPrior().nabla
L_adjoint =  dinv.optim.TVPrior().nabla_adjoint
prior     =  dinv.optim.L1Prior()
Lnorm2       = 8

tensor_shape = x_true.shape
random_tensor = torch.randn(tensor_shape).to(device)
Anorm2 = physics.compute_norm(random_tensor)

"""
Unfolded TV-isotropic with Condat-Vu algorithm
----------------------------------------------------
"""
import torch.nn as nn
class CVunfolded(nn.Module):
    def __init__(self, max_iter=30):
        super(CVunfolded, self).__init__()
        self.device = device

        # Determine the number of iterations based on GPU availability
        self.max_iter = 20 if torch.cuda.is_available() else 20

        # Initialize parameters as trainable
        self.param_regul = torch.nn.Parameter(0.01 * torch.ones(self.max_iter, device=self.device))


   
    def forward(self, x_initial, y, physics, L, L_adjoint, data_fidelity, prior, Anorm2, Lnorm2):

        xk = x_initial.to(device=self.device)
        vk = L(xk).to(device=self.device)
        param_gamma    = 1                   # Set the regularity of the solution
        param_tau      = 0.9/(Anorm2/2 + Lnorm2*param_gamma)

        # Store the convergence criterion across iterations (optional tracking)
        #with torch.no_grad():
        for k in range(self.max_iter):
            x_prev = xk.clone()

            # Update xk using the data fidelity prox
            xk = xk - param_tau*data_fidelity.grad(xk, y, physics) - param_tau*L_adjoint(vk)

            # Update vk using the prior prox
            tmp = vk + param_gamma * L(2 * xk - x_prev)
            vk = tmp - param_gamma * prior.prox(tmp / param_gamma, gamma=torch.exp(self.param_regul[k]) / param_gamma)
                

        # Final output after max_iter iterations
        out = xk


        return out


"""
Estimation of the parameters in the unfolded scheme
----------------------------------------------------
"""

x_initial = back.to(device)
model = CVunfolded()
niter = 10000
psnr       = 1e10 * np.ones(niter)
loss_value = 1e10 * np.ones(niter)

# Initialize optimizer for the parameters
optimizer = torch.optim.Adam(model.parameters())

# Initialize loss function
#loss_fn = dinv.loss.SupLoss()
loss_fn = torch.nn.MSELoss()

for iteration in range(niter):
    print(f'epoch {iteration}/{niter}')
    
    # Zero the gradients
    optimizer.zero_grad()

    # Forward pass
    out = model.forward(x_initial, y, physics, L, L_adjoint, data_fidelity, prior, Anorm2, Lnorm2)

    # Compute the loss
    total_loss = loss_fn(x_true, out)
    psnr[iteration] = perf_psnr(x_true,out).item() 
    loss_value[iteration] = total_loss.item()

    # Backward pass
    total_loss.backward()
    optimizer.step()



# Display results
dinv.utils.plot([x_true, y, out], titles=['original','observation','restored'],figsize=[6,18])
fig, axs = plt.subplots(1, 2, figsize=(10, 4))  # 2 lignes, 1 colonne
axs[0].plot(loss_value)
axs[0].legend()
axs[0].set_title('Objective function w.r.t iterations')
axs[1].plot(psnr)
axs[1].legend()
axs[1].set_title('PSNR function w.r.t iterations')
plt.tight_layout()
plt.show()
