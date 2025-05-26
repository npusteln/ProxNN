# %%
# #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 10:35:06 2024

@author: nellypustelnik
"""

#pip install git+https://github.com/deepinv/deepinv.git#egg=deepinv
import deepinv as dinv
import torch


# %%
"""
Download an image
-----------------
"""

device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"

url = (
    "https://culturezvous.com/wp-content/uploads/2017/10/chateau-azay-le-rideau.jpg?download=true"
)
x_true = dinv.utils.load_url_image(url=url, img_size=100).to(device)


# %%
"""
Define linear operator : Inpainting
-----------------------------------
Load a forward operator $A$ and generate some (noisy) measurements. 
See the [full list is of available operators]
(https://deepinv.github.io/deepinv/deepinv.physics.html).

"""

# Inpainting

physics = dinv.physics.Inpainting(
    tensor_size=x_true.shape[1:],
    mask=0.3,
    device=device
    )
physics.noise_model = dinv.physics.GaussianNoise(sigma=0.01)

y = physics(x_true)

back = physics.A_adjoint(y)
dinv.utils.plot([x_true, y], titles=['original','observation'],figsize=[6,18])


"""
Check the adjoint
------------------
"""

r1 = torch.randn(1, 3, 100, 100)
rt = physics(r1)
r2 = torch.randn(rt.shape).to(device) 
dot_product1 = torch.sum(r2 * physics.A(r1))
dot_product2 = torch.sum(r1 * physics.A_adjoint(r2))
print(f" <y,A(x)> =  {dot_product1.item()}")
print(f" <A_adjoint(y),x> =   {dot_product2.item()}")


"""
Operator norm : ||A||^2
----------------------
"""

tensor_shape = x_true.shape
random_tensor = torch.randn(tensor_shape)
Anorm2 = physics.compute_norm(random_tensor)
print(f" ||A||^2=   {Anorm2.item()}")

# %%
