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
from torchvision.io import read_image

# %%
"""
Download an image from an url
-----------------------------
"""

device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"

url = (
    "https://culturezvous.com/wp-content/uploads/2017/10/chateau-azay-le-rideau.jpg?download=true"
)
#image_url = dinv.utils.load_url_image(url=url, img_size=100).to(device)
image_url = dinv.utils.load_url_image(url=url).to(device)

dinv.utils.plot([image_url])


# %%
"""
Download an image from an file
------------------------------
"""

image_path = "images/393035.jpg"
image_file = read_image(image_path)
image_file = image_file.unsqueeze(0).to(torch.float32)
dinv.utils.plot([image_file])


# %%
