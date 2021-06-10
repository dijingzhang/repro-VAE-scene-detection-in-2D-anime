import torch
import torch.utils.data as data
import numpy as np

"""
Generate the latent space of each frame
return numpy([N, latent_dims])
"""

def generate_code(model, model_state, dataset, latent_dim):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_loader = data.DataLoader(dataset, batch_size=64, shuffle=False)

    latent = []
    model.load_state_dict(model_state)
    model.eval()
    with torch.no_grad():
        for images in data_loader:
            imgs = images.to(device)
            z = model.generate_latent(imgs) # Tensor[B, latent_dims]
            assert z.shape[1] == latent_dim
            z = np.array(z.cpu())
            latent.append(z)

    latent = np.array(latent).reshape((-1, latent_dim))
    return latent  # [N, latent_dims]

def generate_img(model, model_state, img, transform):
    img = transform(img)
    C, H, W = img.shape
    img = img.reshape((1, C, H, W))
    model.load_state_dict(model_state)
    model.eval()
    with torch.no_grad():
        img_re = model.generate_images(img)
    return img_re




