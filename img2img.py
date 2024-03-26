# load_ext autoreload
# autoreload 2

import os
from PIL import Image, ImageOps
import requests
import torch
print(torch.__version__)
# exit()
import matplotlib.pyplot as plt
import numpy as np

import torch
import requests
from tqdm import tqdm
from io import BytesIO
from diffusers import StableDiffusionImg2ImgPipeline
import torchvision.transforms as T

from utils import preprocess, recover_image

import json

torch.cuda.empty_cache()

def pgd(X, model, eps=0.1, step_size=0.015, iters=40, clamp_min=0, clamp_max=1, mask=None):
    X_adv = X.clone().detach() + (torch.rand(*X.shape)*2*eps-eps).cuda()
    pbar = tqdm(range(iters))
    for i in pbar:
        actual_step_size = step_size - (step_size - step_size / 100) / iters * i  

        X_adv.requires_grad_(True)

        loss = (model(X_adv).latent_dist.mean).norm()

        pbar.set_description(f"[Running attack]: Loss {loss.item():.5f} | step size: {actual_step_size:.4}")

        grad, = torch.autograd.grad(loss, [X_adv])
        
        X_adv = X_adv - grad.detach().sign() * actual_step_size
        X_adv = torch.minimum(torch.maximum(X_adv, X - eps), X + eps)
        X_adv.data = torch.clamp(X_adv, min=clamp_min, max=clamp_max)
        X_adv.grad = None    
        
        if mask is not None:
            X_adv.data *= mask
            
    return X_adv

if __name__ == '__main__':
    prompts = {}
    json_file_path = 'flickr8k/gemini_captions.json'

    # Open the JSON file
    with open(json_file_path, 'r') as file:
        prompts = json.load(file)

    to_pil = T.ToPILImage()

    model_id_or_path = "runwayml/stable-diffusion-v1-5"

    pipe_img2img = StableDiffusionImg2ImgPipeline.from_pretrained(
        model_id_or_path,
        revision="fp16", 
        torch_dtype=torch.float16,
    )
    pipe_img2img = pipe_img2img.to("cuda")

    for index in range(6821,8092):
        image_path = "original-images/" + str(index) + ".jpg"

        # response = requests.get(url)
        init_image = Image.open(image_path).convert("RGB")
        resize = T.transforms.Resize(512)
        center_crop = T.transforms.CenterCrop(512)
        init_image = center_crop(resize(init_image))
        init_image


        with torch.autocast('cuda'):
            X = preprocess(init_image).half().cuda()
            adv_X = pgd(X, 
                        model=pipe_img2img.vae.encode, 
                        clamp_min=-1, 
                        clamp_max=1,
                        eps=0.06, # The higher, the less imperceptible the attack is 
                        step_size=0.02, # Set smaller than eps
                        iters=100, # The higher, the stronger your attack will be
                    )
            
            # convert pixels back to [0,1] range
            adv_X = (adv_X / 2 + 0.5).clamp(0, 1)

        adv_image = to_pil(adv_X[0]).convert("RGB")
        adv_image

        prompt = prompts[str(index)]

        # a good seed (uncomment the line below to generate new images)
        SEED = 9222
        # SEED = np.random.randint(low=0, high=10000)

        # Play with these for improving generated image quality
        STRENGTH = 0.5
        GUIDANCE = 7.5
        NUM_STEPS = 50

        with torch.autocast('cuda'):
            torch.manual_seed(SEED)
            image_nat = pipe_img2img(prompt=prompt, image=init_image, strength=STRENGTH, guidance_scale=GUIDANCE, num_inference_steps=NUM_STEPS).images[0]
            torch.manual_seed(SEED)
            image_adv = pipe_img2img(prompt=prompt, image=adv_image, strength=STRENGTH, guidance_scale=GUIDANCE, num_inference_steps=NUM_STEPS).images[0]

        image_nat.save('gen-nat-images/gen_nat_' + str(index) + '.jpg')
        image_adv.save('gen-adv-images/gen_adv_' + str(index) + '.jpg')
        adv_image.save('adv-images/adv_' + str(index) + '.jpg')
        
        # exit()

# torch 2.0.0+cu121
# torchvision 0.16