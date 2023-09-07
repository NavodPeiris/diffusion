from diffusers import StableDiffusionPipeline
import torch

model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)

from fastapi import FastAPI, Request
import os
from fastapi.responses import FileResponse
import socket
from pydantic import BaseModel
import glob

app = FastAPI()

class Prompt(BaseModel):
    prompt: str

hostname = socket.gethostname()
ip_address = socket.gethostbyname(hostname)
print("ip_address : ", ip_address)

@app.get('/diffusion/health')
async def hi():
    return {"response": "server running"}

@app.post('/diffusion/infer')
async def imageGen(request: Request, prompt: Prompt):

    out_path = 'result_images/*'

    # deleting result images
    for path in glob.glob(out_path):
        if os.path.exists(path):
            os.remove(path)

    folder_path = './result_images'

    prompt_text = prompt.prompt

    image = pipe(prompt_text).images[0]

    # Save the image with the given prompt as the filename
    image_path = os.path.join(folder_path, f'{prompt_text}.png')
    image.save(image_path)

    return FileResponse(image_path)
