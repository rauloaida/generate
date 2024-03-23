from fastapi import FastAPI, HTTPException, Form
from fastapi.responses import StreamingResponse, PlainTextResponse, JSONResponse
from diffusers import AutoPipelineForText2Image
import torch
import io
import logging
import time

app = FastAPI()

# Set up basic logging
logging.basicConfig(level=logging.INFO)

# Global list to store the last three image generation times
image_generation_times = []

# Environment setup and model loading
mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
device_type = "cuda" if torch.cuda.is_available() else "mps" if mps_available else "cpu"
torch_dtype = torch.float16 if device_type == "cuda" else torch.float32

# Log the device being used
logging.info(f"Using {device_type} for acceleration")

# Define device based on the type
device = torch.device(device_type)

# Load text-to-image model without the need for an HF token
# Make sure to define and initialize t2i_pipe here
t2i_pipe = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/sdxl-turbo",
    safety_checker=None,
    torch_dtype=torch_dtype
).to(device)

@app.post("/generate-image/")
async def generate_image(prompt: str = Form(...), num_inference_steps: int = Form(2), guidance_scale: float = Form(0.0)):
    global image_generation_times
    logging.info(f"Received prompt: {prompt}")
    start_time = time.time()

    # Text to image generation
    results = t2i_pipe(
        prompt=prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
    )
    img = results.images[0]

    generation_duration = time.time() - start_time
    logging.info(f"Image generation duration: {generation_duration:.2f} seconds")

    # Store the generation duration, keeping only the last three records
    image_generation_times.append(generation_duration)
    image_generation_times = image_generation_times[-3:]

    # Convert the PIL image to a byte stream for response
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)

    logging.info("Image generation successful")
    return StreamingResponse(img_byte_arr, media_type="image/png")

@app.get("/health")
async def health_check():
    return PlainTextResponse("OK", status_code=200)

@app.get("/hardware")
async def hardware_check():
    return PlainTextResponse(device_type, status_code=200)

@app.get("/timing")
async def timing():
    # Return the last three image generation times
    return JSONResponse(content={"last_three_generation_times": image_generation_times})
