from fastapi import FastAPI, HTTPException, Form
from fastapi.responses import StreamingResponse
from diffusers import AutoPipelineForText2Image
import torch
import io
import aioredis  # Ensure correct import for async Redis operations
import uuid 

app = FastAPI()

# Setup for the Diffusers pipeline
MODEL_ID = "stabilityai/sdxl-turbo"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PIPELINE = AutoPipelineForText2Image.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float32 if DEVICE == "cpu" else torch.float16
).to(DEVICE)

REDIS_URL = "redis://localhost:6379"

@app.on_event("startup")
async def startup_event():
    global redis
    # Use aioredis.create_redis_pool for aioredis versions <2.0
    # For aioredis versions >=2.0, from_url is used directly to create a Redis connection
    redis = await aioredis.from_url(REDIS_URL, decode_responses=False)

@app.on_event("shutdown")
async def shutdown_event():
    await redis.close()

@app.post("/generate-image/")
async def api_generate_image(prompt: str = Form(...)):
    request_id = str(uuid.uuid4()) # Hardcoded request ID for debugging
    await redis.hset("status", request_id, "pending")
    
    # Run image generation synchronously
    results = PIPELINE(prompt=prompt, num_inference_steps=2, guidance_scale=0.0)
    img = results.images[0]
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    
    # Store the image data in Redis
    await redis.set(f"image_{request_id}", img_byte_arr.getvalue())
    
    # Update the status to done
    await redis.hset("status", request_id, "done")

    return {"request_id": request_id}

@app.get("/get-image/{request_id}")
async def get_image(request_id: str):
    image_data = await redis.get(f"image_{request_id}")

    if image_data:
        return StreamingResponse(io.BytesIO(image_data), media_type="image/png")
    else:
        raise HTTPException(status_code=404, detail="Request ID not found")

