import os
from fastapi import FastAPI, HTTPException, Form
from fastapi.responses import StreamingResponse, PlainTextResponse, JSONResponse
from diffusers import AutoPipelineForText2Image
import torch
import io
import logging
import time
import aioredis
import uuid
import asyncio

app = FastAPI()

# Set up basic logging
logging.basicConfig(level=logging.INFO)

# Environment setup and model loading
mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
device_type = "cuda" if torch.cuda.is_available() else "mps" if mps_available else "cpu"
torch_dtype = torch.float16 if device_type == "cuda" else torch.float32

# Define device based on the type
device = torch.device(device_type)

# Load text-to-image model
t2i_pipe = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/sdxl-turbo",
    safety_checker=None,
    torch_dtype=torch_dtype
).to(device)

# Redis connection URL
REDIS_URL = "redis://localhost:6379"

# Initialize global variables
redis = None

# Initialize an asyncio Queue
task_queue = asyncio.Queue()

# Constants
USER_REQUEST_PREFIX = "user_req_count_"
MAX_REQUESTS_PER_USER = 3

async def worker(name):
    global redis, image_generation_times  # Global declaration for the image_generation_times list
    while True:
        # Get a "work item" out of the queue.
        prompt, num_inference_steps, guidance_scale, requestId, user_id = await task_queue.get()

        try:
            # Perform the image generation task.
            logging.info(f"Worker {name} is processing the task: {requestId}")
            start_time = time.time()
            results = t2i_pipe(
                prompt=prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
            )
            img = results.images[0]
            generation_duration = time.time() - start_time
            
            # Log the image generation duration
            logging.info(f"Image generation duration: {generation_duration:.2f} seconds")

            # Update the image generation times list
            image_generation_times.append(generation_duration)
            # Keep only the last three image generation times
            image_generation_times = image_generation_times[-3:]

            # Store the result in Redis using the requestId as the key.
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='PNG')
            img_byte_arr.seek(0)
            await redis.set(f"result_{requestId}", img_byte_arr.getvalue())
            await redis.hset("status", requestId, "done")

            logging.info(f"Worker {name} has completed the task: {requestId}")
        except Exception as e:
            logging.error(f"Error processing request {requestId}: {e}")
        finally:
            # Always decrement the user's active request count in a finally block to ensure it's executed
            user_req_key = f"{USER_REQUEST_PREFIX}{user_id}"
            await redis.decr(user_req_key)
        
        task_queue.task_done()

@app.on_event("startup")
async def startup_event():
      global redis
      # Use the REDIS_URL environment variable, default to local Redis
      redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
      redis = await aioredis.from_url(redis_url, decode_responses=False)
      for i in range(3):  # Number of workers
        asyncio.create_task(worker(f'worker-{i}'))

@app.on_event("shutdown")
async def shutdown_event():
    await redis.close()



@app.on_event("shutdown")
async def shutdown_event():
    await redis.close()

@app.post("/generate-image/")
async def api_generate_image(prompt: str = Form(...), num_inference_steps: int = Form(2), guidance_scale: float = Form(0.0), user_id: str = Form(...)):
    logging.info(f"Received prompt: {prompt} from user: {user_id}")

    user_req_key = f"{USER_REQUEST_PREFIX}{user_id}"

    async with await redis.pipeline(transaction=True) as pipe:
        try:
            # This should be an atomic operation
            current_count = await pipe.get(user_req_key).incr(user_req_key).execute()
            current_count = int(current_count[0]) if current_count[0] else 0

            if current_count > MAX_REQUESTS_PER_USER:
                # Compensate for the increment we just did if we're not processing this request
                await redis.decr(user_req_key)
                raise HTTPException(
                    status_code=429, 
                    detail="User quota of 3 requests reached. Please try again later."
                )
        except aioredis.exceptions.WatchError:
            # This means the value changed in the interim; the client should try again.
            raise HTTPException(
                status_code=429, 
                detail="User quota check failed, please try again."
            )

    # Generate a unique request ID
    requestId = str(uuid.uuid4())

    # Include user_id in the task details
    await task_queue.put((prompt, num_inference_steps, guidance_scale, requestId, user_id))
    return {"requestId": requestId}



@app.get("/health")
async def health_check():
    return PlainTextResponse("OK", status_code=200)

@app.get("/hardware")
async def hardware_check():
    return PlainTextResponse(device_type, status_code=200)

image_generation_times = []

@app.get("/timing")
async def timing():
    # Return the last three image generation times
    return JSONResponse(content={"last_three_generation_times": image_generation_times})

@app.get("/get-result/{requestId}")
async def get_result(requestId: str):
    # Attempt to retrieve the image data from Redis using the correct key
    image_data = await redis.get(f"result_{requestId}")  # No encoding parameter needed
    if image_data:
        return StreamingResponse(io.BytesIO(image_data), media_type="image/png")
    else:
        # If the image isn't found, manually decode the status as it's expected to be text
        task_status = await redis.hget("status", requestId)
        if task_status and task_status.decode('utf-8') == "done":
            raise HTTPException(status_code=404, detail="Image not found. Task completed, but no result.")
        return {"status": "processing"}


