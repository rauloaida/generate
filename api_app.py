from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, PlainTextResponse
from diffusers import DiffusionPipeline
from pydantic import BaseModel
import io
import logging

app = FastAPI()

# Set up basic logging
logging.basicConfig(level=logging.INFO)

# Load your model (consider doing this outside of your request handling to save time)
pipe = DiffusionPipeline.from_pretrained("stabilityai/sdxl-turbo").to("cpu")

class ImagePrompt(BaseModel):
    prompt: str

@app.post("/generate-image/")
async def generate_image(image_prompt: ImagePrompt):
    logging.info(f"Received prompt: {image_prompt.prompt}")
    try:
        # Generate the image with the provided prompt
        results = pipe(
            prompt=image_prompt.prompt,
            num_inference_steps=2,
            guidance_scale=0.0,
        )
        img = results.images[0]

        # Convert the PIL image to a byte stream
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)  # Go to the start of the byte stream

        # Return the image directly in the response
        logging.info("Image generation successful")
        return StreamingResponse(img_byte_arr, media_type="image/png")
    except Exception as e:
        logging.error(f"Error during image generation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    return PlainTextResponse("OK", status_code=200)