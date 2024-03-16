from diffusers import DiffusionPipeline

# Path to your local model directory
local_model_dir = "./sdxl-turbo"

# Load the pipeline model from the local directory
pipe = DiffusionPipeline.from_pretrained(local_model_dir).to("cpu")

# Loop to continuously ask for prompts and generate images
while True:
    # Get the prompt from the user
    prompt = input("Enter your prompt, or type 'exit' to quit: ")

    # Check if the user wants to exit the loop
    if prompt.lower() == 'exit':
        break

    # Generate the image
    results = pipe(
        prompt=prompt,
        num_inference_steps=5,
        guidance_scale=0.0,
    )

    # Save the image
    img = results.images[0]
    # Sanitize the prompt to create a valid filename
    safe_prompt = "".join(char for char in prompt if char.isalnum() or char in " -_")
    filename = f"image_{safe_prompt}.png"
    img.save(filename)
    # test comment
    print(f"Image saved as '{filename}'")
