import redis

# Connect to Redis
r = redis.Redis(host='localhost', port=6379, db=0)

# Key for the image data
key = 'image_4ed992b5-3c72-4ce8-a41d-1b115d3aa3b3'

# Fetch the image data
image_data = r.get(key)

# Save the image data to a file
if image_data:
    with open('output_image.png', 'wb') as file:
        file.write(image_data)
    print("Image saved successfully.")
else:
    print("No data found for the key:", key)
