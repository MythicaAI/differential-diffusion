from flask import Flask, request, jsonify
import torch
from PIL import Image
from torchvision import transforms
from diff_pipe import StableDiffusionDiffImg2ImgPipeline
from io import BytesIO
import base64

app = Flask(__name__)

device = "cuda"

#This is the default model, you can use other fine tuned models as well
pipe = StableDiffusionDiffImg2ImgPipeline.from_pretrained("stabilityai/stable-diffusion-2-1-base",
                                                          torch_dtype=torch.float16).to(device)


def preprocess_image(image):
    image = image.convert("RGB")
    image = transforms.CenterCrop((image.size[1] // 64 * 64, image.size[0] // 64 * 64))(image)
    image = transforms.ToTensor()(image)
    image = image * 2 - 1
    image = image.unsqueeze(0).to(device)
    return image


def preprocess_map(map):
    map = map.convert("L")
    map = transforms.CenterCrop((map.size[1] // 64 * 64, map.size[0] // 64 * 64))(map)
    # convert to tensor
    map = transforms.ToTensor()(map)
    map = map.to(device)
    return map

@app.route('/v1/mythica/inpaint', methods=['POST'])
def inpaint():
    try:
        # Get data from the request
        data = request.json

        # Decode image from base64
        image_data = base64.b64decode(data['image'])
        map_data = base64.b64decode(data['map'])

        # Open image and map using PIL
        image = Image.open(BytesIO(image_data))
        map = Image.open(BytesIO(map_data))

        # Preprocess the images
        image = preprocess_image(image)
        map = preprocess_map(map)

        # Get additional parameters
        prompt = data.get('prompt', ["inpaint in the same style as the rest of the image"])
        negative_prompt = data.get('negative_prompt', ["blurry, different, seams"])
        guidance_scale = data.get('guidance_scale', 7)
        num_inference_steps = data.get('num_inference_steps', 50)
        num_images_per_prompt = 1 #data.get('num_images_per_prompt', 1)
        
        # Run the pipeline
        edited_image = pipe(
            prompt=prompt,
            image=image,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_images_per_prompt,
            negative_prompt=negative_prompt,
            map=map,
            num_inference_steps=num_inference_steps
        ).images[0]

        # Convert the edited image to base64
        buffered = BytesIO()
        edited_image.save(buffered, format="PNG")
        edited_image_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return jsonify({"image": edited_image_str})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
