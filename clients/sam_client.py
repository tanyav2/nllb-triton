import tritonclient.grpc as grpcclient
import numpy as np
import argparse
import json
import matplotlib.pyplot as plt
from PIL import Image

def show_on_image(raw_image, boxes=None, points=None, labels=None, masks=None, scores=None, save_path=None):
    plt.figure(figsize=(10, 10))
    plt.imshow(raw_image)
    
    if boxes:
        for box in boxes:
            x0, y0, w, h = box[0], box[1], box[2] - box[0], box[3] - box[1]
            plt.gca().add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))
    
    if points:
        coords = np.array(points)
        pos_points = coords[labels == 1] if labels else coords
        plt.gca().scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=375, edgecolor='white', linewidth=1.25)
        if labels:
            neg_points = coords[labels == 0]
            plt.gca().scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=375, edgecolor='white', linewidth=1.25)
    
    if masks:
        for i, mask in enumerate(masks):
            color = np.array([30/255, 144/255, 255/255, 0.6])
            mask_image = mask.reshape(*mask.shape[-2:], 1) * color.reshape(1, 1, -1)
            plt.gca().imshow(mask_image)
    
    plt.axis('on')
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
        plt.close()
    else:
        plt.show()


def main(model_name, port, data):
    client = grpcclient.InferenceServerClient(url=f"localhost:{port}")
    img = Image.open(data["image_file"])
    with open(data["image_file"], "rb") as f:
        img_buf = np.array([f.read()])

    inputs = [
        grpcclient.InferInput("image", img_buf.shape, datatype="BYTES", data=img_buf),
        grpcclient.InferInput("points", np.array(data.get("points", []), dtype=np.float32)),
        grpcclient.InferInput("labels", np.array(data.get("labels", []), dtype=np.int32)),
        grpcclient.InferInput("box", np.array(data.get("box", []), dtype=np.float32))
    ]

    response = client.infer(model_name=model_name, inputs=inputs)
    masks = response.as_numpy("masks")
    scores = response.as_numpy("scores")
    show_on_image(img, data.get("box"), data.get("points"), data.get("labels"), masks, scores, save_path="result.png")
    print(masks, scores)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="sam-vit-large", help="Name of the model to use for inference.")
    parser.add_argument("--port", default=8001)
    parser.add_argument("--input_file", required=True, help="Path to a JSON file containing segmentation request data.")
    args = parser.parse_args()

    with open(args.input_file, "r") as f:
        data = json.load(f)
    main(args.model_name, args.port, data)
