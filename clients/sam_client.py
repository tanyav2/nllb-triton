import argparse
import json
import numpy as np
import tritonclient.grpc as grpcclient
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')


def show_mask(mask, ax, color=np.array([30/255, 144/255, 255/255, 0.6])):
    mask_image = mask.reshape(*mask.shape[-2:], 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax):
    x0, y0, x1, y1 = box
    ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)


def show_on_image(raw_image, box=None, points=None, labels=None, save_path=None):
    plt.figure(figsize=(10, 10))
    plt.imshow(raw_image)

    if box:
        show_box(box, plt.gca())
    if points:
        np_points = np.array(points)
        labels_array = np.array(labels) if labels else np.ones_like(np_points[:, 0])
        show_points(np_points, labels_array, plt.gca())

    plt.axis('on')
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
        plt.close()
    else:
        plt.show()


def show_masks_on_image(raw_image, masks, scores, save_path=None):
    _, axes = plt.subplots(1, len(scores), figsize=(15, 15))
    for i, (mask, score) in enumerate(zip(masks, scores)):
        axes[i].imshow(np.array(raw_image))
        show_mask(mask, axes[i])
        axes[i].title.set_text(f"Mask {i+1}, Score: {score:.3f}")
        axes[i].axis("off")

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
        plt.close()
    else:
        plt.show()


def main(model_name, port, data):
    client = grpcclient.InferenceServerClient(url=f"localhost:{port}")
    img = Image.open(data["image_file"])
    with open(data["image_file"], "rb") as f:
        img_buf = f.read()

    img_arr = np.array([img_buf])
    inputs = [grpcclient.InferInput("image", img_arr.shape, datatype="BYTES")]
    inputs[0].set_data_from_numpy(img_arr)

    points, labels, box = None, None, None
    if "points" in data:
        points = np.array(data["points"], dtype=np.float32)
        points_t = grpcclient.InferInput("points", points.shape, datatype="FP32")
        points_t.set_data_from_numpy(points)
        inputs.append(points_t)

    if "labels" in data:
        labels = np.array(data["labels"], dtype=np.int32)
        labels_t = grpcclient.InferInput("labels", labels.shape, datatype="INT32")
        labels_t.set_data_from_numpy(labels)
        inputs.append(labels_t)

    if "box" in data:
        box = np.array(data["box"], dtype=np.float32)
        box_t = grpcclient.InferInput("box", box.shape, datatype="FP32")
        box_t.set_data_from_numpy(box)
        inputs.append(box_t)

    # Visualization
    save_path = {"box": "box.png", "points": "points.png", "both": "points_and_box.png"}.get(
        "both" if "box" in data and "points" in data else ("box" if "box" in data else "points"), None)
    show_on_image(img, data.get("box"), data.get("points"), data.get("labels"), save_path)

    response = client.infer(model_name=model_name, inputs=inputs)
    masks = response.as_numpy("masks")
    scores = response.as_numpy("scores")
    show_masks_on_image(img, masks, scores, save_path="masks.png")
    print(masks)
    print(scores)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="sam-vit-large", help="Name of the model to use for inference.")
    parser.add_argument("--port", default=8001)
    parser.add_argument("--input_file", required=True, help="Path to JSON file with 'image_file', 'points', 'labels' and 'box'.")
    args = parser.parse_args()
    with open(args.input_file, "r") as f:
        data = json.load(f)
    main(args.model_name, args.port, data)
