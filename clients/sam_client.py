import tritonclient.grpc as grpcclient
import numpy as np
import argparse
import json
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
matplotlib.use('Agg')


# Utility functions for visualization
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))  


def show_boxes_on_image(raw_image, boxes, save_path=None):
    plt.figure(figsize=(10,10))
    plt.imshow(raw_image)
    for box in boxes:
      show_box(box, plt.gca())
    plt.axis('on')
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
        plt.close()
    else:
        plt.show()


def show_points_on_image(raw_image, input_points, input_labels=None, save_path=None):
    plt.figure(figsize=(10,10))
    plt.imshow(raw_image)
    input_points = np.array(input_points)
    if input_labels is None:
      labels = np.ones_like(input_points[:, 0])
    else:
      labels = np.array(input_labels)
    show_points(input_points, labels, plt.gca())
    plt.axis('on')
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
        plt.close()
    else:
        plt.show()


def show_points_and_boxes_on_image(raw_image, boxes, input_points, input_labels=None, save_path=None):
    plt.figure(figsize=(10,10))
    plt.imshow(raw_image)
    input_points = np.array(input_points)
    if input_labels is None:
      labels = np.ones_like(input_points[:, 0])
    else:
      labels = np.array(input_labels)
    show_points(input_points, labels, plt.gca())
    for box in boxes:
      show_box(box, plt.gca())
    plt.axis('on')
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
        plt.close()
    else:
        plt.show()


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)


def show_masks_on_image(raw_image, masks, scores, save_path=None):
    if len(masks.shape) == 4:
      masks = masks.squeeze()
    if scores.shape[0] == 1:
      scores = scores.squeeze()

    nb_predictions = scores.shape[-1]
    fig, axes = plt.subplots(1, nb_predictions, figsize=(15, 15))
    for i, (mask, score) in enumerate(zip(masks, scores)):
      mask = mask.cpu().detach()
      axes[i].imshow(np.array(raw_image))
      show_mask(mask, axes[i])
      axes[i].title.set_text(f"Mask {i+1}, Score: {score.item():.3f}")
      axes[i].axis("off")
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
        plt.close()
    else:
        plt.show()


def main(model_name, port, data):
    client = grpcclient.InferenceServerClient(url=f"localhost:{port}")
    img_file = data["image_file"]
    img = Image.open(img_file)
    with open(img_file, "rb") as f:
        img_buf = f.read()

    img_arr = np.array([img_buf], dtype=np.object_)
    points = np.array(data["points"], dtype=np.float32)
    labels = np.array(data["labels"], dtype=np.int32)
    box = np.array(data["box"], dtype=np.float32)
    show_points_and_boxes_on_image(img, [box], points, labels, save_path="points_and_boxes.png")

    image_t = grpcclient.InferInput("images", img_arr.shape, datatype="BYTES")
    image_t.set_data_from_numpy(img_arr)

    points_t = grpcclient.InferInput("points", points.shape, datatype="FP32")
    points_t.set_data_from_numpy(points)

    labels_t = grpcclient.InferInput("labels", labels.shape, datatype="INT32")
    labels_t.set_data_from_numpy(labels)
  
    box_t = grpcclient.InferInput("box", box.shape, datatype="FP32")
    box_t.set_data_from_numpy(box)

    response = client.infer(model_name=model_name, inputs=[image_t, points_t, labels_t, box_t])
    masks = response.as_numpy("masks")
    scores = response.as_numpy("scores")
    show_masks_on_image(img, masks, scores, save_path="masks.png")
    print(masks)
    print(scores)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name",
        default="sam-vit-large",
        help="Name of the model to use for inference. Default is 'sam-vit-large'.",
    )
    parser.add_argument("--port", default=8001)
    parser.add_argument(
        "--input_file",
        required=True,
        help=(
            "Path to a JSON file containing a list of segmentation requests. Each request should be an object with fields: "
            "'image_file', 'points', 'labels' and 'boxes'."
        ),
    )
    args = parser.parse_args()
    with open(args.input_file, "r") as f:
        data = json.load(f)
    main(args.model_name, args.port, data)
