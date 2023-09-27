import tritonclient.grpc as grpcclient
import numpy as np
import argparse
import json
import matplotlib.pyplot as plt


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

def show_boxes_on_image(raw_image, boxes):
    plt.figure(figsize=(10,10))
    plt.imshow(raw_image)
    for box in boxes:
      show_box(box, plt.gca())
    plt.axis('on')
    plt.show()

def show_points_on_image(raw_image, input_points, input_labels=None):
    plt.figure(figsize=(10,10))
    plt.imshow(raw_image)
    input_points = np.array(input_points)
    if input_labels is None:
      labels = np.ones_like(input_points[:, 0])
    else:
      labels = np.array(input_labels)
    show_points(input_points, labels, plt.gca())
    plt.axis('on')
    plt.show()

def show_points_and_boxes_on_image(raw_image, boxes, input_points, input_labels=None):
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
    plt.show()


def show_points_and_boxes_on_image(raw_image, boxes, input_points, input_labels=None):
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
    plt.show()


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)


def show_masks_on_image(raw_image, masks, scores):
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
    plt.show()


def main(model_name, port, files):
    client = grpcclient.InferenceServerClient(url=f"localhost:{port}")

    

    input_texts = [item["input_text"] for item in data]
    tgt_langs = [item["tgt_lang"] for item in data]
    src_langs = [item.get("src_lang", "en") for item in data]

    input_arr = np.array([[text.encode()] for text in input_texts], dtype=np.object_)
    tgt_lang_arr = np.array([[l.encode()] for l in tgt_langs], dtype=np.object_)
    src_lang_arr = np.array([[l.encode()] for l in src_langs], dtype=np.object_)

    tensors = [
        grpcclient.InferInput("input", input_arr.shape, datatype="BYTES"),
        grpcclient.InferInput("tgt_lang", tgt_lang_arr.shape, datatype="BYTES"),
        grpcclient.InferInput("src_lang", src_lang_arr.shape, datatype="BYTES"),
    ]

    tensors[0].set_data_from_numpy(input_arr)
    tensors[1].set_data_from_numpy(tgt_lang_arr)
    tensors[2].set_data_from_numpy(src_lang_arr)

    response = client.infer(model_name=model_name, inputs=tensors)
    translation_tensor = response.as_numpy("translation")
    translation = [t.decode() for t in translation_tensor]
    for t in translation:
        print(f"{t}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name",
        default="sam-vit-large",
        help="Name of the model to use for inference. Default is 'sam-vit-large'.",
    )
    parser.add_argument("--port", default=8001)
    parser.add_argument(
        "--files",
        required=True,
        nargs='+',
        help="Paths to the image files to be segmented. Multiple paths can be specified.",
    )
    args = parser.parse_args()
    main(args.model_name, args.port, args.files)
