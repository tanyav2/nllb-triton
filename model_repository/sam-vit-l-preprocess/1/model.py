from transformers import SamModel, SamProcessor
import triton_python_backend_utils as pb_utils
import torch
import numpy as np
from pathlib import Path
import logging
import sys
from PIL import Image
import io
import time
from copy import deepcopy

LOG_LEVEL = logging.DEBUG
logger = logging.getLogger("postprocessing")
logger.setLevel(LOG_LEVEL)
fh = logging.StreamHandler(sys.stdout)
fh.setLevel(LOG_LEVEL)
formatter = logging.Formatter("%(asctime)s\t%(message)s", datefmt="%Y-%m-%d %H:%M:%S")
fh.setFormatter(formatter)
logger.addHandler(fh)

target_length = 1024

def apply_coords(self, coords: np.ndarray, original_size: Tuple[int, ...]) -> np.ndarray:
    """
    Expects a numpy array of length 2 in the final dimension. Requires the
    original image size in (H, W) format.
    """
    old_h, old_w = original_size
    
    # Directly compute the new_h and new_w here
    scale = target_length * 1.0 / max(old_h, old_w)
    new_h, new_w = int(old_h * scale + 0.5), int(old_w * scale + 0.5)

    coords = deepcopy(coords).astype(float)
    coords[..., 0] = coords[..., 0] * (new_w / old_w)
    coords[..., 1] = coords[..., 1] * (new_h / old_h)
    return coords



class TritonPythonModel:
    def initialize(self, args):
        script_dir = Path(__file__).parent
        self.model_dir = script_dir / "huggingface-model-cache"
        self.model = SamModel.from_pretrained(self.model_dir).cuda(0)
        self.processor = SamProcessor.from_pretrained(self.model_dir)


    def execute(self, requests):
        responses = []
        for request in requests:
            image_bytes = pb_utils.get_input_tensor_by_name(request, "image").as_numpy()
            points = pb_utils.get_input_tensor_by_name(request, "points")
            box = pb_utils.get_input_tensor_by_name(request, "box")
            labels = pb_utils.get_input_tensor_by_name(request, "labels")

            if not (points or box):
                return self._error_response("Either points or boxes must be provided.")
            if labels and not points:
                return self._error_response("Labels provided but no points provided.")
            
            points = [points.as_numpy().tolist()] if points else None
            box = [[box.as_numpy().tolist()]] if box else None
            labels = labels.as_numpy().reshape(-1, 1).tolist() if labels else None
            if labels and len(labels) != len(points):
                return self._error_response("Number of labels should be equal to number of points.")

            image = Image.open(io.BytesIO(image_bytes[0]))
            inputs = self.processor(image, return_tensors="pt").to("cuda")
            image_embeddings = self.model.get_image_embeddings(inputs["pixel_values"])

            onnx_coord = np.concatenate([points, np.array([[0.0, 0.0]])], axis=0)[None, :, :]
            onnx_label = np.concatenate([labels, np.array([-1])], axis=0)[None, :].astype(np.float32)
            onnx_coord = apply_coords(onnx_coord, image.shape[:2]).astype(np.float32)
            onnx_mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
            onnx_has_mask_input = np.zeros(1, dtype=np.float32)

            out_image_embeddings = pb_utils.Tensor("image_embeddings", image_embeddings.cpu().numpy())
            out_onnx_coord = pb_utils.Tensor("onnx_coord", onnx_coord)
            out_onnx_label = pb_utils.Tensor("onnx_label", onnx_label)
            out_onnx_mask_input = pb_utils.Tensor("onnx_mask_input", onnx_mask_input)
            out_onnx_has_mask_input = pb_utils.Tensor("onnx_has_mask_input", onnx_has_mask_input)
            out_orig_im_size = pb_utils.Tensor("orig_im_size", np.array(image.shape[:2], dtype=np.float32))

            responses.append(pb_utils.InferenceResponse(output_tensors=[out_image_embeddings, out_onnx_coord, out_onnx_label, out_onnx_mask_input, out_onnx_has_mask_input, out_orig_im_size]))

        return responses
    
    @staticmethod
    def _error_response(msg):
        logger.error(msg)
        return [pb_utils.InferenceResponse(error=pb_utils.InferenceError(message=msg))]
