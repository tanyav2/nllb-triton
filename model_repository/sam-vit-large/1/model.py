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

LOG_LEVEL = logging.DEBUG
logger = logging.getLogger("postprocessing")
logger.setLevel(LOG_LEVEL)
fh = logging.StreamHandler(sys.stdout)
fh.setLevel(LOG_LEVEL)
formatter = logging.Formatter("%(asctime)s\t%(message)s", datefmt="%Y-%m-%d %H:%M:%S")
fh.setFormatter(formatter)
logger.addHandler(fh)


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
            start_time = time.time()
            inputs = self.processor(image, return_tensors="pt").to("cuda")
            image_embeddings = self.model.get_image_embeddings(inputs["pixel_values"])
            end_time = time.time()
            print(f"Image processing and embedding extraction took: {end_time - start_time:.4f} seconds")

            start_time = time.time()
            inputs = self.processor(image, input_points=points, input_boxes=box, input_labels=labels, return_tensors="pt").to("cuda")
            inputs.pop("pixel_values", None)  # pop the pixel_values as they are not needed
            inputs.update({"image_embeddings": image_embeddings})

            with torch.no_grad():
                outputs = self.model(**inputs)
            end_time = time.time()
            print(f"Model input processing and inference took: {end_time - start_time:.4f} seconds")

            start_time = time.time()
            masks = self.processor.image_processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu())
            scores = outputs.iou_scores
            end_time = time.time()
            print(f"Post-processing of outputs took: {end_time - start_time:.4f} seconds")

            out_masks = pb_utils.Tensor("masks", masks[0][0].numpy())
            out_scores = pb_utils.Tensor("scores", scores[0][0].cpu().numpy())
            responses.append(pb_utils.InferenceResponse(output_tensors=[out_masks, out_scores]))

        return responses
    
    @staticmethod
    def _error_response(msg):
        logger.error(msg)
        return [pb_utils.InferenceResponse(error=pb_utils.InferenceError(message=msg))]
