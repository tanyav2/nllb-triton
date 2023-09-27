from transformers import SamModel, SamProcessor
import triton_python_backend_utils as pb_utils
import torch
import numpy as np
from pathlib import Path
import logging
import sys
from PIL import Image
import io

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
            points = pb_utils.get_input_tensor_by_name(request, "points").as_numpy()
            boxes = pb_utils.get_input_tensor_by_name(request, "boxes").as_numpy()
            labels = pb_utils.get_input_tensor_by_name(request, "labels").as_numpy()

            # Ensure at least points or boxes is provided
            if points is None and boxes is None:
                error_message = "Either points or boxes must be provided."
                logger.error(error_message)
                return [pb_utils.InferenceResponse(error=pb_utils.InferenceError(message=error_message))]
            
            # check if label is not none, it is equal to num of points

            image = Image.open(io.BytesIO(image_bytes[0]))
            inputs = self.processor(image, return_tensors="pt").to("cuda")
            image_embeddings = self.model.get_image_embeddings(inputs["pixel_values"])

            inputs = self.processor(image, input_points=points, input_boxes=boxes, input_labels=labels, return_tensors="pt").to("cuda")
            inputs.pop("pixel_values", None) # pop the pixel_values as they are not neded
            inputs.update({"image_embeddings": image_embeddings})

            with torch.no_grad():
                outputs = self.model(**inputs)

            masks = self.processor.image_processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu())
            scores = outputs.iou_scores

            out_masks = pb_utils.Tensor("masks", masks.numpy())
            out_scores = pb_utils.Tensor("scores", scores.numpy())
            responses.append(pb_utils.InferenceResponse(output_tensors=[out_masks, out_scores]))

        return responses
