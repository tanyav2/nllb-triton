from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer
)
import triton_python_backend_utils as pb_utils
import numpy as np


class TritonPythonModel:
    def initialize(self, args):
        # by default, set tokenizer to translate from english
        # can be changed on request
        # self.tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-1.3B")
        self.model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-1.3B", device_map="auto", load_in_8bit=True)

    def execute(self, requests):
        responses = []
        for request in requests:
            input_np = pb_utils.get_input_tensor_by_name(request, "input")
            input = pb_utils.deserialize_bytes_tensor(input_np)
            dst_lang_np = pb_utils.get_input_tensor_by_name(request, "dst_lang")
            src_lang_np = pb_utils.get_input_tensor_by_name(request, "src_lang")
            dst_lang = pb_utils.deserialize_bytes_tensor(dst_lang_np)
            src_lang = pb_utils.deserialize_bytes_tensor(src_lang_np)
            input_str = ''.join(input)
            dst_lang_str = ''.join(dst_lang)
            src_lang_str = ''.join(src_lang)
            self.tokenizer = AutoTokenizer.from_pretrained(
                    "facebook/nllb-200-1.3B",
                    src_lang=src_lang_str
            )
            inputs = self.tokenizer(input_str, padding=True, truncation=True, return_tensors="pt").to("cuda")

            # Inference
            translated_tokens = self.model.generate(
                **inputs, forced_bos_token_id=self.tokenizer.lang_code_to_id[dst_lang_str]
            )
            translation = self.tokenizer.batch_decode(
                translated_tokens, skip_special_tokens=True
            )[0]

            # Postprocessing
            output_tensor = pb_utils.Tensor(
                "translation", np.array([translation], dtype=object)
            )

            # Create inference response
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[output_tensor]
            )
            responses.append(inference_response)
        return responses