from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import triton_python_backend_utils as pb_utils
import numpy as np
from pathlib import Path
import logging
import sys

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
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_dir).cuda(0)

        # Dict to store tokenizers for different source languages. English tokenizer is preloaded.
        self.tokenizers = {"eng_Latn": AutoTokenizer.from_pretrained(self.model_dir)}

    def execute(self, requests):
        responses = []
        for request in requests:
            input_bytes_batch = pb_utils.get_input_tensor_by_name(
                request, "input"
            ).as_numpy()
            tgt_lang_bytes = pb_utils.get_input_tensor_by_name(
                request, "tgt_lang"
            ).as_numpy()
            src_lang_bytes = pb_utils.get_input_tensor_by_name(
                request, "src_lang"
            ).as_numpy()
            input_texts = [input_bytes.decode() for input_bytes in input_bytes_batch]
            tgt_lang = tgt_lang_bytes[0].decode()
            src_lang = src_lang_bytes[0].decode()

            # Check if we have the tokenizer for the current source language
            if src_lang not in self.tokenizers:
                self.tokenizers[src_lang] = AutoTokenizer.from_pretrained(
                    self.model_dir, src_lang=src_lang
                )

            tokenizer = self.tokenizers[src_lang]
            inputs = tokenizer(
                input_texts, padding=True, truncation=True, return_tensors="pt"
            ).to(0)

            translated_tokens = self.model.generate(
                **inputs, forced_bos_token_id=tokenizer.lang_code_to_id[tgt_lang]
            )
            translation = tokenizer.batch_decode(
                translated_tokens, skip_special_tokens=True
            )

            output_tensor = pb_utils.Tensor(
                "translation", np.array(translation, dtype=object)
            )
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[output_tensor]
            )
            responses.append(inference_response)

        return responses
