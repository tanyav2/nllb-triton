from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer
)
import triton_python_backend_utils as pb_utils
import numpy as np
from pathlib import Path


class TritonPythonModel:
    def initialize(self, args):
        self.model_dir = Path(__name__).with_name("huggingface-model-cache")
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_dir)
        
        # Dict to store tokenizers for different source languages. English tokenizer is preloaded.
        self.tokenizers = {"eng_Latn": AutoTokenizer.from_pretrained(self.model_dir)}

    def execute(self, requests):
        responses = []
        for request in requests:
            input_bytes = pb_utils.get_input_tensor_by_name(request, "input")
            tgt_lang_bytes = pb_utils.get_input_tensor_by_name(request, "tgt_lang")
            src_lang_bytes = pb_utils.get_input_tensor_by_name(request, "src_lang")
            input_str = input_bytes.as_numpy()[0].decode()
            tgt_lang = tgt_lang_bytes.as_numpy()[0].decode()
            src_lang = src_lang_bytes.as_numpy()[0].decode()
            
            # Check if we have the tokenizer for the current source language
            if src_lang not in self.tokenizers:
                self.tokenizers[src_lang] = AutoTokenizer.from_pretrained(
                    self.model_dir,
                    src_lang=src_lang
                )

            tokenizer = self.tokenizers[src_lang]
            inputs = tokenizer(input_str, padding=True, truncation=True, return_tensors="pt")

            translated_tokens = self.model.generate(
                **inputs, forced_bos_token_id=tokenizer.lang_code_to_id[tgt_lang]
            )
            translation = tokenizer.batch_decode(
                translated_tokens, skip_special_tokens=True
            )[0]

            output_tensor = pb_utils.Tensor(
                "translation", np.array([translation], dtype=object)
            )
            inference_response = pb_utils.InferenceResponse(output_tensors=[output_tensor])
            responses.append(inference_response)

        return responses