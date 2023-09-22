from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers.generation.logits_process import ForcedBOSTokenLogitsProcessor
import triton_python_backend_utils as pb_utils
import torch
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

BEAM_SIZE = 5 # taken from config.json

def force_bos_tokens_logits_processor_dropin___call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
    cur_len = input_ids.shape[-1]
    if cur_len == 1:
        num_tokens = scores.shape[1]
        bos_token_id_replicated = self.bos_token_id.repeat_interleave(BEAM_SIZE)
        bos_tokens = torch.arange(num_tokens)[None].repeat(scores.shape[0], 1) == bos_token_id_replicated[:,None]
        scores[~bos_tokens] = -float("inf")
        scores[bos_tokens] = 0
    return scores

ForcedBOSTokenLogitsProcessor.__call__ = force_bos_tokens_logits_processor_dropin___call__


class TritonPythonModel:
    def initialize(self, args):
        script_dir = Path(__file__).parent
        self.model_dir = script_dir / "huggingface-model-cache"
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_dir).cuda(0)

        # Dict to store tokenizers for different source languages. English tokenizer is preloaded.
        self.tokenizers = {"en": AutoTokenizer.from_pretrained(self.model_dir, src_lang="en")}
        self.valid_lang_codes = {'ja', 'oc', 'jv', 'ig', 'uz', 'vi', 'is', 'sv', 'ba', 'fa', 'ar', 'ast', 'id', 'post 1500', 'ne', 'ff', 'af', 'gl', 'mr', 'pa', 'lb', 'fy', 'hu', 'kn', 'my', 'ceb', 'pl', 'bg', 'ps', 'si', 'ms', 'ns', 'hy', 'am', 'ru', 'ta', 'lo', 'sq', 'sl', 'tl', 'cy', 'nl', 'ml', 'mg', 'so', 'mn', 'sr', 'sk', 'en', 'sw', 'uk', 'hi', 'el', 'zh', 'tr', 'it', 'ln', 'lt', 'ka', 'no', 'he', 'sd', 'su', 'or', 'km', 'ro', 'ha', 'zu', 'gu', 'kk', 'cs', 'bn', 'ga', 'pt', 'lg', 'ur', 'br', 'bs', 'da', 'ko', 'yi', 'hr', 'fi', 'az', 'ilo', 'et', 'ht', 'ca', 'ss', 'mk', 'th', 'xh', 'yo', 'de', 'fr', 'lv', 'gd', 'es', 'tn', 'be', 'wo'}


    def execute(self, requests):
        responses = []
        for request in requests:
            input_batch = pb_utils.get_input_tensor_by_name(request, "input").as_numpy()
            tgt_langs = pb_utils.get_input_tensor_by_name(
                request, "tgt_lang"
            ).as_numpy()
            src_langs = pb_utils.get_input_tensor_by_name(
                request, "src_lang"
            ).as_numpy()

            input_texts_by_src = {}
            tgt_langs_list = []
            for input_bytes, tgt_lang_bytes, src_lang_bytes in zip(
                input_batch, tgt_langs, src_langs
            ):
                input_text = input_bytes[0].decode()
                tgt_lang = tgt_lang_bytes[0].decode()
                src_lang = src_lang_bytes[0].decode()

                if src_lang not in self.valid_lang_codes:
                    logger.error(f"Invalid src_lang: {src_lang}. Skipping translation.")
                    continue

                if tgt_lang not in self.valid_lang_codes:
                    logger.error(f"Invalid tgt_lang: {tgt_lang}. Skipping translation.")
                    continue

                if src_lang not in input_texts_by_src:
                    input_texts_by_src[src_lang] = []

                input_texts_by_src[src_lang].append(input_text)
                tgt_langs_list.append(tgt_lang)

            # Tokenize by src_lang and then combine
            tokenized_inputs = []
            for src_lang, input_texts in input_texts_by_src.items():
                if src_lang not in self.tokenizers:
                    self.tokenizers[src_lang] = AutoTokenizer.from_pretrained(
                        self.model_dir, src_lang=src_lang
                    )

                tokenizer = self.tokenizers[src_lang]
                inputs = tokenizer(
                    input_texts, padding="longest", return_tensors="pt"
                ).to(0)
                tokenized_inputs.append(inputs)

            # Find max length and pad accordingly
            max_length = max([i['input_ids'].shape[1] for i in tokenized_inputs])
            input_ids, attention_mask = zip(*[
                (
                    torch.cat(
                        [i['input_ids'], torch.zeros(i['input_ids'].shape[0], max_length - i['input_ids'].shape[1]).type(torch.long).to(0)],
                        dim=1
                    ),
                    torch.cat(
                        [i['attention_mask'], torch.zeros(i['input_ids'].shape[0], max_length - i['input_ids'].shape[1]).type(torch.long).to(0)],
                        dim=1
                    ),
                )
                for i in tokenized_inputs
            ])
            input_ids, attention_mask = torch.cat(input_ids, dim=0), torch.cat(attention_mask, dim=0)

            # assuming default tokenizer for getting correct lang id
            bos_token_ids = [self.tokenizers["en"].get_lang_id(tgt_lang) for tgt_lang in tgt_langs_list]

            translated_tokens = self.model.generate(
                input_ids=input_ids, 
                attention_mask=attention_mask,
                forced_bos_token_id=torch.LongTensor(bos_token_ids)
            )

            # assuming default tokenizer for decoding
            translations = self.tokenizers["en"].batch_decode(translated_tokens, skip_special_tokens=True)

            output_tensor = pb_utils.Tensor(
                "translation", np.array(translations, dtype=object)
            )
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[output_tensor]
            )
            responses.append(inference_response)

        return responses
