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
        self.valid_langs = {
            "en", "es", "fr", "de", "it", "pt", "nl", "ru", "zh", "ja", "ko", "ar",
            "hi", "bn", "tr", "sv", "el", "fi", "pl", "vi", "he"
        }
        self.tokenizers = {
            lang: AutoTokenizer.from_pretrained(self.model_dir, src_lang=lang)
            for lang in self.valid_langs
        }


    def execute(self, requests):
        responses = []
        for request in requests:
            inputs = pb_utils.get_input_tensor_by_name(request, "input").as_numpy()
            srcs = pb_utils.get_input_tensor_by_name(request, "src_lang").as_numpy()
            tgts = pb_utils.get_input_tensor_by_name(request, "tgt_lang").as_numpy()

            texts_by_src, tgt_list, errors = {}, [], {}
            for i, (in_data, src, tgt) in enumerate(zip(inputs, srcs, tgts)):
                s, t = src[0].decode(), tgt[0].decode()
                if s not in self.valid_langs:
                    errors[i] = f"ERROR: invalid src lang ({s})"
                    continue
                if t not in self.valid_langs:
                    errors[i] = f"ERROR: invalid tgt lang ({t})"
                    continue
                texts_by_src.setdefault(s, []).append((in_data[0].decode(), i))
                tgt_list.append(t)

            # Tokenize by src_lang and then combine
            tokenized, order_map = [], []
            for src, texts in texts_by_src.items():
                in_data, idxs = zip(*texts)
                toks = self.tokenizers[src](list(in_data), padding="longest", return_tensors="pt").to(0)
                tokenized.append(toks)
                order_map.extend(idxs)
            
            max_len = max(inp['input_ids'].shape[1] for inp in tokenized)
            ids = torch.cat([
                torch.nn.functional.pad(inp['input_ids'], (0, max_len - inp['input_ids'].shape[1]))
                for inp in tokenized
            ])
            masks = torch.cat([
                torch.nn.functional.pad(inp['attention_mask'], (0, max_len - inp['attention_mask'].shape[1]))
                for inp in tokenized
            ])

            bos_ids = [self.tokenizers["en"].get_lang_id(t) for t in tgt_list]

            translated_tokens = self.model.generate(
                input_ids=ids, 
                attention_mask=masks,
                forced_bos_token_id=torch.LongTensor(bos_ids)
            )
            translations = self.tokenizers["en"].batch_decode(translated_tokens, skip_special_tokens=True)
            translations = [x for _, x in sorted(zip(order_map, translations), key=lambda pair: pair[0])]
            for i, err in errors.items():
                translations.insert(i, err)

            out_tensor = pb_utils.Tensor("translation", np.array(translations, dtype=object))
            responses.append(pb_utils.InferenceResponse(output_tensors=[out_tensor]))

        return responses
