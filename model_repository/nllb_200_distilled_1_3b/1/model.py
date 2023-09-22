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

def force_bos_tokens_logits_processor_dropin___call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
    cur_len = input_ids.shape[-1]
    if cur_len == 1:
        num_tokens = scores.shape[1]
        bos_tokens = torch.arange(num_tokens)[None].repeat(scores.shape[0], 1) == self.bos_token_id[:,None]
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
        self.tokenizers = {"eng_Latn": AutoTokenizer.from_pretrained(self.model_dir)}
        self.valid_bcp_47_codes = {
            "ltz_Latn",
            "ilo_Latn",
            "zho_Hans",
            "bod_Tibt",
            "tat_Cyrl",
            "tur_Latn",
            "mar_Deva",
            "tel_Telu",
            "bug_Latn",
            "ckb_Arab",
            "mkd_Cyrl",
            "pag_Latn",
            "yue_Hant",
            "kas_Deva",
            "gle_Latn",
            "nno_Latn",
            "arb_Arab",
            "arb_Latn",
            "ces_Latn",
            "ell_Grek",
            "nus_Latn",
            "azb_Arab",
            "nso_Latn",
            "nld_Latn",
            "grn_Latn",
            "kon_Latn",
            "mos_Latn",
            "som_Latn",
            "vie_Latn",
            "kor_Hang",
            "twi_Latn",
            "ben_Beng",
            "ceb_Latn",
            "aeb_Arab",
            "sin_Sinh",
            "kat_Geor",
            "npi_Deva",
            "eus_Latn",
            "sag_Latn",
            "kmb_Latn",
            "vec_Latn",
            "min_Arab",
            "xho_Latn",
            "bul_Cyrl",
            "tpi_Latn",
            "ars_Arab",
            "bel_Cyrl",
            "bjn_Arab",
            "khk_Cyrl",
            "run_Latn",
            "ayr_Latn",
            "taq_Latn",
            "ukr_Cyrl",
            "wol_Latn",
            "cym_Latn",
            "knc_Arab",
            "kam_Latn",
            "bos_Latn",
            "swh_Latn",
            "tso_Latn",
            "lim_Latn",
            "amh_Ethi",
            "tgl_Latn",
            "lmo_Latn",
            "lus_Latn",
            "ast_Latn",
            "tuk_Latn",
            "apc_Arab",
            "acm_Arab",
            "bak_Cyrl",
            "bam_Latn",
            "fra_Latn",
            "urd_Arab",
            "est_Latn",
            "deu_Latn",
            "afr_Latn",
            "fon_Latn",
            "kab_Latn",
            "kas_Arab",
            "uzn_Latn",
            "oci_Latn",
            "hrv_Latn",
            "mlt_Latn",
            "pan_Guru",
            "fuv_Latn",
            "glg_Latn",
            "tha_Thai",
            "slk_Latn",
            "jav_Latn",
            "als_Latn",
            "uig_Arab",
            "acq_Arab",
            "cat_Latn",
            "gaz_Latn",
            "tum_Latn",
            "slv_Latn",
            "ron_Latn",
            "asm_Beng",
            "spa_Latn",
            "azj_Latn",
            "mal_Mlym",
            "war_Latn",
            "kac_Latn",
            "mni_Beng",
            "plt_Latn",
            "tzm_Tfng",
            "epo_Latn",
            "crh_Latn",
            "ajp_Arab",
            "por_Latn",
            "szl_Latn",
            "ibo_Latn",
            "ace_Arab",
            "sat_Olck",
            "hin_Deva",
            "fij_Latn",
            "ind_Latn",
            "jpn_Jpan",
            "tir_Ethi",
            "arz_Arab",
            "kin_Latn",
            "hye_Armn",
            "ssw_Latn",
            "quy_Latn",
            "kmr_Latn",
            "mai_Deva",
            "ary_Arab",
            "rus_Cyrl",
            "kir_Cyrl",
            "mag_Deva",
            "nob_Latn",
            "snd_Arab",
            "guj_Gujr",
            "pbt_Arab",
            "eng_Latn",
            "dan_Latn",
            "khm_Khmr",
            "dik_Latn",
            "lin_Latn",
            "sot_Latn",
            "ace_Latn",
            "knc_Latn",
            "awa_Deva",
            "min_Latn",
            "shn_Mymr",
            "lit_Latn",
            "scn_Latn",
            "lij_Latn",
            "kea_Latn",
            "ewe_Latn",
            "nya_Latn",
            "dzo_Tibt",
            "prs_Arab",
            "zul_Latn",
            "cjk_Latn",
            "kaz_Cyrl",
            "yor_Latn",
            "ita_Latn",
            "hau_Latn",
            "pap_Latn",
            "ory_Orya",
            "bjn_Latn",
            "ydd_Hebr",
            "san_Deva",
            "tsn_Latn",
            "dyu_Latn",
            "sna_Latn",
            "isl_Latn",
            "pes_Arab",
            "fin_Latn",
            "smo_Latn",
            "lug_Latn",
            "tgk_Cyrl",
            "sun_Latn",
            "bem_Latn",
            "tam_Taml",
            "umb_Latn",
            "gla_Latn",
            "heb_Hebr",
            "luo_Latn",
            "ban_Latn",
            "srp_Cyrl",
            "zsm_Latn",
            "mya_Mymr",
            "zho_Hant",
            "aka_Latn",
            "hat_Latn",
            "lua_Latn",
            "hne_Deva",
            "lao_Laoo",
            "pol_Latn",
            "srd_Latn",
            "fao_Latn",
            "kik_Latn",
            "taq_Tfng",
            "ltg_Latn",
            "mri_Latn",
            "fur_Latn",
            "kan_Knda",
            "kbp_Latn",
            "bho_Deva",
            "lvs_Latn",
            "swe_Latn",
            "hun_Latn",
        }

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

                if src_lang not in self.valid_bcp_47_codes:
                    logger.error(f"Invalid src_lang: {src_lang}. Skipping translation.")
                    continue

                if tgt_lang not in self.valid_bcp_47_codes:
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
            bos_token_ids = [self.tokenizers["eng_Latn"].lang_code_to_id[tgt_lang] for tgt_lang in tgt_langs_list]

            translated_tokens = self.model.generate(
                input_ids=input_ids, 
                attention_mask=attention_mask,
                forced_bos_token_id=torch.LongTensor(bos_token_ids)
            )

            # assuming default tokenizer for decoding
            translations = self.tokenizers["eng_Latn"].batch_decode(translated_tokens, skip_special_tokens=True)

            output_tensor = pb_utils.Tensor(
                "translation", np.array(translations, dtype=object)
            )
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[output_tensor]
            )
            responses.append(inference_response)

        return responses
