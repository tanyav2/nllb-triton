# Test script to run the pipeline once it has been loaded in triton
import tritonclient.http as httpclient
import numpy as np
import argparse
from tritonclient import utils

BCP_47_CODES_URL = "https://github.com/facebookresearch/flores/blob/main/flores200/README.md#languages-in-flores-200"

def main(model_name, input_text, tgt_lang, src_lang="eng_Latn"):
    client = httpclient.InferenceServerClient(url="localhost:8000")

    inputs = [
        ("input", input_text),
        ("tgt_lang", tgt_lang),
        ("src_lang", src_lang)
    ]

    # Create tensors for each input
    tensors = []
    for name, input in inputs:
        # arr = utils.serialize_byte_tensor(np.array(list(str)))
        arr = np.array([input.encode()], dtype=np.object_)
        tensor = httpclient.InferInput(name, arr.shape, datatype="BYTES")
        tensor.set_data_from_numpy(arr)
        tensors.append(tensor)

    response = client.infer(model_name=model_name, inputs=tensors)
    print("Translation:", response.as_numpy("translation")[0].decode("utf-8"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--model_name", default="nllb", help="Name of the model to use for inference.")
    parser.add_argument("--input", required=True, help="The text you want to translate.")
    parser.add_argument("--tgt_lang", required=True, help=f"Target language for translation. This argument is always required. Refer to the provided link for available language BCP-47 codes: {BCP_47_CODES_URL}")
    parser.add_argument("--src_lang", default="eng_Latn", help=f"Source language for translation. By default, English (eng_Latn) is set. In order to translate from a different language, specify the BCP-47 code. Refer to the provided link for available BCP-47 codes: {BCP_47_CODES_URL}")
    
    args = parser.parse_args()
    main(model_name=args.model_name, input_text=args.input, tgt_lang=args.tgt_lang, src_lang=args.src_lang)
