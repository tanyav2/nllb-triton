import tritonclient.grpc as grpcclient
import numpy as np
import argparse
import json


BCP_47_CODES_URL = "https://github.com/facebookresearch/flores/blob/main/flores200/README.md#languages-in-flores-200"


def main(model_name, port, data):
    client = grpcclient.InferenceServerClient(url=f"localhost:{port}")

    input_texts = [item["input_text"] for item in data]
    tgt_langs = [item["tgt_lang"] for item in data]
    src_langs = [item.get("src_lang", "en") for item in data]

    input_arr = np.array([[text.encode()] for text in input_texts], dtype=np.object_)
    tgt_lang_arr = np.array([[l.encode()] for l in tgt_langs], dtype=np.object_)
    src_lang_arr = np.array([[l.encode()] for l in src_langs], dtype=np.object_)

    tensors = [
        grpcclient.InferInput("input", input_arr.shape, datatype="BYTES"),
        grpcclient.InferInput("tgt_lang", tgt_lang_arr.shape, datatype="BYTES"),
        grpcclient.InferInput("src_lang", src_lang_arr.shape, datatype="BYTES"),
    ]

    tensors[0].set_data_from_numpy(input_arr)
    tensors[1].set_data_from_numpy(tgt_lang_arr)
    tensors[2].set_data_from_numpy(src_lang_arr)

    response = client.infer(model_name=model_name, inputs=tensors)
    translation_tensor = response.as_numpy("translation")
    translation = [t.decode() for t in translation_tensor]
    for t in translation:
        print(f"{t}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name",
        default="m2m100-1.2b",
        help="Name of the model to use for inference. Default is 'm2m100-1.2b'.",
    )
    parser.add_argument("--port", default=8001)
    parser.add_argument(
        "--input_file",
        required=True,
        help=(
            "Path to a JSON file containing a list of translation requests. Each request should be an object with fields: "
            "'input_text' and 'tgt_lang'. The optional 'src_lang' field indicates the source language. If not specified, "
            "'eng_Latn' (English) will be used as the default source language. "
            "The 'tgt_lang' field specifies the target language for translation. "
            f"Refer to the provided link for available language BCP-47 codes: {BCP_47_CODES_URL}. "
            "Example JSON structure: [{'input_text': 'Hello', 'tgt_lang': 'fr_Latn'}, ...]."
        ),
    )

    args = parser.parse_args()

    with open(args.input_file, "r") as f:
        data = json.load(f)

    main(model_name=args.model_name, port=args.port, data=data)
