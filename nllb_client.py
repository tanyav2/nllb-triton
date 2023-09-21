import tritonclient.grpc as grpcclient
import numpy as np
import argparse

BCP_47_CODES_URL = "https://github.com/facebookresearch/flores/blob/main/flores200/README.md#languages-in-flores-200"


def main(model_name, port, input_texts, tgt_lang, src_lang="eng_Latn"):
    client = grpcclient.InferenceServerClient(url=f"localhost:{port}")

    input_arr = np.array([text.encode() for text in input_texts], dtype=np.object_)
    tgt_lang_arr = np.array([tgt_lang.encode()], dtype=np.object_)
    src_lang_arr = np.array([src_lang.encode()], dtype=np.object_)

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
        default="nllb_200_distilled_1_3b",
        help="Name of the model to use for inference. Default is 'nllb_200_distilled_1_3b'.",
    )
    parser.add_argument("--port", default=8001)
    parser.add_argument(
        "--input", nargs="+", required=True, help="List of texts you want to translate."
    )
    parser.add_argument(
        "--tgt_lang",
        required=True,
        help=f"Target language for translation. This argument is always required. Refer to the provided link for available language BCP-47 codes: {BCP_47_CODES_URL}",
    )
    parser.add_argument(
        "--src_lang",
        default="eng_Latn",
        help=f"Source language for translation. By default, English (eng_Latn) is set. In order to translate from a different language, specify the BCP-47 code. Refer to the provided link for available BCP-47 codes: {BCP_47_CODES_URL}",
    )

    args = parser.parse_args()
    main(
        model_name=args.model_name,
        port=args.port,
        input_texts=args.input,
        tgt_lang=args.tgt_lang,
        src_lang=args.src_lang,
    )
