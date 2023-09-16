# Test script to run the pipeline once it has been loaded in triton
import tritonclient.http as httpclient
import numpy as np
import argparse


def main(model_name, dst_lang, src_lang="eng_Latn"):
    # Setting up client
    client = httpclient.InferenceServerClient(url="localhost:8000")

    # Create input tensor
    inputs = ["Back in the spring of 2018, I sat at the Cancún airport, almost in a state of paralysis. I was staring at the planes as my thoughts raced back to what I had just witnessed. Could it possibly be true?",
              dst_lang,
              src_lang]
    input_arr = [np.array(list(s)) for s in inputs]

    input_tensor = httpclient.InferInput("text", input_arr[0].shape, datatype="STRING")
    input_tensor.set_data_from_numpy(input_arr[0])
    dst_tensor = httpclient.InferInput("dst_lang", input_arr[1].shape, datatype="STRING")
    dst_tensor.set_data_from_numpy(input_arr[1])
    src_tensor = httpclient.InferInput("src_lang", input_arr[2].shape, datatype="STRING")
    src_tensor.set_data_from_numpy(input_arr[2])

    # Query the server
    response = client.infer(model_name=model_name, inputs=[input_tensor, dst_tensor, src_tensor])

    # Get the output tensor
    output_tensor = response.as_numpy("translation")

    print("Translation:", output_tensor[0].decode("utf-8"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", default="nllb"
    )
    parser.add_argument(
        "--src_lang", default="eng_Latn"
    )
    parser.add_argument(
        "--dst_lang", required=True
    )
    args = parser.parse_args()
    main(args.model_name, args.dst_lang, src_lang=args.src_lang)