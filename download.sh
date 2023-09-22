downloads="./model_repository/nllb_200_distilled_1_3b/1/huggingface-model-cache/pytorch_model.bin.download"

for filepath in $downloads; do
    filename=$(basename $filepath)
    download_filename=${filename:0:-9}
    output_path="$(dirname $filepath)/$download_filename"
    if [ ! -e "$output_path" ]
    then
        curl --create-dirs -Lo "$output_path" $(cat $filepath | head -n 1)
    fi
    rm $filepath
done
