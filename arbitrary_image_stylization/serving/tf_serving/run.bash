MODEL_PATH=$1
MODEL_NAME=$2

MODEL_PATH="$(realpath ${MODEL_PATH})"
docker run -it --rm -p 8501:8501 \
    -v "${MODEL_PATH}:/models/${MODEL_NAME}" \
    -e MODEL_NAME=${MODEL_NAME} \
    -e TF_CPP_MIN_VLOG_LEVEL=1 \
    tensorflow/serving
