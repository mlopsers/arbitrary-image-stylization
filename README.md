# arbitrary-image-stylization

## Quick setup

```bash
pipenv install
pipenv run python3 arbitrary_image_stylization/model/download_model.py \
    arbitrary_image_stylization/model/checkpoint/raw/1/
pipenv run python3 arbitrary_image_stylization/model/inference.py \
    --model-checkpoint-dir arbitrary_image_stylization/model/checkpoint/raw/1/ \
    --output-image-path out.png
```


### Serving

1. Convert model
```bash
pipenv run python3 arbitrary_image_stylization/serving/tf_serving/export_model.py \
    --model-checkpoint-dir=arbitrary_image_stylization/model/checkpoint/raw/1 \
    --model-output-dir=arbitrary_image_stylization/model/checkpoint/tfserving/1
```

2. Run TF Serving server
```bash
./arbitrary_image_stylization/serving/tf_serving/run.bash ./arbitrary-image-stylization/arbitrary_image_stylization/model/checkpoint/tfserving stylization
```
3. Send request
```bash
pipenv run python3 arbitrary_image_stylization/serving/tf_serving/send_request.py \
    --serving_url="http://localhost:8501/v1/models/stylization/versions/1:predict" \
    --output_image_size=1024 \
    --style_image_size=256 \
    --content_image_path=examples/brangelina.jpg \
    --style_image_path=examples/picasso.jpeg \
    --output_image_path=out.png
```
