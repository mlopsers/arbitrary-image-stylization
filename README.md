# arbitrary-image-stylization

## Quick setup

```bash
pipenv install
pipenv run python3 arbitrary_image_stylization/model/download_model.py \
    arbitrary_image_stylization/model/checkpoint
pipenv run python3 arbitrary_image_stylization/model/inference.py \
    --model-checkpoint-dir arbitrary_image_stylization/model/checkpoint \
    --output-image-path out.png
```
