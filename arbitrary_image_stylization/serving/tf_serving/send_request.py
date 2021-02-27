import json
import os

import cv2
import fire
import numpy as np
import requests


def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized

def send_simple_prediction_request(
    output_image_path,
    content_image_path='examples/brangelina.jpg',
    style_image_path='examples/picasso.jpg',
    output_image_size=1024,
    style_image_size=256,
    serving_url='http://localhost:8501/v1/models/stylization/versions/1:predict',
):
    content_image = cv2.imread(content_image_path)
    content_image = image_resize(content_image, width=output_image_size)
    content_image = cv2.cvtColor(content_image, cv2.COLOR_RGB2BGR)

    style_image = cv2.imread(style_image_path)
    style_image = cv2.cvtColor(style_image, cv2.COLOR_RGB2BGR)

    request_data = json.dumps({
        'inputs': {
            'images': [content_image.tolist()],
            'styles': [style_image.tolist()],
            'style_image_size': [style_image_size, style_image_size]
        }
    })

    res = requests.post(serving_url, data=request_data, timeout=300)
    if res.status_code != 200:
        print(res.json())
        return

    res = np.array(res.json()["outputs"])
    stylized_image = (res[0] * 255).round().astype(np.uint8)
    stylized_image = cv2.cvtColor(stylized_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_image_path, stylized_image)


if __name__ == '__main__':
    fire.Fire(send_simple_prediction_request)
