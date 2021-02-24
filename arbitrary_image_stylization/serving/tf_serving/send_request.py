import json

import cv2
import fire
import numpy as np
import requests
import tensorflow as tf

from arbitrary_image_stylization.model.inference import crop_center, load_image

def send_prediction_request(
    output_image_path,
    content_image_url='https://biografia24.pl/wp-content/uploads/2013/03/468px-Brad_pitt_2020.jpg',
    style_image_url='https://upload.wikimedia.org/wikipedia/commons/0/0a/The_Great_Wave_off_Kanagawa.jpg',
    output_image_size=512,
    style_image_size=64,
    serving_url='http://localhost:8501/v1/models/stylization/versions/1:predict',
):

    content_img_size = (output_image_size, output_image_size)
    # The style prediction model was trained with image size 256 and it's the
    # recommended image size for the style image (though, other sizes work as
    # well but will lead to different results).
    style_image_size = (style_image_size, style_image_size)  # Recommended to keep it at 256.

    content_image = load_image(content_image_url, content_img_size)
    style_image = load_image(style_image_url, style_image_size)
    style_image = tf.nn.avg_pool(style_image, ksize=[3,3], strides=[1,1], padding='SAME')

    request_data = json.dumps({
        'inputs': {
            'placeholder': content_image.numpy().tolist(),
            'placeholder_1': style_image.numpy().tolist()
        }
    })

    res = requests.post(serving_url, data=request_data)
    res = np.array(res.json()["outputs"])

    stylized_image = (res[0] * 255).round().astype(np.uint8)
    stylized_image = cv2.cvtColor(stylized_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_image_path, stylized_image)


if __name__ == '__main__':
    fire.Fire(send_prediction_request)
