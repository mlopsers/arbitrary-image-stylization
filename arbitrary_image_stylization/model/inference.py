import functools
import os

import cv2
import fire
from matplotlib import gridspec
import matplotlib.pylab as plt
import numpy as np
import tensorflow as tf


def crop_center(image):
  """Returns a cropped square image."""
  shape = image.shape
  new_shape = min(shape[1], shape[2])
  offset_y = max(shape[1] - shape[2], 0) // 2
  offset_x = max(shape[2] - shape[1], 0) // 2
  image = tf.image.crop_to_bounding_box(
      image, offset_y, offset_x, new_shape, new_shape)
  return image

@functools.lru_cache(maxsize=None)
def load_image(image_url, image_size=(256, 256), preserve_aspect_ratio=True):
  """Loads and preprocesses images."""
  # Cache image file locally.
  image_path = tf.keras.utils.get_file(os.path.basename(image_url)[-128:], image_url)
  # Load and convert to float32 numpy array, add batch dimension, and normalize to range [0, 1].
  img = plt.imread(image_path).astype(np.float32)[np.newaxis, ...]
  if img.max() > 1.0:
    img = img / 255.

  if len(img.shape) == 3:
    img = tf.stack([img, img, img], axis=-1)

  img = crop_center(img)
  img = tf.image.resize(img, image_size, preserve_aspect_ratio=True)
  return img


def single_image_inference(
    model_checkpoint_dir,
    output_image_path,
    content_image_url='https://biografia24.pl/wp-content/uploads/2013/03/468px-Brad_pitt_2020.jpg',
    style_image_url='https://upload.wikimedia.org/wikipedia/commons/0/0a/The_Great_Wave_off_Kanagawa.jpg',
    output_image_size=2048,
    style_image_size=256,
):
    content_img_size = (output_image_size, output_image_size)
    # The style prediction model was trained with image size 256 and it's the
    # recommended image size for the style image (though, other sizes work as
    # well but will lead to different results).
    style_image_size = (style_image_size, style_image_size)  # Recommended to keep it at 256.

    content_image = load_image(content_image_url, content_img_size)
    style_image = load_image(style_image_url, style_image_size)
    style_image = tf.nn.avg_pool(style_image, ksize=[3,3], strides=[1,1], padding='SAME')

    model = tf.saved_model.load(model_checkpoint_dir)
    outputs = model(tf.constant(content_image), tf.constant(style_image))

    stylized_image = outputs[0][0].numpy()
    stylized_image = (stylized_image * 255).round().astype(np.uint8)
    stylized_image = cv2.cvtColor(stylized_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_image_path, stylized_image)


if __name__ == '__main__':
    fire.Fire(single_image_inference)
