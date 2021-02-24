import os

import fire
import tensorflow as tf
import tensorflow_hub as hub


def download_model(output_dir='checkpoint'):
    os.makedirs(output_dir, exist_ok=True)

    hub_handle = 'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2'
    hub_module = hub.load(hub_handle)

    signatures = {
        'serving_default': hub_module.signatures['serving_default']
    }
    tf.saved_model.save(hub_module, output_dir, signatures=signatures)


if __name__ == '__main__':
    fire.Fire(download_model)
