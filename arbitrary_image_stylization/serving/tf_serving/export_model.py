import os

import fire
import tensorflow as tf


class ServingModel(tf.keras.Model):
    def __init__(self, model):
        super().__init__(self)
        self.model = model

    @tf.function(input_signature=[
        tf.TensorSpec([None, None, None, 3], dtype=tf.uint8, name='images'),
        tf.TensorSpec([None, None, None, 3], dtype=tf.uint8, name='styles'),
        tf.TensorSpec([2], dtype=tf.int32, name='style_image_size')
    ])
    def serve_images(self, images, styles, style_image_size):
        images = tf.cast(images, tf.float32) / 255
        styles = tf.cast(styles, tf.float32) / 255
        styles = tf.nn.avg_pool(styles, ksize=[3,3], strides=[1,1], padding='SAME')

        styles = tf.image.resize(styles, style_image_size,
            preserve_aspect_ratio=True)

        styled = self.model(images, styles)
        return {
            "outputs": styled[0]
        }


def export_model(
    model_checkpoint_dir='arbitrary_image_stylization/model/checkpoint/2',
    model_output_dir='arbitrary_image_stylization/model/checkpoint/4'
):
    os.makedirs(model_output_dir, exist_ok=True)
    model = tf.saved_model.load(model_checkpoint_dir)
    serving_model = ServingModel(model)
    tf.saved_model.save(serving_model, model_output_dir, signatures={'serving_default': serving_model.serve_images})


if __name__ == '__main__':
    fire.Fire(export_model)
