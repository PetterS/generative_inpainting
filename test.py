import argparse
import sys

import cv2
import numpy as np
import tensorflow as tf
import neuralgym as ng

from inpaint_model import InpaintCAModel


parser = argparse.ArgumentParser()
parser.add_argument('images', nargs="+", type=str,
                    help='The filename of image to be completed.')
parser.add_argument('--checkpoint_dir', default='', type=str,
                    help='The directory of tensorflow checkpoint.')


def process_image(model, name: str):
    image = cv2.imread(name, cv2.IMREAD_UNCHANGED)
    alpha = image[:, :, 3].copy()
    image = image[:, :, 0:3].copy()

    bool_mask = alpha == 0
    mask = np.zeros(image.shape, dtype=np.uint8)
    mask[bool_mask, 0] = 255
    mask[bool_mask, 1] = 255
    mask[bool_mask, 2] = 255

    print("Image", image.shape, image.dtype)

    assert image.shape == mask.shape

    h, w, _ = image.shape
    grid = 8
    image = image[:h//grid*grid, :w//grid*grid, :]
    mask = mask[:h//grid*grid, :w//grid*grid, :]

    image = np.expand_dims(image, 0)
    mask = np.expand_dims(mask, 0)
    input_image = np.concatenate([image, mask], axis=2)

    print("After modification:", image.shape, image.dtype)

    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    with tf.Session(config=sess_config) as sess:
        input_image = tf.constant(input_image, dtype=tf.float32)
        output = model.build_server_graph(FLAGS, input_image, reuse=model.petter_reuse)
        model.petter_reuse = True
        output = (output + 1.) * 127.5
        output = tf.reverse(output, [-1])
        output = tf.saturate_cast(output, tf.uint8)
        # load pretrained model
        vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        assign_ops = []
        for var in vars_list:
            vname = var.name
            from_name = vname
            var_value = tf.contrib.framework.load_variable(args.checkpoint_dir, from_name)
            assign_ops.append(tf.assign(var, var_value))
        sess.run(assign_ops)
        print('Model loaded.')
        result = sess.run(output)
        cv2.imwrite(name + "_output.png", result[0][:, :, ::-1])


if __name__ == "__main__":
    FLAGS = ng.Config('inpaint.yml')
    # ng.get_gpus(1)
    args, unknown = parser.parse_known_args()
    model = InpaintCAModel()
    model.petter_reuse = False

    for name in args.images:
        process_image(model, name)
