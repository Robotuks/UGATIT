import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from tensorflow.contrib import slim
import cv2
import os, random
import numpy as np

class ImageData:

    def __init__(self, load_size, channels, augment_flag):
        self.load_size = load_size
        self.channels = channels
        self.augment_flag = augment_flag

    def image_processing(self, filename):
        x = tf.read_file(filename)
        x_decode = tf.image.decode_jpeg(x, channels=self.channels)
        
        shape = tf.shape(x_decode)
        height = shape[0]
        width = shape[1]

        height_smaller_than_width = tf.less_equal(height, width)
        
        new_height_and_width = tf.cond(
            height_smaller_than_width,
            lambda: (self.load_size, _compute_longer_edge(height, width, self.load_size)),
            lambda: (_compute_longer_edge(width, height, self.load_size), self.load_size)
        )   
        img_resized = tf.image.resize_images(x_decode, new_height_and_width)
        seed = random.randint(0, 2 ** 31 - 1)
        img = tf.image.random_crop(img_resized, [self.load_size, self.load_size, 3], seed = seed)

        # tf.print(height, width, new_height_and_width, tf.shape(img_resized), tf.shape(img))
        # img = tf.image.resize_images(x_decode, [self.load_size, self.load_size],
        #                             preserve_aspect_ratio=True)
        img = tf.cast(img, tf.float32) / 127.5 - 1

        if self.augment_flag :
            augment_size = self.load_size + (30 if self.load_size == 256 else 15)
            p = random.random()
            if p > 0.5:
                img = augmentation(img, augment_size)

        return img

def _compute_longer_edge(height, width, new_shorter_edge):
    return tf.cast(width*new_shorter_edge/height, tf.int32)

def load_test_data(image_path, size=256):
    img = cv2.imread(image_path, flags=cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = cv2.resize(img, dsize=(size, size))

    img = np.expand_dims(img, axis=0)
    img = img/127.5 - 1

    return img

def augmentation(image, augment_size):
    seed = random.randint(0, 2 ** 31 - 1)
    ori_image_shape = tf.shape(image)
    image = tf.image.random_flip_left_right(image, seed=seed)
    image = tf.image.resize_images(image, [augment_size, augment_size])
    image = tf.random_crop(image, ori_image_shape, seed=seed)
    return image

def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)

def inverse_transform(images):
    return ((images+1.) / 2) * 255.0


def imsave(images, size, path):
    images = merge(images, [1, 1])
    images = cv2.cvtColor(images.astype('uint8'), cv2.COLOR_RGB2BGR)
    resized_image = cv2.resize(images,None,fx=size[0],fy=size[1])
    return cv2.imwrite(path, images)

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[h*j:h*(j+1), w*i:w*(i+1), :] = image

    return img

def show_all_variables():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)

def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir

def str2bool(x):
    return x.lower() in ('true')
