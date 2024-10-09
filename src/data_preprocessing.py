import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import yaml

class Preprocessing:
    """
    Class for managing data preprocessing and augmentation for training, validation, and testing datasets.

    Attributes:
        train_dir (str): Directory path for the training data.
        val_dir (str): Directory path for the validation data.
        test_dir (str): Directory path for the test data.
        img_size (tuple): Target size for input images.
        batch_size (int): Number of samples per batch.
        augmentation_params (dict): Parameters for data augmentation.
    """

    def __init__(self, config):
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        self.train_dir = os.path.join(base_dir, config['train_dir'])
        self.val_dir = os.path.join(base_dir, config['val_dir'])
        self.test_dir = os.path.join(base_dir, config['test_dir'])
        self.img_size = tuple(config['img_size'])
        self.batch_size = config['batch_size']
        self.augmentation_params = config['augmentation_params']

    def data_preprocessing(self):
        """
        Prepares data generators for training, validation, and testing.

        Returns:
            tuple: Three TensorFlow data generators for training, validation, and testing.
        """
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=self.augmentation_params['rotation_range'],
            width_shift_range=self.augmentation_params['width_shift_range'],
            height_shift_range=self.augmentation_params['height_shift_range'],
            shear_range=self.augmentation_params['shear_range'],
            zoom_range=self.augmentation_params['zoom_range'],
            horizontal_flip=self.augmentation_params['horizontal_flip'],
            fill_mode='nearest'
        )

        train_generator = train_datagen.flow_from_directory(
            self.train_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='binary'
        )
        val_generator = train_datagen.flow_from_directory(
            self.val_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='binary'
        )
        test_generator = train_datagen.flow_from_directory(
            self.test_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='binary'
        )

        return train_generator, val_generator, test_generator
