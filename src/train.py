import os
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from model import ModelBuilder
from data_preprocessing import Preprocessing
import yaml

class Trainer:
    """
    Trainer class to handle model training, including configuration of callbacks and data generators.

    Attributes:
        config (dict): Configuration dictionary loaded from a YAML file.
        model (tf.keras.Model): Compiled model to be trained.
        callbacks (list): List of Keras callbacks used during training.
        train_generator (tf.data.Dataset): Training data generator.
        val_generator (tf.data.Dataset): Validation data generator.
        test_generator (tf.data.Dataset): Testing data generator.
    """

    def __init__(self, config):
        self.config = config
        self.model_builder = ModelBuilder(self.config['learning_rate'])
        self.model = self.model_builder.model
        self.callbacks = self.configure_callbacks()
        self.train_generator, self.val_generator, self.test_generator = self.configure_data_generators()

    def configure_data_generators(self):
        """
        Configures the data generators for training, validation, and testing.

        Returns:
            tuple: Training, validation, and testing data generators.
        """
        preprocessing = Preprocessing(self.config)
        return preprocessing.data_preprocessing()

    def configure_callbacks(self):
        """
        Configures the callbacks for model training.

        Returns:
            list: List of Keras callback functions.
        """
        model_save_path = self.config['model_save_path']
        log_dir = self.config['log_dir']

        epoch_checkpoint_cb = ModelCheckpoint(f'{model_save_path}/model_epoch_{{epoch:02d}}.keras', 
                                              save_best_only=False, 
                                              monitor='val_loss', 
                                              mode='min', 
                                              verbose=1)

        best_checkpoint_cb = ModelCheckpoint(f'{model_save_path}/best_model.keras', 
                                             save_best_only=True, 
                                             monitor='val_loss', 
                                             mode='min', 
                                             verbose=1)

        early_stopping_cb = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
        reduce_lr_cb = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6, verbose=1)
        tensorboard_cb = TensorBoard(log_dir=log_dir, histogram_freq=1)

        return [epoch_checkpoint_cb, best_checkpoint_cb, early_stopping_cb, reduce_lr_cb, tensorboard_cb]

    def train(self):
        """
        Trains the model using the configured data generators and callbacks.
        """
        history = self.model.fit(
            self.train_generator,
            epochs=self.config['epochs'],
            validation_data=self.val_generator,
            callbacks=self.callbacks
        )

        return history

def main():
    try:
        with open('config.yaml') as f:
            config = yaml.safe_load(f)

        train_model = os.getenv('TRAIN_MODEL', 'true').lower() == 'true'

        if train_model:
            print("Starting training...")
            trainer = Trainer(config)
            trainer.model.summary()
            trainer.train()
        else:
            print("Launching TensorBoard...")
            os.system("tensorboard --logdir {}".format(config['log_dir']))
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    main()
