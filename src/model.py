from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import Adam

class ModelBuilder:
    """
    Class for building the deep learning model architecture.

    Attributes:
        learning_rate (float): Learning rate for the optimizer.
        model (tf.keras.Model): Compiled Keras model instance.
    """

    def __init__(self, learning_rate):
        self.learning_rate = learning_rate
        self.model = self.build_model()

    def build_model(self):
        """
        Builds and compiles the deep learning model using a pre-trained ResNet50 as the base.

        Returns:
            tf.keras.Model: The compiled Keras model.
        """
        base_model = ResNet50(weights='imagenet', input_shape=(150, 150, 3), include_top=False)
        for layer in base_model.layers:
            layer.trainable = False

        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(1, activation='sigmoid')
        ])

        model.compile(optimizer=Adam(learning_rate=self.learning_rate), 
                      loss='binary_crossentropy', 
                      metrics=['binary_accuracy', 'AUC'])

        return model
