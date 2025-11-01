"""
Modified PointNet Architecture for Foot Segmentation
Segments foot point cloud into 22 anatomical regions
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import numpy as np
from typing import Tuple, Optional

# Define foot segment classes
FOOT_SEGMENTS = {
    0: 'background',
    # Toe segments (5)
    1: 'hallux',
    2: 'toe_2',
    3: 'toe_3',
    4: 'toe_4',
    5: 'toe_5',
    # Forefoot (4)
    6: 'medial_ball',
    7: 'lateral_ball',
    8: 'metatarsal_1',
    9: 'metatarsal_5',
    # Midfoot (3)
    10: 'medial_arch',
    11: 'lateral_arch',
    12: 'plantar_fascia',
    # Hindfoot (3)
    13: 'heel_pad',
    14: 'calcaneus_sides',
    15: 'achilles_insertion',
    # Dorsal regions (4)
    16: 'instep',
    17: 'dorsal_midfoot',
    18: 'ankle_transition',
    # Special features (3)
    19: 'bunion_area',
    20: 'tailors_bunion',
    21: 'hammer_toe_region'
}

NUM_CLASSES = len(FOOT_SEGMENTS)

class TNet(keras.Model):
    """Transformation Network for spatial transformer"""

    def __init__(self, k: int = 3):
        """
        Initialize T-Net

        Args:
            k: Dimension of transformation (3 for input, 64/128 for features)
        """
        super(TNet, self).__init__()
        self.k = k

        # Shared MLPs
        self.conv1 = layers.Conv1D(64, 1, activation='relu')
        self.conv2 = layers.Conv1D(128, 1, activation='relu')
        self.conv3 = layers.Conv1D(1024, 1, activation='relu')

        # Global feature extraction
        self.pool = layers.GlobalMaxPooling1D()

        # Fully connected layers
        self.fc1 = layers.Dense(512, activation='relu')
        self.fc2 = layers.Dense(256, activation='relu')

        # Output transformation matrix
        self.fc3 = layers.Dense(k * k,
                               kernel_initializer='zeros',
                               bias_initializer=tf.keras.initializers.Constant(np.eye(k).flatten()))

        self.dropout = layers.Dropout(0.3)
        self.bn1 = layers.BatchNormalization()
        self.bn2 = layers.BatchNormalization()
        self.bn3 = layers.BatchNormalization()
        self.bn4 = layers.BatchNormalization()
        self.bn5 = layers.BatchNormalization()

    def call(self, inputs, training=False):
        # Shared MLP
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)

        x = self.conv2(x)
        x = self.bn2(x, training=training)

        x = self.conv3(x)
        x = self.bn3(x, training=training)

        # Global feature
        x = self.pool(x)

        # FC layers
        x = self.fc1(x)
        x = self.bn4(x, training=training)
        x = self.dropout(x, training=training)

        x = self.fc2(x)
        x = self.bn5(x, training=training)
        x = self.dropout(x, training=training)

        # Output transformation
        x = self.fc3(x)
        x = tf.reshape(x, (-1, self.k, self.k))

        return x

class PointNetBackbone(keras.Model):
    """PointNet feature extraction backbone"""

    def __init__(self):
        super(PointNetBackbone, self).__init__()

        # Input transformation
        self.input_transform = TNet(k=3)

        # Shared MLPs for local features
        self.conv1 = layers.Conv1D(64, 1, activation='relu')
        self.conv2 = layers.Conv1D(64, 1, activation='relu')

        # Feature transformation
        self.feature_transform = TNet(k=64)

        # More shared MLPs
        self.conv3 = layers.Conv1D(64, 1, activation='relu')
        self.conv4 = layers.Conv1D(128, 1, activation='relu')
        self.conv5 = layers.Conv1D(1024, 1, activation='relu')

        # Batch normalization layers
        self.bn1 = layers.BatchNormalization()
        self.bn2 = layers.BatchNormalization()
        self.bn3 = layers.BatchNormalization()
        self.bn4 = layers.BatchNormalization()
        self.bn5 = layers.BatchNormalization()

    def call(self, inputs, training=False):
        # Input transformation
        transform_3d = self.input_transform(inputs, training=training)
        points_transformed = tf.matmul(inputs, transform_3d)

        # First shared MLP
        x = self.conv1(points_transformed)
        x = self.bn1(x, training=training)

        x = self.conv2(x)
        local_features = self.bn2(x, training=training)

        # Feature transformation
        transform_64d = self.feature_transform(local_features, training=training)
        features_transformed = tf.matmul(local_features, transform_64d)

        # Second shared MLP
        x = self.conv3(features_transformed)
        x = self.bn3(x, training=training)

        x = self.conv4(x)
        x = self.bn4(x, training=training)

        global_features = self.conv5(x)
        global_features = self.bn5(global_features, training=training)

        return local_features, global_features, transform_3d, transform_64d

class PointNetFootSegmentation(keras.Model):
    """Complete PointNet model for foot segmentation"""

    def __init__(self, num_classes: int = NUM_CLASSES):
        super(PointNetFootSegmentation, self).__init__()
        self.num_classes = num_classes

        # Feature extraction backbone
        self.backbone = PointNetBackbone()

        # Global feature aggregation
        self.pool = layers.GlobalMaxPooling1D()

        # Segmentation head
        self.seg_conv1 = layers.Conv1D(512, 1, activation='relu')
        self.seg_conv2 = layers.Conv1D(256, 1, activation='relu')
        self.seg_conv3 = layers.Conv1D(128, 1, activation='relu')
        self.seg_conv4 = layers.Conv1D(num_classes, 1)

        self.seg_bn1 = layers.BatchNormalization()
        self.seg_bn2 = layers.BatchNormalization()
        self.seg_bn3 = layers.BatchNormalization()

        self.dropout = layers.Dropout(0.3)

    def call(self, inputs, training=False):
        # Extract features
        local_features, global_features, trans_3d, trans_64d = self.backbone(inputs, training=training)

        # Get global feature vector
        global_vector = self.pool(global_features)
        global_vector = tf.expand_dims(global_vector, axis=1)

        # Tile global features to concatenate with local features
        num_points = tf.shape(inputs)[1]
        global_tiled = tf.tile(global_vector, [1, num_points, 1])

        # Concatenate local and global features
        combined_features = tf.concat([local_features, global_tiled], axis=-1)

        # Segmentation head
        x = self.seg_conv1(combined_features)
        x = self.seg_bn1(x, training=training)
        x = self.dropout(x, training=training)

        x = self.seg_conv2(x)
        x = self.seg_bn2(x, training=training)
        x = self.dropout(x, training=training)

        x = self.seg_conv3(x)
        x = self.seg_bn3(x, training=training)

        # Output logits
        logits = self.seg_conv4(x)

        return {
            'segmentation': logits,
            'transform_3d': trans_3d,
            'transform_64d': trans_64d
        }

def create_foot_segmentation_model(input_shape: Tuple[int, int] = (10000, 3),
                                  num_classes: int = NUM_CLASSES) -> keras.Model:
    """
    Create and compile the foot segmentation model

    Args:
        input_shape: Shape of input point cloud (num_points, 3)
        num_classes: Number of segmentation classes

    Returns:
        Compiled Keras model
    """
    inputs = keras.Input(shape=input_shape)
    model = PointNetFootSegmentation(num_classes)

    # Build model by calling it
    outputs = model(inputs, training=False)

    # Create functional model
    functional_model = keras.Model(inputs=inputs, outputs=outputs['segmentation'])

    return functional_model

def orthogonal_regularizer(weight_matrix):
    """Regularizer for transformation matrices to keep them orthogonal"""
    num_batch = tf.shape(weight_matrix)[0]
    w_t = tf.transpose(weight_matrix, perm=[0, 2, 1])
    identity = tf.eye(tf.shape(weight_matrix)[1], batch_shape=[num_batch])
    loss = tf.nn.l2_loss(tf.matmul(weight_matrix, w_t) - identity)
    return loss

class FootSegmentationLoss:
    """Custom loss function for foot segmentation"""

    def __init__(self, class_weights: Optional[np.ndarray] = None,
                 transform_reg_weight: float = 0.001):
        """
        Initialize loss function

        Args:
            class_weights: Weights for each class to handle imbalance
            transform_reg_weight: Weight for transformation regularization
        """
        self.class_weights = class_weights
        self.transform_reg_weight = transform_reg_weight

    def __call__(self, y_true, model_outputs):
        """Calculate total loss"""
        if isinstance(model_outputs, dict):
            y_pred = model_outputs['segmentation']
            trans_3d = model_outputs.get('transform_3d')
            trans_64d = model_outputs.get('transform_64d')
        else:
            y_pred = model_outputs
            trans_3d = None
            trans_64d = None

        # Segmentation loss (sparse categorical crossentropy)
        seg_loss = tf.keras.losses.sparse_categorical_crossentropy(
            y_true, y_pred, from_logits=True
        )

        # Apply class weights if provided
        if self.class_weights is not None:
            weights = tf.gather(self.class_weights, tf.cast(y_true, tf.int32))
            seg_loss = seg_loss * weights

        seg_loss = tf.reduce_mean(seg_loss)

        # Transformation regularization
        total_loss = seg_loss
        if trans_3d is not None:
            total_loss += self.transform_reg_weight * orthogonal_regularizer(trans_3d)
        if trans_64d is not None:
            total_loss += self.transform_reg_weight * orthogonal_regularizer(trans_64d)

        return total_loss

def compile_model(model: keras.Model,
                 learning_rate: float = 0.001,
                 class_weights: Optional[np.ndarray] = None):
    """
    Compile the model with appropriate optimizer and loss

    Args:
        model: Keras model to compile
        learning_rate: Learning rate for Adam optimizer
        class_weights: Optional class weights for handling imbalance
    """
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

    # Use custom loss if we need class weights or transformation regularization
    loss_fn = FootSegmentationLoss(class_weights=class_weights)

    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=['accuracy', keras.metrics.SparseCategoricalAccuracy()]
    )

    return model