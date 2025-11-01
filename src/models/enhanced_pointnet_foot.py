"""
Enhanced PointNet Architecture for Detailed Foot Segmentation
Segments foot point cloud into 45 detailed anatomical regions for medical analysis
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import numpy as np
from typing import Tuple, Optional, Dict

# Expanded foot segment classes with medical relevance
DETAILED_FOOT_SEGMENTS = {
    # Toe segments - Individual phalanges (15 regions)
    0: 'background',
    1: 'hallux_distal_phalanx',
    2: 'hallux_proximal_phalanx',
    3: 'hallux_mtp_joint',  # Metatarsophalangeal joint
    4: 'toe2_phalanges',
    5: 'toe2_mtp_joint',
    6: 'toe3_phalanges',
    7: 'toe3_mtp_joint',
    8: 'toe4_phalanges',
    9: 'toe4_mtp_joint',
    10: 'toe5_phalanges',
    11: 'toe5_mtp_joint',
    12: 'interdigital_space_1',  # Between toes
    13: 'interdigital_space_2',
    14: 'interdigital_space_3',
    15: 'interdigital_space_4',

    # Forefoot - Metatarsal regions (8 regions)
    16: 'first_metatarsal_head',
    17: 'first_metatarsal_shaft',
    18: 'second_metatarsal',
    19: 'third_metatarsal',
    20: 'fourth_metatarsal',
    21: 'fifth_metatarsal_head',
    22: 'fifth_metatarsal_shaft',
    23: 'transverse_arch',

    # Midfoot - Detailed arch anatomy (8 regions)
    24: 'medial_longitudinal_arch',
    25: 'lateral_longitudinal_arch',
    26: 'navicular_region',
    27: 'cuboid_region',
    28: 'cuneiform_region',
    29: 'plantar_fascia_origin',
    30: 'plantar_fascia_central',
    31: 'plantar_fascia_insertion',

    # Hindfoot - Heel and ankle (6 regions)
    32: 'calcaneus_plantar',  # Heel pad
    33: 'calcaneus_medial',
    34: 'calcaneus_lateral',
    35: 'calcaneus_posterior',
    36: 'sustentaculum_tali',  # Medial shelf of calcaneus
    37: 'achilles_insertion',

    # Dorsal surface (5 regions)
    38: 'dorsal_forefoot',
    39: 'dorsal_midfoot',
    40: 'instep_medial',
    41: 'instep_lateral',
    42: 'dorsal_ankle_transition',

    # Special medical regions (3 regions)
    43: 'bunion_region',  # First MTP prominence
    44: 'tailors_bunion_region',  # Fifth MTP prominence
    45: 'plantar_fat_pad'  # Important for pressure distribution
}

NUM_DETAILED_CLASSES = len(DETAILED_FOOT_SEGMENTS)

# Medical condition indicators mapped to relevant regions
CONDITION_REGION_MAPPING = {
    'bunion': [3, 16, 17, 43],  # Hallux MTP and first metatarsal
    'hammer_toe': [4, 6, 8, 10],  # Lesser toe phalanges
    'plantar_fasciitis': [29, 30, 31, 32],  # Plantar fascia and heel
    'flat_foot': [24, 25, 26, 27, 28],  # All arch regions
    'high_arch': [24, 25, 26],  # Longitudinal arches
    'gout': [3, 5, 7, 9, 11, 37],  # MTP joints and ankle
    'morton_neuroma': [12, 13, 14, 15, 18, 19],  # Interdigital spaces
    'heel_spur': [32, 29],  # Plantar calcaneus
    'achilles_tendinitis': [37, 35],  # Achilles insertion
    'metatarsalgia': [16, 18, 19, 20, 21]  # Metatarsal heads
}

class EnhancedTNet(keras.Model):
    """Enhanced Transformation Network with attention mechanism"""

    def __init__(self, k: int = 3):
        super(EnhancedTNet, self).__init__()
        self.k = k

        # Enhanced feature extraction
        self.conv1 = layers.Conv1D(64, 1, activation='relu')
        self.conv2 = layers.Conv1D(128, 1, activation='relu')
        self.conv3 = layers.Conv1D(256, 1, activation='relu')
        self.conv4 = layers.Conv1D(512, 1, activation='relu')
        self.conv5 = layers.Conv1D(1024, 1, activation='relu')

        # Attention mechanism
        self.attention = layers.MultiHeadAttention(
            num_heads=4,
            key_dim=256
        )

        # Global feature extraction
        self.pool = layers.GlobalMaxPooling1D()

        # Dense layers with dropout
        self.fc1 = layers.Dense(512, activation='relu')
        self.fc2 = layers.Dense(256, activation='relu')
        self.fc3 = layers.Dense(128, activation='relu')

        # Output transformation matrix
        self.fc_final = layers.Dense(
            k * k,
            kernel_initializer='zeros',
            bias_initializer=tf.keras.initializers.Constant(np.eye(k).flatten())
        )

        # Normalization and regularization
        self.bn_layers = [layers.BatchNormalization() for _ in range(8)]
        self.dropout = layers.Dropout(0.3)

    def call(self, inputs, training=False):
        # Progressive feature extraction
        x = self.conv1(inputs)
        x = self.bn_layers[0](x, training=training)

        x = self.conv2(x)
        x = self.bn_layers[1](x, training=training)

        x = self.conv3(x)
        x = self.bn_layers[2](x, training=training)

        # Apply attention
        x = self.attention(x, x, training=training)

        x = self.conv4(x)
        x = self.bn_layers[3](x, training=training)

        x = self.conv5(x)
        x = self.bn_layers[4](x, training=training)

        # Global features
        x = self.pool(x)

        # Dense layers
        x = self.fc1(x)
        x = self.bn_layers[5](x, training=training)
        x = self.dropout(x, training=training)

        x = self.fc2(x)
        x = self.bn_layers[6](x, training=training)
        x = self.dropout(x, training=training)

        x = self.fc3(x)
        x = self.bn_layers[7](x, training=training)

        # Output transformation
        x = self.fc_final(x)
        x = tf.reshape(x, (-1, self.k, self.k))

        return x

class MedicalFeatureExtractor(keras.Model):
    """Extracts medical condition-specific features from point cloud"""

    def __init__(self):
        super(MedicalFeatureExtractor, self).__init__()

        # Specialized convolutions for medical features
        self.pressure_conv = layers.Conv1D(64, 1, activation='relu', name='pressure_features')
        self.deformity_conv = layers.Conv1D(64, 1, activation='relu', name='deformity_features')
        self.inflammation_conv = layers.Conv1D(64, 1, activation='relu', name='inflammation_features')
        self.structural_conv = layers.Conv1D(128, 1, activation='relu', name='structural_features')

        # Feature fusion
        self.fusion_conv = layers.Conv1D(256, 1, activation='relu')

        # Batch normalization
        self.bn_layers = [layers.BatchNormalization() for _ in range(5)]

    def call(self, x, training=False):
        # Extract different medical feature types
        pressure = self.pressure_conv(x)
        pressure = self.bn_layers[0](pressure, training=training)

        deformity = self.deformity_conv(x)
        deformity = self.bn_layers[1](deformity, training=training)

        inflammation = self.inflammation_conv(x)
        inflammation = self.bn_layers[2](inflammation, training=training)

        structural = self.structural_conv(x)
        structural = self.bn_layers[3](structural, training=training)

        # Concatenate all features
        combined = tf.concat([pressure, deformity, inflammation, structural], axis=-1)

        # Fusion layer
        fused = self.fusion_conv(combined)
        fused = self.bn_layers[4](fused, training=training)

        return fused

class EnhancedPointNetBackbone(keras.Model):
    """Enhanced backbone with medical feature extraction"""

    def __init__(self):
        super(EnhancedPointNetBackbone, self).__init__()

        # Input transformation
        self.input_transform = EnhancedTNet(k=3)

        # Local feature extraction
        self.conv1 = layers.Conv1D(64, 1, activation='relu')
        self.conv2 = layers.Conv1D(128, 1, activation='relu')

        # Feature transformation
        self.feature_transform = EnhancedTNet(k=128)

        # Medical feature extraction
        self.medical_features = MedicalFeatureExtractor()

        # Deep feature extraction
        self.conv3 = layers.Conv1D(256, 1, activation='relu')
        self.conv4 = layers.Conv1D(512, 1, activation='relu')
        self.conv5 = layers.Conv1D(1024, 1, activation='relu')
        self.conv6 = layers.Conv1D(2048, 1, activation='relu')

        # Batch normalization
        self.bn_layers = [layers.BatchNormalization() for _ in range(6)]

    def call(self, inputs, training=False):
        # Input transformation
        transform_3d = self.input_transform(inputs, training=training)
        points_transformed = tf.matmul(inputs, transform_3d)

        # Initial feature extraction
        x = self.conv1(points_transformed)
        x = self.bn_layers[0](x, training=training)

        x = self.conv2(x)
        local_features = self.bn_layers[1](x, training=training)

        # Feature transformation
        transform_128d = self.feature_transform(local_features, training=training)
        features_transformed = tf.matmul(local_features, transform_128d)

        # Extract medical features
        medical_feats = self.medical_features(features_transformed, training=training)

        # Deep feature extraction
        x = self.conv3(features_transformed)
        x = self.bn_layers[2](x, training=training)

        x = self.conv4(x)
        x = self.bn_layers[3](x, training=training)

        x = self.conv5(x)
        x = self.bn_layers[4](x, training=training)

        global_features = self.conv6(x)
        global_features = self.bn_layers[5](global_features, training=training)

        return local_features, medical_feats, global_features, transform_3d, transform_128d

class DetailedFootSegmentation(keras.Model):
    """Complete model for detailed foot segmentation with medical analysis"""

    def __init__(self, num_classes: int = NUM_DETAILED_CLASSES):
        super(DetailedFootSegmentation, self).__init__()
        self.num_classes = num_classes

        # Feature extraction backbone
        self.backbone = EnhancedPointNetBackbone()

        # Global feature aggregation
        self.pool = layers.GlobalMaxPooling1D()

        # Segmentation head with skip connections
        self.seg_conv1 = layers.Conv1D(1024, 1, activation='relu')
        self.seg_conv2 = layers.Conv1D(512, 1, activation='relu')
        self.seg_conv3 = layers.Conv1D(256, 1, activation='relu')
        self.seg_conv4 = layers.Conv1D(128, 1, activation='relu')
        self.seg_conv5 = layers.Conv1D(num_classes, 1)  # Output logits

        # Medical condition detection heads
        self.condition_heads = {
            'plantar_pressure': layers.Conv1D(1, 1, activation='sigmoid'),
            'deformity_score': layers.Conv1D(1, 1, activation='sigmoid'),
            'inflammation_score': layers.Conv1D(1, 1, activation='sigmoid')
        }

        self.seg_bn_layers = [layers.BatchNormalization() for _ in range(4)]
        self.dropout = layers.Dropout(0.3)

    def call(self, inputs, training=False):
        # Extract features
        local_features, medical_features, global_features, trans_3d, trans_128d = \
            self.backbone(inputs, training=training)

        # Get global feature vector
        global_vector = self.pool(global_features)
        global_vector = tf.expand_dims(global_vector, axis=1)

        # Tile global features
        num_points = tf.shape(inputs)[1]
        global_tiled = tf.tile(global_vector, [1, num_points, 1])

        # Concatenate local, medical, and global features
        combined_features = tf.concat(
            [local_features, medical_features, global_tiled],
            axis=-1
        )

        # Progressive segmentation refinement
        x = self.seg_conv1(combined_features)
        x = self.seg_bn_layers[0](x, training=training)
        x = self.dropout(x, training=training)

        x = self.seg_conv2(x)
        x = self.seg_bn_layers[1](x, training=training)
        x = self.dropout(x, training=training)

        x = self.seg_conv3(x)
        x = self.seg_bn_layers[2](x, training=training)
        x = self.dropout(x, training=training)

        x = self.seg_conv4(x)
        x = self.seg_bn_layers[3](x, training=training)

        # Final segmentation
        segmentation_logits = self.seg_conv5(x)

        # Medical condition scores
        condition_scores = {}
        for name, head in self.condition_heads.items():
            condition_scores[name] = head(medical_features)

        return {
            'segmentation': segmentation_logits,
            'condition_scores': condition_scores,
            'medical_features': medical_features,
            'transform_3d': trans_3d,
            'transform_128d': trans_128d
        }

def create_enhanced_segmentation_model(
    input_shape: Tuple[int, int] = (10000, 3),
    num_classes: int = NUM_DETAILED_CLASSES
) -> keras.Model:
    """Create and compile the enhanced foot segmentation model"""

    inputs = keras.Input(shape=input_shape)
    model = DetailedFootSegmentation(num_classes)

    # Build model
    outputs = model(inputs, training=False)

    # Create functional model
    functional_model = keras.Model(inputs=inputs, outputs=outputs)

    return functional_model

class MedicalSegmentationLoss:
    """Custom loss function for medical foot segmentation"""

    def __init__(self,
                 class_weights: Optional[np.ndarray] = None,
                 condition_weight: float = 0.1,
                 transform_reg_weight: float = 0.001):
        self.class_weights = class_weights
        self.condition_weight = condition_weight
        self.transform_reg_weight = transform_reg_weight

    def __call__(self, y_true, model_outputs):
        """Calculate total loss including segmentation and medical conditions"""

        # Extract outputs
        seg_logits = model_outputs['segmentation']
        condition_scores = model_outputs.get('condition_scores', {})
        trans_3d = model_outputs.get('transform_3d')
        trans_128d = model_outputs.get('transform_128d')

        # Segmentation loss
        seg_loss = tf.keras.losses.sparse_categorical_crossentropy(
            y_true['segmentation'], seg_logits, from_logits=True
        )

        # Apply class weights if provided
        if self.class_weights is not None:
            weights = tf.gather(self.class_weights, tf.cast(y_true['segmentation'], tf.int32))
            seg_loss = seg_loss * weights

        seg_loss = tf.reduce_mean(seg_loss)

        # Medical condition losses (if labels provided)
        condition_loss = 0
        for name, scores in condition_scores.items():
            if name in y_true:
                cond_loss = tf.keras.losses.binary_crossentropy(
                    y_true[name], scores
                )
                condition_loss += tf.reduce_mean(cond_loss)

        # Transformation regularization
        transform_loss = 0
        if trans_3d is not None:
            transform_loss += orthogonal_regularizer(trans_3d)
        if trans_128d is not None:
            transform_loss += orthogonal_regularizer(trans_128d)

        # Total loss
        total_loss = seg_loss + \
                    self.condition_weight * condition_loss + \
                    self.transform_reg_weight * transform_loss

        return total_loss

def orthogonal_regularizer(weight_matrix):
    """Regularizer for transformation matrices"""
    num_batch = tf.shape(weight_matrix)[0]
    w_t = tf.transpose(weight_matrix, perm=[0, 2, 1])
    identity = tf.eye(tf.shape(weight_matrix)[1], batch_shape=[num_batch])
    loss = tf.nn.l2_loss(tf.matmul(weight_matrix, w_t) - identity)
    return loss