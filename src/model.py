"""
Contextual Emotion Distribution Vectorizer (CEDV) Model Architecture.

Architecture:
- TextVectorization (vocab_size=15,000, sequence_length=128)
- Embedding (128-dim)
- SpatialDropout1D (0.25)
- Bidirectional LSTM (64 units, return_sequences=True)
- Conv1D (128 filters, kernel_size=3, padding='same', activation='relu')
- MaskedFeaturePooling (Masked Global Max + Masked Global Average Pooling)
- Dense (64 units, ReLU) + Dropout (0.4) + L2 Regularization
- Dense (8 units, Linear Logits z, dtype='float32')

Distribution Heads:
- Option A: Softmax Simplex Distribution (sum=1.0, v_i >= 0) with temperature tau.
- Option B: Z-Score Standardization (mean=0.0, std=1.0).
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import keras
from keras import layers
import numpy as np
import tensorflow as tf

from src.preprocess import CLASS_LABELS, NUM_CLASSES, clean_text


@keras.saving.register_keras_serializable(package="cedv")
class MaskedFeaturePooling(layers.Layer):
    """
    Mask-aware dual pooling layer (Masked Max Pooling + Masked Mean Pooling).
    Eliminates padding artifacts for short conversational texts by strictly ignoring zero-padded timesteps.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        features, token_ids = inputs
        # mask is 1.0 for real tokens (> 0), 0.0 for padding tokens (== 0)
        mask = keras.ops.cast(keras.ops.not_equal(token_ids, 0), features.dtype)
        mask_exp = keras.ops.expand_dims(mask, -1)

        # 1. Masked Max Pooling: set padding timesteps to -1e4 (fits cleanly in float16 range [-65504, 65504])
        neg_pad = -1e4 * (1.0 - mask_exp)
        masked_features = features + neg_pad
        pooled_max = keras.ops.maximum(keras.ops.max(masked_features, axis=1), 0.0)

        # 2. Masked Average Pooling: average only across non-zero tokens
        lengths = keras.ops.maximum(keras.ops.sum(mask_exp, axis=1), 1.0)
        pooled_mean = keras.ops.sum(features * mask_exp, axis=1) / lengths

        # Concatenate both pooled features (captures sharp keyword spikes + overall semantic tone)
        return keras.ops.concatenate([pooled_max, pooled_mean], axis=-1)

    def compute_output_shape(self, input_shape):
        feat_shape, _ = input_shape
        return (feat_shape[0], feat_shape[2] * 2)

    def get_config(self):
        return super().get_config()


def build_cedv_model(
    vocab_size: int = 15000,
    seq_length: int = 128,
    embedding_dim: int = 128,
    lstm_units: int = 64,
    conv_filters: int = 128,
    dense_units: int = 64,
    num_classes: int = NUM_CLASSES,
    vectorizer: Optional[layers.TextVectorization] = None,
) -> keras.Model:
    """
    Construct CEDV native Keras model with mask-aware pooling for raw text input.
    """
    if vectorizer is None:
        vectorizer = layers.TextVectorization(
            max_tokens=vocab_size,
            output_mode="int",
            output_sequence_length=seq_length,
            name="text_vectorization",
        )

    # Input layer accepts raw strings
    text_input = layers.Input(shape=(1,), dtype=tf.string, name="text_input")
    
    # Text vectorization layer produces integer token IDs
    token_ids = vectorizer(text_input)
    
    # Embedding with Spatial Dropout
    embedded = layers.Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        name="embedding",
    )(token_ids)
    dropped_emb = layers.SpatialDropout1D(0.25, name="spatial_dropout")(embedded)
    
    # Temporal sequence modeling via Bidirectional LSTM
    lstm_out = layers.Bidirectional(
        layers.LSTM(lstm_units, return_sequences=True),
        name="bidirectional_lstm",
    )(dropped_emb)
    
    # Local feature extraction via 1D Convolution
    conv_out = layers.Conv1D(
        filters=conv_filters,
        kernel_size=3,
        padding="same",
        activation="relu",
        name="conv1d",
    )(lstm_out)
    
    # Masked Dual Pooling (Max + Average) strictly over active tokens
    pooled = MaskedFeaturePooling(name="masked_pooling")([conv_out, token_ids])
    
    # Dense bottleneck projection with regularizer and dropout
    dense = layers.Dense(
        dense_units,
        activation="relu",
        kernel_regularizer=keras.regularizers.l2(1e-4),
        name="dense_proj",
    )(pooled)
    dense_drop = layers.Dropout(0.4, name="dense_dropout")(dense)
    
    # Linear projection to 8 logits (strictly float32 for numerical stability)
    logits = layers.Dense(
        num_classes,
        activation=None,
        dtype="float32",
        name="logits",
    )(dense_drop)

    model = keras.Model(inputs=text_input, outputs=logits, name="CEDV_Model")
    return model


def simplex_distribution(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """
    Convert raw logits to normalized simplex probability distribution (Option A).
    sum(v_i) == 1.0, v_i >= 0
    """
    scaled = logits / max(temperature, 1e-5)
    shifted = scaled - np.max(scaled, axis=-1, keepdims=True)
    exp_vals = np.exp(shifted)
    probs = exp_vals / np.sum(exp_vals, axis=-1, keepdims=True)
    return probs


def zscore_distribution(logits: np.ndarray, epsilon: float = 1e-7) -> np.ndarray:
    """
    Convert raw logits to standardized Z-scores (Option B).
    mean(v_i) == 0.0, std(v_i) == 1.0
    """
    mean = np.mean(logits, axis=-1, keepdims=True)
    std = np.std(logits, axis=-1, keepdims=True)
    zscores = (logits - mean) / (std + epsilon)
    return zscores


class CEDVInferenceEngine:
    """
    Production-ready inference wrapper for CEDV.
    Performs sanitization, forward pass, and dual-mode vector transformations.
    """

    def __init__(self, model: keras.Model, class_labels: Optional[List[str]] = None):
        self.model = model
        self.class_labels = class_labels or CLASS_LABELS

    @classmethod
    def load_from_path(cls, model_path: str) -> "CEDVInferenceEngine":
        """Load trained Keras CEDV model from disk."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        model = keras.models.load_model(model_path, compile=False)
        return cls(model=model)

    def predict_vectors(
        self,
        texts: Union[str, List[str]],
        temperature: float = 1.0,
    ) -> Dict[str, np.ndarray]:
        """
        Execute forward pass and return raw logits, simplex probabilities, and Z-scores.
        """
        if isinstance(texts, str):
            single_input = True
            raw_list = [texts]
        else:
            single_input = False
            raw_list = list(texts)

        cleaned_list = [clean_text(t) for t in raw_list]
        input_tensor = tf.constant(cleaned_list, dtype=tf.string)

        logits = self.model(input_tensor, training=False).numpy()
        simplex = simplex_distribution(logits, temperature=temperature)
        zscore = zscore_distribution(logits)

        return {
            "logits": logits[0] if single_input else logits,
            "simplex": simplex[0] if single_input else simplex,
            "zscore": zscore[0] if single_input else zscore,
            "cleaned_texts": cleaned_list[0] if single_input else cleaned_list,
        }

    def analyze_text(
        self,
        text: str,
        temperature: float = 1.0,
        zscore_threshold: float = 0.5,
    ) -> Dict[str, object]:
        """
        Comprehensive emotional profile analysis for a single text input.
        """
        results = self.predict_vectors(text, temperature=temperature)
        simplex = results["simplex"]
        zscore = results["zscore"]

        primary_idx = int(np.argmax(simplex))
        primary_label = self.class_labels[primary_idx]
        primary_confidence = float(simplex[primary_idx])

        secondary_nuances = []
        for idx, (label, z_val, p_val) in enumerate(zip(self.class_labels, zscore, simplex)):
            if idx != primary_idx and z_val >= zscore_threshold:
                secondary_nuances.append({
                    "label": label,
                    "probability": float(p_val),
                    "zscore": float(z_val),
                })
        
        secondary_nuances.sort(key=lambda item: item["probability"], reverse=True)

        breakdown = []
        for idx, label in enumerate(self.class_labels):
            breakdown.append({
                "label": label,
                "class_id": idx,
                "probability": float(simplex[idx]),
                "zscore": float(zscore[idx]),
                "logit": float(results["logits"][idx]),
            })

        return {
            "cleaned_text": results["cleaned_texts"],
            "primary_emotion": primary_label,
            "primary_confidence": primary_confidence,
            "secondary_nuances": secondary_nuances,
            "breakdown": breakdown,
            "simplex_vector": simplex.tolist(),
            "zscore_vector": zscore.tolist(),
        }
