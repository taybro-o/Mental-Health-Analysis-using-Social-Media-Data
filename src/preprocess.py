"""
Text Preprocessing and tf.data Pipeline for CEDV.

Provides:
- Robust text cleaning (URLs, markdown, reddit handles, HTML entities).
- Label encoding & one-hot categorical matrices for 8 affective classes.
- Stratified dataset partitioning (Train / Validation / Test).
- Optimized tf.data.Dataset creation with AUTOTUNE prefetching.
"""

import html
import re
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf

# Canonical 8-class label mapping
CLASS_LABELS: List[str] = [
    "depression",        # 0
    "anxiety",           # 1
    "stress_burnout",    # 2
    "adhd",              # 3
    "autism",            # 4
    "loneliness",        # 5
    "anger",             # 6
    "positive_neutral",  # 7
]

LABEL_TO_INDEX: Dict[str, int] = {label: idx for idx, label in enumerate(CLASS_LABELS)}
INDEX_TO_LABEL: Dict[int, str] = {idx: label for idx, label in enumerate(CLASS_LABELS)}
NUM_CLASSES: int = len(CLASS_LABELS)


def clean_text(text: str) -> str:
    """
    Clean and normalize conversational / social media text.
    Preserves semantic affective markers while removing URLs, handles, and markup.
    """
    if not isinstance(text, str):
        return ""

    # Unescape HTML entities (&amp;, &gt;, &#x200B;, etc.)
    text = html.unescape(text)
    
    # Remove URLs and links
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    
    # Remove Reddit sub/user references (r/depression -> depression, u/username -> "")
    text = re.sub(r"\br/([a-zA-Z0-9_]+)", r"\1", text)
    text = re.sub(r"\bu/([a-zA-Z0-9_-]+)", "", text)
    
    # Remove Reddit boilerplate
    text = re.sub(r"\[(deleted|removed)\]", "", text, flags=re.IGNORECASE)
    
    # Remove markdown formatting characters like **, __, ~~, `, >, #
    text = re.sub(r"[\*_~`>#]", " ", text)
    
    # Collapse multiple punctuation sequences (e.g. "??????" -> "???")
    text = re.sub(r"([!?.,])\1{3,}", r"\1\1\1", text)
    
    # Normalize whitespaces and linebreaks
    text = re.sub(r"\s+", " ", text).strip()
    
    return text


def load_and_preprocess_dataframe(
    csv_path: str,
    min_word_count: int = 3,
) -> pd.DataFrame:
    """
    Load dataset from CSV, clean text, filter invalid/too-short entries, and assign class_id.
    """
    df = pd.read_csv(csv_path)
    required_cols = {"text", "label"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Dataset must contain columns {required_cols}, found {df.columns.tolist()}")

    # Drop nulls
    df = df.dropna(subset=["text", "label"]).copy()
    
    # Filter to known 8 classes
    df = df[df["label"].isin(LABEL_TO_INDEX.keys())].copy()
    
    # Clean texts
    df["clean_text"] = df["text"].astype(str).apply(clean_text)
    
    # Filter by word length
    df["word_count"] = df["clean_text"].apply(lambda t: len(t.split()))
    df = df[df["word_count"] >= min_word_count].copy()
    
    # Map labels to integer class IDs
    df["class_id"] = df["label"].map(LABEL_TO_INDEX).astype(int)
    
    # Drop duplicates by clean text
    df = df.drop_duplicates(subset=["clean_text"]).reset_index(drop=True)
    
    return df


def prepare_dataset_splits(
    df: pd.DataFrame,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split dataframe into stratified Train, Validation, and Test partitions.
    """
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df["class_id"],
        random_state=random_state,
    )

    relative_val_size = val_size / (1.0 - test_size)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=relative_val_size,
        stratify=train_val_df["class_id"],
        random_state=random_state,
    )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def create_tf_dataset(
    texts: Union[List[str], np.ndarray, pd.Series],
    labels: Optional[Union[List[int], np.ndarray, pd.Series]] = None,
    batch_size: int = 32,
    shuffle: bool = False,
    shuffle_buffer: int = 1000,
    one_hot: bool = True,
    num_classes: int = NUM_CLASSES,
) -> tf.data.Dataset:
    """
    Construct high-throughput tf.data.Dataset with AUTOTUNE prefetching.
    Yields (raw_string_tensor, one_hot_tensor) or (raw_string_tensor, None).
    """
    text_array = np.array(texts, dtype=object)

    if labels is not None:
        label_array = np.array(labels, dtype=np.int32)
        if one_hot:
            label_matrix = tf.keras.utils.to_categorical(label_array, num_classes=num_classes)
            ds = tf.data.Dataset.from_tensor_slices((text_array, label_matrix))
        else:
            ds = tf.data.Dataset.from_tensor_slices((text_array, label_array))
    else:
        ds = tf.data.Dataset.from_tensor_slices(text_array)

    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(text_array), shuffle_buffer), reshuffle_each_iteration=True)

    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    return ds
