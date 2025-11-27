"""
Dataset loading and preprocessing utilities for Reddit temporal classification.
"""

import os
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer


def load_reddit_data(
    data_dir: str = "data/sampled_comments",
    start_year: int = 2006,
    end_year: int = 2024,
    months: Optional[List[int]] = None,
    max_samples_per_month: Optional[int] = None,
    specific_years: Optional[List[int]] = None
) -> pd.DataFrame:
    """
    Load Reddit comment data from CSV files.
    
    Args:
        data_dir: Directory containing monthly CSV files
        start_year: Starting year (inclusive)
        end_year: Ending year (inclusive)
        months: List of months to include (1-12). If None, includes all months.
        max_samples_per_month: Maximum samples per month file. If None, loads all.
        specific_years: If provided, only load data from these years (overrides start_year/end_year)
    
    Returns:
        DataFrame with columns: subreddit, subreddit_id, body, date_created_utc, year
    """
    dfs = []
    
    years_to_load = specific_years if specific_years else list(range(start_year, end_year + 1))
    months_to_load = months if months else list(range(1, 13))
    
    for year in years_to_load:
        for month in months_to_load:
            path = os.path.join(data_dir, f"RC_{year}-{month:02d}.csv")
            try:
                df = pd.read_csv(path, header=None)
                if max_samples_per_month:
                    df = df.head(max_samples_per_month)
                df.columns = ["subreddit", "subreddit_id", "body", "date_created_utc"]
                df["year"] = pd.to_datetime(df["date_created_utc"], unit="s").dt.year
                dfs.append(df)
            except FileNotFoundError:
                continue
        
        print(f"  Loaded {year}")
    
    if not dfs:
        raise ValueError(f"No data files found in {data_dir}")
    
    final_df = pd.concat(dfs, ignore_index=True)
    return final_df


def prepare_classification_labels(df: pd.DataFrame, years: List[int]) -> pd.DataFrame:
    """
    Prepare labels for binary/multi-class classification.
    
    Args:
        df: DataFrame with 'year' column
        years: List of years to classify (e.g., [2013, 2024])
    
    Returns:
        DataFrame with 'label' column added (0-indexed class labels)
    """
    df = df[df['year'].isin(years)].copy()
    df['label'] = df['year'].apply(lambda y: years.index(y))
    return df


def prepare_regression_labels(df: pd.DataFrame, start_year: int = 2006, end_year: int = 2024) -> pd.DataFrame:
    """
    Prepare labels for regression (year index).
    
    Args:
        df: DataFrame with 'year' column
        start_year: Starting year (maps to label 0)
        end_year: Ending year (maps to label end_year - start_year)
    
    Returns:
        DataFrame with 'label' column added (0-indexed year indices)
    """
    keep_years = list(range(start_year, end_year + 1))
    df = df[df['year'].isin(keep_years)].copy()
    df['label'] = df['year'].apply(lambda y: keep_years.index(y))
    return df


def split_data(
    df: pd.DataFrame,
    train_size: float = 0.8,
    val_size: float = 0.1,
    test_size: float = 0.1,
    stratify_col: Optional[str] = 'label',
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data into train/validation/test sets.
    
    Args:
        df: DataFrame to split
        train_size: Proportion for training set
        val_size: Proportion for validation set
        test_size: Proportion for test set (should sum to 1.0 with train_size + val_size)
        stratify_col: Column to stratify on (usually 'label')
        random_state: Random seed
    
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    assert abs(train_size + val_size + test_size - 1.0) < 1e-6, "Sizes must sum to 1.0"
    
    stratify = df[stratify_col] if stratify_col else None
    
    # First split: train vs (val + test)
    train_df, temp_df = train_test_split(
        df,
        test_size=(val_size + test_size),
        stratify=stratify,
        random_state=random_state
    )
    
    # Second split: val vs test
    val_ratio = val_size / (val_size + test_size)
    stratify_temp = temp_df[stratify_col] if stratify_col else None
    
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1 - val_ratio),
        stratify=stratify_temp,
        random_state=random_state
    )
    
    print(f"Train size: {len(train_df):,} ({len(train_df)/len(df)*100:.1f}%)")
    print(f"Val size:   {len(val_df):,} ({len(val_df)/len(df)*100:.1f}%)")
    print(f"Test size:  {len(test_df):,} ({len(test_df)/len(df)*100:.1f}%)\n")
    
    return train_df, val_df, test_df


def tokenize_texts(
    texts: List[str],
    tokenizer: AutoTokenizer,
    max_length: int = 256,
    padding: str = 'longest',
    truncation: bool = True
) -> dict:
    """
    Tokenize a list of texts.
    
    Args:
        texts: List of text strings
        tokenizer: HuggingFace tokenizer
        max_length: Maximum sequence length
        padding: Padding strategy ('max_length' or 'longest')
        truncation: Whether to truncate sequences
    
    Returns:
        Dictionary of tokenized encodings
    """
    # Clean and convert to strings
    texts = ["" if pd.isna(text) else str(text) for text in texts]
    
    return tokenizer(
        texts,
        truncation=truncation,
        padding=padding,
        max_length=max_length,
        return_tensors=None
    )

