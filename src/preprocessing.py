"""
Source code for preprocessing.py
"""
import re
import html
import pandas as pd
from sklearn.model_selection import train_test_split

def clean_social_media_text(text):
    """
    Cleans social media text while preserving emotionally meaningful linguistic features.
    
    Decisions & Rationale:
    - HTML Entities: Unescaped (e.g. &amp; to &) as they are artifacts of web scraping.
    - URLs: Removed. URLs in Reddit posts usually point to external context not available to the model.
    - Punctuation/Emojis: PRESERVED. Multiple question marks (???), exclamation points, and emojis carry heavy emotional sentiment critical for distress detection.
    - Negations: PRESERVED. Negations (not, never) are fundamental to semantic meaning.
    - Whitespace: Normalized to prevent tokenization artifacts.
    """
    if not isinstance(text, str):
        return ""
    
    # Decode HTML entities
    text = html.unescape(text)
    
    # Remove URLs
    text = re.sub(r'http\S+|www\.\S+', '', text)
    
    # Normalize whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def remove_data_leakage_and_noise(df, text_col, label_col):
    """
    Handles missing text, empty text, duplicate text, and conflicting labels to prevent data leakage.
    """
    # 1. Remove missing or empty text
    df_clean = df.dropna(subset=[text_col, label_col]).copy()
    df_clean = df_clean[df_clean[text_col].astype(str).str.strip() != ""]
    
    # 2. Find all duplicates based on text
    duplicates = df_clean[df_clean.duplicated(subset=[text_col], keep=False)]
    
    # Find texts that have conflicting labels across identical texts
    leakage_groups = duplicates.groupby(text_col)[label_col].nunique()
    conflicting_texts = leakage_groups[leakage_groups > 1].index
    
    # 3. Remove all texts with conflicting labels (they represent pure noise or leakage)
    df_clean = df_clean[~df_clean[text_col].isin(conflicting_texts)]
    
    # 4. For the remaining duplicates (which have the same label), keep only the first occurrence
    df_clean = df_clean.drop_duplicates(subset=[text_col], keep='first')
    
    return df_clean

def create_stratified_splits(df, label_col, train_size=0.8, val_size=0.1, test_size=0.1, seed=42):
    """
    Creates reproducible stratified train, validation, and test splits.
    Returns the updated DataFrame with a new 'split' column to avoid duplicating data files.
    """
    # Normalize sizes to ensure they sum to 1.0 (or close due to float precision)
    total = train_size + val_size + test_size
    train_size_norm = train_size / total
    val_size_norm = val_size / total
    test_size_norm = test_size / total
    
    # First split: Train vs Temp (Val + Test)
    train_df, temp_df = train_test_split(
        df, 
        train_size=train_size_norm, 
        stratify=df[label_col], 
        random_state=seed
    )
    
    # Second split: Val vs Test
    relative_val_size = val_size_norm / (val_size_norm + test_size_norm)
    val_df, test_df = train_test_split(
        temp_df,
        train_size=relative_val_size,
        stratify=temp_df[label_col],
        random_state=seed
    )
    
    df_result = df.copy()
    df_result['split'] = 'unassigned'
    df_result.loc[train_df.index, 'split'] = 'train'
    df_result.loc[val_df.index, 'split'] = 'val'
    df_result.loc[test_df.index, 'split'] = 'test'
    
    return df_result
