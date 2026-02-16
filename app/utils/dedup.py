from typing import Tuple

import pandas as pd


def split_duplicates_by_phash(df: pd.DataFrame, phash_col: str = "phash") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a DataFrame into kept (first occurrence per pHash) and removed (subsequent duplicates).

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing a column for pHash.
    phash_col : str
        Column name holding the pHash values. Default is "phash".

    Returns
    -------
    (kept_df, removed_df) : Tuple[pandas.DataFrame, pandas.DataFrame]
        kept_df contains the first row for each unique pHash; removed_df contains subsequent duplicates.
    """

    if df is None or df.empty:
        return pd.DataFrame(), pd.DataFrame()

    if phash_col not in df.columns:
        raise ValueError(f"Missing required column '{phash_col}' in DataFrame")

    duplicate_mask = df[phash_col].duplicated(keep="first")
    removed_df = df[duplicate_mask].copy()
    kept_df = df[~duplicate_mask].copy()

    return kept_df, removed_df
