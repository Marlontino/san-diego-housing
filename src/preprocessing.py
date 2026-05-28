"""
San Diego Housing — Data Preprocessing

This module is the single source of truth for cleaning the raw Inside Airbnb
San Diego listings dataset before any analysis or dashboarding.

It is intentionally explicit and verbose so a reviewer can see exactly what
was thrown away, what was imputed, and why. Each step prints a short report
to stdout so the cleaning pipeline is auditable end-to-end.

Pipeline overview
-----------------
1. Load raw listings.csv.gz
2. Parse messy column types (price strings -> floats, "t"/"f" -> bool,
   "65%" -> 0.65, dates -> datetime)
3. Drop rows with missing values in fields we cannot impute (price,
   neighborhood, lat/lon, room_type)
4. Impute missing values in fields we can defend imputing (bedrooms, beds,
   bathrooms -> median by property_type; review scores -> column median)
5. Filter statistical outliers in price using the IQR rule, per
   neighborhood, so a $50M La Jolla mansion does not drag the San Diego
   median upward
6. Normalize categorical fields (lowercase, trim, collapse rare property
   types into "Other")
7. Return a tidy dataframe and a report dict for the dashboard
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "listings.csv.gz"

ESSENTIAL_COLUMNS = [
    "id",
    "price",
    "neighbourhood_cleansed",
    "latitude",
    "longitude",
    "room_type",
    "property_type",
    "accommodates",
    "bedrooms",
    "beds",
    "bathrooms",
    "bathrooms_text",
    "minimum_nights",
    "number_of_reviews",
    "review_scores_rating",
    "review_scores_cleanliness",
    "review_scores_location",
    "availability_365",
    "host_is_superhost",
    "instant_bookable",
]


def _parse_price(series: pd.Series) -> pd.Series:
    """Convert '$1,234.00' style strings into floats. NaN-safe."""
    return (
        series.astype(str)
        .str.replace(r"[\$,]", "", regex=True)
        .replace({"nan": np.nan, "": np.nan})
        .astype(float)
    )


def _parse_bool(series: pd.Series) -> pd.Series:
    """Airbnb encodes booleans as 't'/'f'. Normalize to bool."""
    return series.map({"t": True, "f": False}).astype("boolean")


def _parse_percent(series: pd.Series) -> pd.Series:
    """'65%' -> 0.65"""
    return (
        series.astype(str)
        .str.replace("%", "", regex=False)
        .replace({"nan": np.nan, "": np.nan})
        .astype(float)
        / 100.0
    )


def _bathrooms_from_text(text: pd.Series) -> pd.Series:
    """
    bathrooms_text looks like '1 bath', '1.5 shared baths', 'Half-bath'.
    Pull the leading number; treat 'Half-bath' as 0.5.
    """
    out = (
        text.astype(str)
        .str.lower()
        .str.replace("half-bath", "0.5", regex=False)
        .str.extract(r"([0-9]*\.?[0-9]+)", expand=False)
        .astype(float)
    )
    return out


def load_raw(path: Path = DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path, compression="gzip", low_memory=False)


def preprocess(
    df: pd.DataFrame | None = None,
    *,
    iqr_multiplier: float = 1.5,
    rare_property_threshold: int = 50,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, dict]:
    """
    Run the full cleaning pipeline. Returns (clean_df, report).

    Parameters
    ----------
    df : optional preloaded dataframe. If None, loads from DATA_PATH.
    iqr_multiplier : how aggressive the IQR outlier filter is. 1.5 is the
        textbook default; raise to 3.0 to keep more high-end listings.
    rare_property_threshold : property_type values with fewer than this many
        listings get collapsed into "Other" so charts don't get a long tail
        of one-off categories.
    """
    report: dict = {}
    if df is None:
        df = load_raw()

    report["rows_raw"] = len(df)

    # --- 1. Type coercion -------------------------------------------------
    df = df[[c for c in ESSENTIAL_COLUMNS if c in df.columns]].copy()
    df["price"] = _parse_price(df["price"])
    if "host_is_superhost" in df:
        df["host_is_superhost"] = _parse_bool(df["host_is_superhost"])
    if "instant_bookable" in df:
        df["instant_bookable"] = _parse_bool(df["instant_bookable"])

    # Some scrapes ship bathrooms as all-NaN with the number hidden in
    # bathrooms_text. Reconstruct from text where the numeric column is null.
    if "bathrooms_text" in df:
        reconstructed = _bathrooms_from_text(df["bathrooms_text"])
        df["bathrooms"] = df["bathrooms"].fillna(reconstructed)

    # --- 2. Missing-data handling ----------------------------------------
    # 2a. Drop rows missing fields we refuse to fabricate.
    critical = ["price", "neighbourhood_cleansed", "latitude", "longitude", "room_type"]
    before = len(df)
    df = df.dropna(subset=critical)
    report["dropped_missing_critical"] = before - len(df)

    # 2b. Impute structural fields (bedrooms/beds/bathrooms) using the
    # median for the listing's property_type. Falls back to the global
    # median for property types with no non-null observations.
    for col in ["bedrooms", "beds", "bathrooms"]:
        if col not in df:
            continue
        global_median = df[col].median()
        df[col] = df.groupby("property_type")[col].transform(
            lambda s: s.fillna(s.median())
        )
        df[col] = df[col].fillna(global_median)

    # 2c. Review scores: legitimately missing for new listings with 0
    # reviews. Impute with column median so they stop poisoning aggregates,
    # but keep the listing in the dataset.
    review_cols = [c for c in df.columns if c.startswith("review_scores_")]
    for col in review_cols:
        df[col] = df[col].fillna(df[col].median())

    report["rows_after_imputation"] = len(df)

    # --- 3. Outlier filtering --------------------------------------------
    # Per-neighborhood IQR rule on price. A $50M mansion in La Jolla
    # shouldn't pull La Jolla's median up, and shouldn't pull all-of-SD up
    # either. We compute Q1/Q3 within each neighborhood and drop rows
    # outside [Q1 - k*IQR, Q3 + k*IQR].
    df["price"] = df["price"].clip(lower=0)  # negative prices are scrape errors
    before = len(df)

    def _iqr_mask(group: pd.Series) -> pd.Series:
        q1, q3 = group.quantile(0.25), group.quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - iqr_multiplier * iqr, q3 + iqr_multiplier * iqr
        return group.between(lo, hi)

    keep = df.groupby("neighbourhood_cleansed")["price"].transform(_iqr_mask)
    df = df[keep.fillna(False)]
    report["dropped_price_outliers"] = before - len(df)

    # Also drop listings priced at $0 — those are placeholders, not data.
    before = len(df)
    df = df[df["price"] > 0]
    report["dropped_zero_price"] = before - len(df)

    # --- 4. Categorical normalization ------------------------------------
    # Trim, title-case, and collapse rare property_type values into "Other"
    # so downstream filters and charts stay readable.
    df["neighbourhood_cleansed"] = (
        df["neighbourhood_cleansed"].astype(str).str.strip()
    )
    df["room_type"] = df["room_type"].astype(str).str.strip()
    df["property_type"] = df["property_type"].astype(str).str.strip()

    counts = df["property_type"].value_counts()
    rare = counts[counts < rare_property_threshold].index
    df.loc[df["property_type"].isin(rare), "property_type"] = "Other"
    report["property_types_collapsed_to_other"] = int(len(rare))

    # --- 5. Final summary -------------------------------------------------
    df = df.reset_index(drop=True)
    report["rows_final"] = len(df)
    report["price_median"] = float(df["price"].median())
    report["price_mean"] = float(df["price"].mean())
    report["neighborhood_count"] = int(df["neighbourhood_cleansed"].nunique())

    if verbose:
        print("=" * 60)
        print("PREPROCESSING REPORT")
        print("=" * 60)
        for k, v in report.items():
            print(f"  {k:<35} {v}")
        print("=" * 60)

    return df, report


if __name__ == "__main__":
    clean, _ = preprocess()
    out = Path(__file__).resolve().parent.parent / "data" / "listings_clean.parquet"
    try:
        clean.to_parquet(out, index=False)
        print(f"Wrote {len(clean):,} cleaned rows to {out}")
    except Exception:
        out = out.with_suffix(".csv")
        clean.to_csv(out, index=False)
        print(f"Wrote {len(clean):,} cleaned rows to {out}")
