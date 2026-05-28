"""
San Diego Housing — Streamlit Dashboard

Run locally:
    streamlit run src/dashboard.py

Deploy free at https://streamlit.io/cloud by pointing it at this file.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from preprocessing import preprocess, load_raw

st.set_page_config(page_title="San Diego Housing", layout="wide")


@st.cache_data(show_spinner="Cleaning San Diego listings...")
def get_clean_data() -> pd.DataFrame:
    df, _ = preprocess(verbose=False)
    return df


df = get_clean_data()

st.title("San Diego Short-Term Rental Explorer")
st.caption(
    f"{len(df):,} listings after cleaning (missing-data drops, per-neighborhood "
    f"IQR outlier filter, categorical normalization). See `src/preprocessing.py`."
)

# --- Sidebar filters -----------------------------------------------------
st.sidebar.header("Filters")

neighborhoods = sorted(df["neighbourhood_cleansed"].unique())
picked_neighborhoods = st.sidebar.multiselect(
    "Neighborhood",
    neighborhoods,
    default=neighborhoods[:5],
    help="Choose one or more San Diego neighborhoods.",
)

price_min = int(df["price"].min())
price_slider_cap = 1000
price_range = st.sidebar.slider(
    "Nightly price ($)",
    min_value=price_min,
    max_value=price_slider_cap,
    value=(price_min, min(price_slider_cap, 500)),
    step=10,
    help=f"Listings above ${price_slider_cap} are excluded from the slider range.",
)

room_types = sorted(df["room_type"].unique())
picked_rooms = st.sidebar.multiselect(
    "Room type", room_types, default=room_types
)

min_bedrooms = st.sidebar.slider(
    "Minimum bedrooms",
    min_value=0,
    max_value=int(df["bedrooms"].max()),
    value=0,
)

# --- Apply filters -------------------------------------------------------
mask = (
    df["neighbourhood_cleansed"].isin(picked_neighborhoods)
    & df["price"].between(*price_range)
    & df["room_type"].isin(picked_rooms)
    & (df["bedrooms"] >= min_bedrooms)
)
view = df[mask]

# --- KPI row -------------------------------------------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Listings", f"{len(view):,}")
c2.metric("Median price", f"${view['price'].median():,.0f}" if len(view) else "—")
c3.metric("Mean price", f"${view['price'].mean():,.0f}" if len(view) else "—")
c4.metric(
    "Avg review score",
    f"{view['review_scores_rating'].mean():.2f}" if len(view) else "—",
)

if len(view) == 0:
    st.warning("No listings match these filters — widen the price range or pick more neighborhoods.")
    st.stop()

# --- Charts --------------------------------------------------------------
left, right = st.columns(2)

with left:
    st.subheader("Median price by neighborhood")
    by_nbhd = (
        view.groupby("neighbourhood_cleansed")["price"]
        .median()
        .sort_values(ascending=False)
        .rename("Median nightly price ($)")
        .reset_index()
        .rename(columns={"neighbourhood_cleansed": "Neighborhood"})
    )
    st.bar_chart(
        by_nbhd,
        x="Neighborhood",
        y="Median nightly price ($)",
        x_label="Neighborhood",
        y_label="Median nightly price ($)",
    )

with right:
    st.subheader("Price distribution")
    bins = pd.cut(view["price"], bins=30)
    hist = (
        view.assign(_bin_mid=bins.apply(lambda iv: round(float(iv.mid), 2)))
        .groupby("_bin_mid")
        .size()
        .rename("Listings")
        .reset_index()
        .rename(columns={"_bin_mid": "Nightly price ($)"})
    )
    st.bar_chart(
        hist,
        x="Nightly price ($)",
        y="Listings",
        x_label="Nightly price ($)",
        y_label="Number of listings",
    )

st.subheader("Listings map")
st.map(view.rename(columns={"latitude": "lat", "longitude": "lon"})[["lat", "lon"]])

with st.expander("Sample of filtered listings"):
    st.dataframe(
        view[
            [
                "neighbourhood_cleansed",
                "room_type",
                "property_type",
                "bedrooms",
                "price",
                "review_scores_rating",
            ]
        ].head(200)
    )
