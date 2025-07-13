# San Diego Housing Market Analysis

## Project Overview
This is an end-to-end data analysis project examining the San Diego Airbnb market using comprehensive datasets including listings, calendar data, reviews, and neighborhood information.

## Data Sources
- **listings.csv.gz**: Detailed property listings with pricing, amenities, and location data
- **calendar.csv.gz**: Time-series data showing pricing trends and availability
- **reviews.csv.gz**: Customer reviews and sentiment data
- **neighbourhoods.csv**: Neighborhood information for geographic analysis
- **neighbourhoods.geojson**: GeoJSON file for mapping and spatial analysis

## Project Structure
```
san-diego-housing/
├── data/                   # Place your downloaded .gz files here
├── notebooks/              # Jupyter notebooks for analysis
├── src/                    # Python source code
├── results/                # Generated visualizations and reports
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Setup Instructions

1. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Download the data files** and place them in the `data/` directory:
   - listings.csv.gz
   - calendar.csv.gz
   - reviews.csv.gz
   - neighbourhoods.csv
   - neighbourhoods.geojson

3. **Start Jupyter notebook:**
   ```bash
   jupyter notebook
   ```

## Analysis Plan
1. **Data Exploration**: Understand the structure and quality of each dataset
2. **Data Cleaning**: Handle missing values, outliers, and data type conversions
3. **Exploratory Data Analysis**: Identify patterns in pricing, availability, and reviews
4. **Geographic Analysis**: Map neighborhoods and analyze spatial patterns
5. **Time Series Analysis**: Examine seasonal trends and pricing dynamics
6. **Sentiment Analysis**: Analyze customer reviews and satisfaction
7. **Market Insights**: Generate actionable business recommendations

## Key Questions to Answer
- What are the pricing trends across different neighborhoods?
- How does seasonality affect availability and pricing?
- Which amenities contribute most to higher ratings?
- What are the most profitable neighborhoods for hosts?
- How do customer reviews correlate with pricing and amenities? 