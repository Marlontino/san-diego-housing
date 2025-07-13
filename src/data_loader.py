"""
Data loading utilities for San Diego Housing Market Analysis
"""

import pandas as pd
import numpy as np
import geopandas as gpd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class SanDiegoDataLoader:
    """
    A class to load and perform basic processing on San Diego Airbnb datasets
    """
    
    def __init__(self, data_path='./data'):
        """
        Initialize the data loader
        
        Parameters:
        -----------
        data_path : str
            Path to the data directory containing the .gz files
        """
        self.data_path = Path(data_path)
        self.listings_df = None
        self.calendar_df = None
        self.reviews_df = None
        self.neighborhoods_df = None
        self.neighborhoods_gdf = None
        
    def load_all_data(self):
        """
        Load all available datasets
        
        Returns:
        --------
        dict : Dictionary containing all loaded dataframes
        """
        data_dict = {}
        
        # Load listings data
        try:
            self.listings_df = pd.read_csv(self.data_path / 'listings.csv.gz', compression='gzip')
            data_dict['listings'] = self.listings_df
            print(f"✅ Loaded listings data: {self.listings_df.shape[0]:,} rows, {self.listings_df.shape[1]} columns")
        except FileNotFoundError:
            print("❌ listings.csv.gz not found")
        except Exception as e:
            print(f"❌ Error loading listings data: {e}")
            
        # Load calendar data
        try:
            self.calendar_df = pd.read_csv(self.data_path / 'calendar.csv.gz', compression='gzip')
            data_dict['calendar'] = self.calendar_df
            print(f"✅ Loaded calendar data: {self.calendar_df.shape[0]:,} rows, {self.calendar_df.shape[1]} columns")
        except FileNotFoundError:
            print("❌ calendar.csv.gz not found")
        except Exception as e:
            print(f"❌ Error loading calendar data: {e}")
            
        # Load reviews data
        try:
            self.reviews_df = pd.read_csv(self.data_path / 'reviews.csv.gz', compression='gzip')
            data_dict['reviews'] = self.reviews_df
            print(f"✅ Loaded reviews data: {self.reviews_df.shape[0]:,} rows, {self.reviews_df.shape[1]} columns")
        except FileNotFoundError:
            print("❌ reviews.csv.gz not found")
        except Exception as e:
            print(f"❌ Error loading reviews data: {e}")
            
        # Load neighborhoods data
        try:
            self.neighborhoods_df = pd.read_csv(self.data_path / 'neighbourhoods.csv')
            data_dict['neighborhoods'] = self.neighborhoods_df
            print(f"✅ Loaded neighborhoods data: {self.neighborhoods_df.shape[0]:,} rows, {self.neighborhoods_df.shape[1]} columns")
        except FileNotFoundError:
            print("❌ neighbourhoods.csv not found")
        except Exception as e:
            print(f"❌ Error loading neighborhoods data: {e}")
            
        # Load neighborhoods geojson
        try:
            self.neighborhoods_gdf = gpd.read_file(self.data_path / 'neighbourhoods.geojson')
            data_dict['neighborhoods_geo'] = self.neighborhoods_gdf
            print(f"✅ Loaded neighborhoods geojson: {self.neighborhoods_gdf.shape[0]:,} rows, {self.neighborhoods_gdf.shape[1]} columns")
        except FileNotFoundError:
            print("❌ neighbourhoods.geojson not found")
        except Exception as e:
            print(f"❌ Error loading neighborhoods geojson: {e}")
            
        return data_dict
    
    def get_data_summary(self):
        """
        Get a summary of all loaded datasets
        
        Returns:
        --------
        pd.DataFrame : Summary statistics for all datasets
        """
        datasets = {
            'Listings': self.listings_df,
            'Calendar': self.calendar_df,
            'Reviews': self.reviews_df,
            'Neighborhoods': self.neighborhoods_df,
            'Neighborhoods_Geo': self.neighborhoods_gdf
        }
        
        summary_data = []
        for name, df in datasets.items():
            if df is not None:
                summary_data.append({
                    'Dataset': name,
                    'Rows': df.shape[0],
                    'Columns': df.shape[1],
                    'Memory_MB': df.memory_usage(deep=True).sum() / 1024**2,
                    'Missing_Values': df.isnull().sum().sum(),
                    'Duplicate_Rows': df.duplicated().sum()
                })
        
        return pd.DataFrame(summary_data)
    
    def clean_listings_data(self):
        """
        Perform basic cleaning on listings data
        
        Returns:
        --------
        pd.DataFrame : Cleaned listings dataframe
        """
        if self.listings_df is None:
            print("❌ No listings data loaded")
            return None
            
        df = self.listings_df.copy()
        
        # Convert price column to numeric
        if 'price' in df.columns:
            df['price'] = df['price'].str.replace('$', '').str.replace(',', '').astype(float)
            
        # Convert date columns
        date_columns = ['host_since', 'first_review', 'last_review']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
                
        # Convert numeric columns
        numeric_columns = ['host_response_rate', 'host_acceptance_rate']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = df[col].str.replace('%', '').astype(float) / 100
                
        return df
    
    def clean_calendar_data(self):
        """
        Perform basic cleaning on calendar data
        
        Returns:
        --------
        pd.DataFrame : Cleaned calendar dataframe
        """
        if self.calendar_df is None:
            print("❌ No calendar data loaded")
            return None
            
        df = self.calendar_df.copy()
        
        # Convert date column
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            
        # Convert price column to numeric
        if 'price' in df.columns:
            df['price'] = df['price'].str.replace('$', '').str.replace(',', '').astype(float)
            
        # Convert available column to boolean
        if 'available' in df.columns:
            df['available'] = df['available'] == 't'
            
        return df
    
    def clean_reviews_data(self):
        """
        Perform basic cleaning on reviews data
        
        Returns:
        --------
        pd.DataFrame : Cleaned reviews dataframe
        """
        if self.reviews_df is None:
            print("❌ No reviews data loaded")
            return None
            
        df = self.reviews_df.copy()
        
        # Convert date column
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            
        return df

def load_sample_data():
    """
    Load a sample of data for quick testing
    
    Returns:
    --------
    dict : Dictionary containing sample dataframes
    """
    loader = SanDiegoDataLoader()
    data_dict = loader.load_all_data()
    
    # Take samples for quick testing
    sample_dict = {}
    for name, df in data_dict.items():
        if df is not None:
            sample_dict[name] = df.head(1000)  # First 1000 rows
            print(f"📊 {name}: Loaded sample of {len(sample_dict[name])} rows")
    
    return sample_dict

if __name__ == "__main__":
    # Example usage
    loader = SanDiegoDataLoader()
    data = loader.load_all_data()
    summary = loader.get_data_summary()
    print("\n📈 DATA SUMMARY")
    print("=" * 50)
    print(summary) 