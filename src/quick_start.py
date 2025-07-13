#!/usr/bin/env python3
"""
Quick Start Script for San Diego Housing Market Analysis
Run this script to quickly load and explore your data
"""

import sys
import os
sys.path.append('src')

from data_loader import SanDiegoDataLoader
import pandas as pd

def main():
    print("🏠 San Diego Housing Market Analysis - Quick Start")
    print("=" * 60)
    
    # Initialize data loader
    loader = SanDiegoDataLoader()
    
    # Load all data
    print("\n📂 Loading datasets...")
    data = loader.load_all_data()
    
    # Get summary
    print("\n📊 Dataset Summary:")
    print("-" * 40)
    summary = loader.get_data_summary()
    print(summary.to_string(index=False))
    
    # Quick exploration of listings data
    if 'listings' in data:
        print(f"\n🏠 Listings Data Preview:")
        print("-" * 40)
        listings = data['listings']
        print(f"Total listings: {len(listings):,}")
        print(f"Columns: {list(listings.columns)}")
        
        # Show key statistics if price column exists
        if 'price' in listings.columns:
            print(f"\n💰 Price Statistics:")
            print("-" * 20)
            # Clean price column for analysis
            prices = listings['price'].str.replace('$', '').str.replace(',', '').astype(float)
            print(f"Average price: ${prices.mean():.2f}")
            print(f"Median price: ${prices.median():.2f}")
            print(f"Min price: ${prices.min():.2f}")
            print(f"Max price: ${prices.max():.2f}")
    
    # Quick exploration of calendar data
    if 'calendar' in data:
        print(f"\n📅 Calendar Data Preview:")
        print("-" * 40)
        calendar = data['calendar']
        print(f"Total calendar entries: {len(calendar):,}")
        print(f"Date range: {calendar['date'].min()} to {calendar['date'].max()}")
        
        # Show availability statistics
        if 'available' in calendar.columns:
            availability = calendar['available'].value_counts()
            print(f"Availability: {availability.get('t', 0):,} available, {availability.get('f', 0):,} unavailable")
    
    # Quick exploration of reviews data
    if 'reviews' in data:
        print(f"\n💬 Reviews Data Preview:")
        print("-" * 40)
        reviews = data['reviews']
        print(f"Total reviews: {len(reviews):,}")
        print(f"Date range: {reviews['date'].min()} to {reviews['date'].max()}")
    
    print(f"\n✅ Data loading complete!")
    print(f"\nNext steps:")
    print(f"1. Run: jupyter notebook")
    print(f"2. Open: notebooks/01_data_exploration.ipynb")
    print(f"3. Start your analysis!")

if __name__ == "__main__":
    main() 