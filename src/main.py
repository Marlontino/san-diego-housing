#!/usr/bin/env python3
"""
San Diego Housing Market Analysis - Main Script
This script connects the data summary, visualization, and predictive modeling modules.
"""

import pandas as pd
import os
from data_summary import analyze_listings_data
from visualizations import create_market_analysis_plots, create_price_analysis_plots, create_geographic_heatmap
from predictive_model import run_predictive_analysis

def main():
    """Main function to run the complete data exploration analysis"""
    
    print("🏠 San Diego Housing Market Analysis")
    print("=" * 60)
    
    # Import libraries
    print("📦 Importing libraries...")
    print("✅ Libraries imported successfully!")
    
    # Load listings data
    print("\n📂 Loading listings data...")
    try:
        listings = pd.read_csv('data/listings.csv.gz', compression='gzip')
        print(f"✅ Loaded {len(listings):,} listings with {len(listings.columns)} columns")
    except FileNotFoundError:
        print("❌ listings.csv.gz not found in data/ directory")
        return
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # Run data summary analysis
    print("\n" + "="*60)
    print("📊 RUNNING DATA SUMMARY ANALYSIS")
    print("="*60)
    
    summary_stats = analyze_listings_data(listings)
    
    # Run visualizations
    print("\n" + "="*60)
    print("📈 CREATING VISUALIZATIONS")
    print("="*60)
    
    # Create main market analysis plots
    create_market_analysis_plots(listings)
    
    # Create additional price analysis plots
    create_price_analysis_plots(listings)
    
    # Create geographic heatmap
    create_geographic_heatmap(listings)
    
    # Run predictive modeling analysis
    print("="*60)
    
    model_results = run_predictive_analysis(listings)
    
    print(f"\n🎉 Analysis complete!")
    print(f"📁 Results saved in 'results/' directory")
    print(f"📊 Summary statistics: {len(summary_stats)} metrics calculated")
    print(f"🤖 Model results: {len(model_results)} model types trained")

if __name__ == "__main__":
    main()
