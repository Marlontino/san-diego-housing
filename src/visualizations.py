#!/usr/bin/env python3
"""
San Diego Housing Market Analysis - Visualizations
This module contains functions for creating visualizations of the Airbnb data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
warnings.filterwarnings('ignore')

def create_market_analysis_plots(listings):
    """
    Create comprehensive visualizations for the San Diego Airbnb market analysis.
    
    Args:
        listings (pd.DataFrame): The listings dataframe with price_clean column
    """
    
    # Use clean data if available, otherwise use original data
    if hasattr(listings, 'clean_listings'):
        clean_listings = listings.clean_listings
        print("📊 Using cleaned data for visualizations")
    else:
        clean_listings = listings
        print("📊 Using original data for visualizations")
    
    try:
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create a figure with multiple subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('San Diego Airbnb Market Analysis', fontsize=16, fontweight='bold')
        
        if 'price_clean' in clean_listings.columns:
            # 1. Price distribution (using clean data)
            price_data = clean_listings['price_clean']
            axes[0, 0].hist(price_data, bins=50, alpha=0.7, color='#1f77b4', edgecolor='black')
            axes[0, 0].set_title('Price Distribution (Clean Data)')
            axes[0, 0].set_xlabel('Price ($)')
            axes[0, 0].set_ylabel('Number of Listings')
        
        if 'room_type' in listings.columns:
            # 2. Room type distribution (including all room types)
            room_counts = listings['room_type'].value_counts()
            axes[0, 1].pie(room_counts.values, labels=room_counts.index, autopct='%1.1f%%')
            axes[0, 1].set_title('Distribution by Room Type (All Types)')
        
        if 'price_clean' in clean_listings.columns and 'room_type' in clean_listings.columns:
            # 3. Average price by room type (using clean data - hotel rooms already excluded)
            room_prices = clean_listings.groupby('room_type')['price_clean'].mean()
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
            axes[1, 0].bar(room_prices.index, room_prices.values, color=colors[:len(room_prices)])
            axes[1, 0].set_title('Average Price by Room Type (Clean Data)')
            axes[1, 0].set_ylabel('Average Price ($)')
            axes[1, 0].tick_params(axis='x', rotation=45)
        
        if 'neighbourhood_cleansed' in clean_listings.columns:
            # 4. Top 10 neighborhoods by number of listings
            top_neighborhoods = clean_listings['neighbourhood_cleansed'].value_counts().head(10)
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9']
            axes[1, 1].barh(range(len(top_neighborhoods)), top_neighborhoods.values, color=colors[:len(top_neighborhoods)])
            axes[1, 1].set_yticks(range(len(top_neighborhoods)))
            axes[1, 1].set_yticklabels(top_neighborhoods.index)
            axes[1, 1].set_title('Top 10 Neighborhoods by Listings (Clean Data)')
            axes[1, 1].set_xlabel('Number of Listings')
        
        plt.tight_layout()
        
        # Ensure results directory exists
        os.makedirs('results', exist_ok=True)
        
        # Save the plot
        plt.savefig('results/market_analysis_plots.png', dpi=300, bbox_inches='tight')
        print("✅ Visualizations saved to results/market_analysis_plots.png")
        
        # Close the plot to free memory
        plt.close()
        
    except Exception as e:
        print(f"❌ Error creating visualizations: {e}")

def create_price_analysis_plots(listings):
    """
    Create additional price-focused visualizations.
    
    Args:
        listings (pd.DataFrame): The listings dataframe with price_clean column
    """
    
    # Use clean data if available, otherwise use original data
    if hasattr(listings, 'clean_listings'):
        clean_listings = listings.clean_listings
        print("📊 Using cleaned data for price analysis plots")
    else:
        clean_listings = listings
        print("📊 Using original data for price analysis plots")
    
    if 'price_clean' not in clean_listings.columns:
        print("❌ Price data not available for visualization")
        return
    
    try:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Price Analysis Visualizations', fontsize=16, fontweight='bold')
        
        # 1. Correlation heatmap
        # Select numeric columns for correlation
        numeric_cols = ['price_clean', 'accommodates', 'bathrooms', 'bedrooms', 'beds', 
                       'number_of_reviews', 'review_scores_rating', 'availability_365']
        
        # Filter columns that exist in the dataset
        available_cols = [col for col in numeric_cols if col in listings.columns]
        correlation_data = listings[available_cols].corr()
        
        # Create heatmap
        im = axes[0, 0].imshow(correlation_data.values, cmap='coolwarm', aspect='auto')
        axes[0, 0].set_title('Correlation Matrix of Key Variables')
        axes[0, 0].set_xticks(range(len(correlation_data.columns)))
        axes[0, 0].set_yticks(range(len(correlation_data.columns)))
        axes[0, 0].set_xticklabels(correlation_data.columns, rotation=45, ha='right')
        axes[0, 0].set_yticklabels(correlation_data.columns)
        
        # Add correlation values as text
        for i in range(len(correlation_data.columns)):
            for j in range(len(correlation_data.columns)):
                text = axes[0, 0].text(j, i, f'{correlation_data.iloc[i, j]:.2f}',
                                       ha="center", va="center", color="black", fontsize=8)
        
        # Add colorbar
        plt.colorbar(im, ax=axes[0, 0], shrink=0.8)
        
        # 2. Box plot of prices by room type (including all room types)
        if 'room_type' in listings.columns:
            # Filter to reasonable price range for box plot
            room_prices = listings[listings['price_clean'] <= 1000]
            axes[0, 1].boxplot([room_prices[room_prices['room_type'] == rt]['price_clean'] 
                               for rt in room_prices['room_type'].unique()], 
                              labels=room_prices['room_type'].unique())
            axes[0, 1].set_title('Price Distribution by Room Type (All Types)')
            axes[0, 1].set_ylabel('Price ($)')
            axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Average price by neighborhood (top 15)
        if 'neighbourhood_cleansed' in clean_listings.columns:
            neighborhood_prices = clean_listings.groupby('neighbourhood_cleansed')['price_clean'].mean().sort_values(ascending=False).head(15)
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9', '#F1948A', '#85C1E9', '#D7BDE2', '#A9CCE3', '#FAD7A0']
            axes[1, 0].barh(range(len(neighborhood_prices)), neighborhood_prices.values, color=colors[:len(neighborhood_prices)])
            axes[1, 0].set_yticks(range(len(neighborhood_prices)))
            axes[1, 0].set_yticklabels(neighborhood_prices.index)
            axes[1, 0].set_title('Average Price by Neighborhood (Top 15) - Clean Data')
            axes[1, 0].set_xlabel('Average Price ($)')
        
        # 4. Price vs Rating scatter plot
        if 'review_scores_rating' in clean_listings.columns:
            valid_data = clean_listings.dropna(subset=['price_clean', 'review_scores_rating'])
            scatter = axes[1, 1].scatter(valid_data['price_clean'], valid_data['review_scores_rating'], 
                                        alpha=0.6, c=valid_data['price_clean'], cmap='viridis')
            axes[1, 1].set_xlabel('Price ($)')
            axes[1, 1].set_ylabel('Rating')
            axes[1, 1].set_title('Price vs Rating (Clean Data)')
            axes[1, 1].set_xlim(0, 2000)  # Updated limit for clean data range
            plt.colorbar(scatter, ax=axes[1, 1], label='Price ($)')
        
        plt.tight_layout()
        
        # Save the plot
        plt.savefig('results/price_analysis_plots.png', dpi=300, bbox_inches='tight')
        print("✅ Price analysis plots saved to results/price_analysis_plots.png")
        
        # Close the plot to free memory
        plt.close()
        
    except Exception as e:
        print(f"❌ Error creating price analysis plots: {e}")

def create_executive_summary_dashboard(listings):
    """
    Create a comprehensive executive summary dashboard with key metrics and insights.
    
    Args:
        listings (pd.DataFrame): The listings dataframe with price_clean column
    """
    
    # Use clean data if available, otherwise use original data
    if hasattr(listings, 'clean_listings'):
        clean_listings = listings.clean_listings
        print("📊 Creating executive summary dashboard with clean data")
    else:
        clean_listings = listings
        print("📊 Creating executive summary dashboard with original data")
    
    if 'price_clean' not in clean_listings.columns:
        print("❌ Price data not available for dashboard")
        return
    
    try:
        # Create a large figure with multiple subplots
        fig = plt.figure(figsize=(20, 16))
        fig.suptitle('SAN DIEGO AIRBNB MARKET - EXECUTIVE SUMMARY DASHBOARD', 
                     fontsize=20, fontweight='bold', y=0.98)
        
        # Create grid layout
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
        
        # 1. Market Overview - Large text box
        ax1 = fig.add_subplot(gs[0, :2])
        ax1.axis('off')
        
        # Calculate key metrics
        total_listings = len(listings)
        clean_listings_count = len(clean_listings)
        avg_price = clean_listings['price_clean'].mean()
        median_price = clean_listings['price_clean'].median()
        avg_rating = clean_listings['review_scores_rating'].mean()
        
        overview_text = f"""MARKET OVERVIEW
        
Total Listings: {total_listings:,}
Clean Dataset: {clean_listings_count:,}
Market Size: {clean_listings_count:,} active properties

PRICING INSIGHTS
Average Price: ${avg_price:.0f}
Median Price: ${median_price:.0f}
Price Range: ${clean_listings['price_clean'].min():.0f} - ${clean_listings['price_clean'].max():.0f}
75th Percentile: ${clean_listings['price_clean'].quantile(0.75):.0f}

QUALITY METRICS
Average Rating: {avg_rating:.2f}/5.0
Listings with Reviews: {clean_listings['review_scores_rating'].notna().sum():,}
Avg Availability: {clean_listings['availability_365'].mean():.0f} days/year"""
        
        ax1.text(0.05, 0.95, overview_text, transform=ax1.transAxes, fontsize=12,
                verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
        
        # 2. Room Type Distribution - Pie chart
        ax2 = fig.add_subplot(gs[0, 2:])
        room_counts = clean_listings['room_type'].value_counts()
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        pie_result = ax2.pie(room_counts.values, labels=room_counts.index, 
                            autopct='%1.1f%%', colors=colors[:len(room_counts)])
        wedges, texts, autotexts = pie_result
        ax2.set_title('Market Composition by Room Type', fontweight='bold')
        
        # 3. Price Distribution - Histogram
        ax3 = fig.add_subplot(gs[1, :2])
        ax3.hist(clean_listings['price_clean'], bins=50, alpha=0.7, color='#4ECDC4', edgecolor='black')
        ax3.set_title('Price Distribution (Clean Data)', fontweight='bold')
        ax3.set_xlabel('Price ($)')
        ax3.set_ylabel('Number of Listings')
        ax3.axvline(avg_price, color='red', linestyle='--', label=f'Mean: ${avg_price:.0f}')
        ax3.axvline(median_price, color='orange', linestyle='--', label=f'Median: ${median_price:.0f}')
        ax3.legend()
        
        # 4. Top Neighborhoods - Horizontal bar chart
        ax4 = fig.add_subplot(gs[1, 2:])
        top_neighborhoods = clean_listings['neighbourhood_cleansed'].value_counts().head(8)
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8', '#F7DC6F']
        bars = ax4.barh(range(len(top_neighborhoods)), top_neighborhoods.values, color=colors[:len(top_neighborhoods)])
        ax4.set_yticks(range(len(top_neighborhoods)))
        ax4.set_yticklabels(top_neighborhoods.index)
        ax4.set_title('Top 8 Neighborhoods by Listings', fontweight='bold')
        ax4.set_xlabel('Number of Listings')
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax4.text(width + 10, bar.get_y() + bar.get_height()/2, f'{int(width):,}',
                    ha='left', va='center', fontweight='bold')
        
        # 5. Price vs Rating Scatter
        ax5 = fig.add_subplot(gs[2, :2])
        valid_data = clean_listings.dropna(subset=['price_clean', 'review_scores_rating'])
        scatter = ax5.scatter(valid_data['price_clean'], valid_data['review_scores_rating'], 
                             alpha=0.6, c=valid_data['price_clean'], cmap='viridis')
        ax5.set_xlabel('Price ($)')
        ax5.set_ylabel('Rating')
        ax5.set_title('Price vs Rating Correlation', fontweight='bold')
        ax5.set_xlim(0, 2000)
        plt.colorbar(scatter, ax=ax5, label='Price ($)')
        
        # 6. Average Price by Room Type
        ax6 = fig.add_subplot(gs[2, 2:])
        room_prices = clean_listings.groupby('room_type')['price_clean'].mean()
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        bars = ax6.bar(room_prices.index, room_prices.values, color=colors[:len(room_prices)])
        ax6.set_title('Average Price by Room Type', fontweight='bold')
        ax6.set_ylabel('Average Price ($)')
        ax6.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height + 10,
                    f'${height:.0f}', ha='center', va='bottom', fontweight='bold')
        
        # 7. Key Findings - Text box
        ax7 = fig.add_subplot(gs[3, :2])
        ax7.axis('off')
        
        findings_text = """KEY FINDINGS

1. Entire homes/apartments dominate (84.2%)
2. High availability = premium pricing
3. Higher ratings correlate with lower prices
4. Accommodates has strongest price correlation
5. Most listings priced under $400

BUSINESS RECOMMENDATIONS

1. Target entire homes/apartments
2. Price competitively ($200-400 range)
3. Focus on 4+ guest capacity
4. Consider year-round availability
5. Target high-rating, moderate-price areas"""
        
        ax7.text(0.05, 0.95, findings_text, transform=ax7.transAxes, fontsize=11,
                verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8))
        
        # 8. Market Trends - Line chart
        ax8 = fig.add_subplot(gs[3, 2:])
        
        # Create price ranges and count listings in each
        price_ranges = [0, 100, 200, 300, 400, 500, 1000, 2000]
        range_labels = ['$0-100', '$100-200', '$200-300', '$300-400', '$400-500', '$500-1000', '$1000+']
        range_counts = []
        
        for i in range(len(price_ranges)-1):
            count = len(clean_listings[(clean_listings['price_clean'] >= price_ranges[i]) & 
                                     (clean_listings['price_clean'] < price_ranges[i+1])])
            range_counts.append(count)
        
        ax8.plot(range_labels, range_counts, marker='o', linewidth=2, markersize=8, color='#FF6B6B')
        ax8.set_title('Market Distribution by Price Range', fontweight='bold')
        ax8.set_xlabel('Price Range')
        ax8.set_ylabel('Number of Listings')
        ax8.tick_params(axis='x', rotation=45)
        
        # Add value labels on points
        for i, (label, count) in enumerate(zip(range_labels, range_counts)):
            ax8.text(i, count + 20, f'{count:,}', ha='center', va='bottom', fontweight='bold')
        
        # Ensure results directory exists
        os.makedirs('results', exist_ok=True)
        
        # Save the dashboard
        plt.savefig('results/executive_summary_dashboard.png', dpi=300, bbox_inches='tight')
        print("✅ Executive summary dashboard saved to results/executive_summary_dashboard.png")
        
        # Close the plot to free memory
        plt.close()
        
    except Exception as e:
        print(f"❌ Error creating executive summary dashboard: {e}")

def create_geographic_heatmap(listings):
    """
    Create a geographic heatmap showing price distribution across San Diego.
    
    Args:
        listings (pd.DataFrame): The listings dataframe with price_clean column
    """
    
    # Use clean data if available, otherwise use original data
    if hasattr(listings, 'clean_listings'):
        clean_listings = listings.clean_listings
        print("🗺️ Creating geographic heatmap with clean data")
    else:
        clean_listings = listings
        print("🗺️ Creating geographic heatmap with original data")
    
    if 'price_clean' not in clean_listings.columns or 'latitude' not in clean_listings.columns or 'longitude' not in clean_listings.columns:
        print("❌ Geographic data not available for heatmap")
        return
    
    try:
        import folium
        from folium import plugins
        import branca.colormap as cm
        
        print("🗺️ GEOGRAPHIC ANALYSIS")
        print("=" * 40)
        
        # Create a map centered on San Diego
        sd_map = folium.Map(location=[32.7157, -117.1611], zoom_start=11)
        
        # Add price heatmap
        price_data = clean_listings[['latitude', 'longitude', 'price_clean']].dropna()
        price_data = price_data.values.tolist()
        
        # Create heatmap
        plugins.HeatMap(price_data, radius=15).add_to(sd_map)
        
        # Create a color map for the legend
        min_price = clean_listings['price_clean'].min()
        max_price = clean_listings['price_clean'].max()
        
        # Create a linear color map
        colormap = cm.LinearColormap(
            colors=['blue', 'green', 'yellow', 'orange', 'red'],
            vmin=min_price,
            vmax=max_price,
            caption='Price Range ($)'
        )
        
        # Add the color map to the map
        colormap.add_to(sd_map)
        
        # Ensure results directory exists
        os.makedirs('results', exist_ok=True)
        
        # Save the map
        sd_map.save('results/san_diego_price_heatmap.html')
        print("✅ Price heatmap with legend saved as results/san_diego_price_heatmap.html")
        
        # Neighborhood clustering
        neighborhood_coords = clean_listings.groupby('neighbourhood_cleansed').agg({
            'latitude': 'mean',
            'longitude': 'mean',
            'price_clean': 'mean',
            'review_scores_rating': 'mean'
        }).dropna()
        
        print(f"\nNeighborhood coordinates calculated for {len(neighborhood_coords)} neighborhoods")
        print(f"Price range: ${min_price:.0f} - ${max_price:.0f}")
        
        # Create a second map with neighborhood markers
        neighborhood_map = folium.Map(location=[32.7157, -117.1611], zoom_start=11)
        
        # Add neighborhood markers
        for idx, row in neighborhood_coords.iterrows():
            # Create popup content
            popup_content = f"""
            <b>{idx}</b><br>
            Average Price: ${row['price_clean']:.0f}<br>
            Average Rating: {row['review_scores_rating']:.2f}/5.0
            """
            
            # Color marker based on price
            if row['price_clean'] < 200:
                color = 'green'
            elif row['price_clean'] < 400:
                color = 'yellow'
            elif row['price_clean'] < 600:
                color = 'orange'
            else:
                color = 'red'
            
            folium.Marker(
                location=[row['latitude'], row['longitude']],
                popup=folium.Popup(popup_content, max_width=300),
                tooltip=f"{idx}: ${row['price_clean']:.0f}",
                icon=folium.Icon(color=color, icon='info-sign')
            ).add_to(neighborhood_map)
        
        # Add legend to the neighborhood map using a simpler approach
        legend_html = '''
        <div style="position: fixed; 
                    top: 10px; right: 10px; width: 250px; height: 250px; 
                    background-color: white; border:3px solid grey; z-index:9999; 
                    font-size:14px; padding: 15px; border-radius: 5px; box-shadow: 2px 2px 5px rgba(0,0,0,0.3)">
        <p><b>Neighborhood Price Legend</b></p>
        <p><i class="fa fa-map-marker fa-2x" style="color:green"></i> Under $200</p>
        <p><i class="fa fa-map-marker fa-2x" style="color:yellow"></i> $200-$399</p>
        <p><i class="fa fa-map-marker fa-2x" style="color:orange"></i> $400-$599</p>
        <p><i class="fa fa-map-marker fa-2x" style="color:red"></i> $600+</p>
        </div>
        '''
        # Add legend directly to the map
        neighborhood_map.get_root().html.add_child(folium.Element(legend_html))
        
        # Save the neighborhood map
        neighborhood_map.save('results/san_diego_neighborhood_map.html')
        print("✅ Neighborhood map with legend saved as results/san_diego_neighborhood_map.html")
        
    except ImportError:
        print("❌ Folium not available. Install with: pip install folium")
    except Exception as e:
        print(f"❌ Error creating geographic heatmap: {e}") 