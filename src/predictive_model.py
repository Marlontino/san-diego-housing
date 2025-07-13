#!/usr/bin/env python3
"""
San Diego Housing Market Analysis - Predictive Modeling
This module contains functions for training and evaluating predictive models.
"""

import pandas as pd
import numpy as np
import json
import warnings
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
warnings.filterwarnings('ignore')

def train_basic_model(clean_listings):
    """
    Train a basic predictive model for price prediction.
    
    Args:
        clean_listings (pd.DataFrame): Clean listings data
        
    Returns:
        dict: Model performance metrics and feature importance
    """
    # First, let's check what data we have
    print(f"Original clean_listings shape: {clean_listings.shape}")
    print(f"Missing values in key columns:")
    print(clean_listings[['price_clean', 'accommodates', 'bathrooms', 'bedrooms', 
                         'beds', 'number_of_reviews', 'review_scores_rating', 
                         'availability_365', 'room_type', 'neighbourhood_cleansed']].isnull().sum())

    # Create a complete dataset with no missing values
    complete_data = clean_listings[['price_clean', 'accommodates', 'bathrooms', 'bedrooms', 
                                   'beds', 'number_of_reviews', 'review_scores_rating', 
                                   'availability_365', 'room_type', 'neighbourhood_cleansed']].dropna()

    print(f"\nComplete dataset shape: {complete_data.shape}")

    # Create dummy variables for categorical features
    room_type_dummies = pd.get_dummies(complete_data['room_type'], prefix='room_type')
    neighborhood_dummies = pd.get_dummies(complete_data['neighbourhood_cleansed'], prefix='neighborhood')

    # Prepare numerical features
    numerical_features = complete_data[['accommodates', 'bathrooms', 'bedrooms', 
                                       'beds', 'number_of_reviews', 'review_scores_rating', 
                                       'availability_365']]

    # Combine all features
    X = pd.concat([numerical_features, room_type_dummies, neighborhood_dummies], axis=1)
    y = complete_data['price_clean']

    print(f"Feature matrix shape: {X.shape}")
    print(f"Target variable shape: {y.shape}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")

    # Train model
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)

    # Make predictions
    y_pred = rf_model.predict(X_test)

    # Evaluate model
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"\nModel Performance:")
    print(f"Mean Absolute Error: ${mae:.2f}")
    print(f"R² Score: {r2:.3f}")

    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)

    print(f"\nTop 15 most important features:")
    print(feature_importance.head(15))

    # Show feature importance by category
    print(f"\nFeature importance by category:")
    numerical_importance = feature_importance[feature_importance['feature'].isin(numerical_features.columns)]
    room_type_importance = feature_importance[feature_importance['feature'].str.startswith('room_type')]
    neighborhood_importance = feature_importance[feature_importance['feature'].str.startswith('neighborhood')]

    print(f"\nNumerical features:")
    print(numerical_importance)

    print(f"\nTop 5 room type features:")
    print(room_type_importance.head(5))

    print(f"\nTop 5 neighborhood features:")
    print(neighborhood_importance.head(5))

    # Create a simple prediction example
    print(f"\n📊 Sample Prediction:")
    sample_data = X_test.iloc[0:1]  # Take first test sample
    predicted_price = rf_model.predict(sample_data)[0]
    actual_price = y_test.iloc[0]

    print(f"Sample listing features:")
    for col in numerical_features.columns:
        print(f"  {col}: {sample_data[col].iloc[0]}")
    print(f"Predicted price: ${predicted_price:.2f}")
    print(f"Actual price: ${actual_price:.2f}")
    print(f"Difference: ${abs(predicted_price - actual_price):.2f}")

    # Save model results
    model_results = {
        'mae': float(mae),
        'r2_score': float(r2),
        'training_samples': int(X_train.shape[0]),
        'test_samples': int(X_test.shape[0]),
        'total_features': int(X.shape[1]),
        'top_features': feature_importance.head(10)['feature'].tolist()
    }

    # Ensure results directory exists
    import os
    os.makedirs('results', exist_ok=True)
    
    with open('results/model_performance.json', 'w') as f:
        json.dump(model_results, f, indent=2)

    print(f"\n✅ Model results saved to results/model_performance.json")
    
    return model_results

def train_full_model_with_1000_test(clean_listings):
    """
    Train a model on the full dataset with a 1000-entry test set.
    
    Args:
        clean_listings (pd.DataFrame): Clean listings data
        
    Returns:
        dict: Full model performance metrics
    """
    # Create a complete dataset with no missing values
    complete_data = clean_listings[['price_clean', 'accommodates', 'bathrooms', 'bedrooms', 
                                   'beds', 'number_of_reviews', 'review_scores_rating', 
                                   'availability_365', 'room_type', 'neighbourhood_cleansed']].dropna()

    # Use the complete dataset for training
    print(f"Full dataset size: {len(complete_data)} entries")

    # Create features for the full dataset
    full_room_dummies = pd.get_dummies(complete_data['room_type'], prefix='room_type')
    full_neighborhood_dummies = pd.get_dummies(complete_data['neighbourhood_cleansed'], prefix='neighborhood')
    full_numerical = complete_data[['accommodates', 'bathrooms', 'bedrooms', 
                                   'beds', 'number_of_reviews', 'review_scores_rating', 
                                   'availability_365']]

    # Combine features for full dataset
    X_full = pd.concat([full_numerical, full_room_dummies, full_neighborhood_dummies], axis=1)
    y_full = complete_data['price_clean']

    print(f"Full feature matrix shape: {X_full.shape}")
    print(f"Full target variable shape: {y_full.shape}")

    # Split: use 1000 entries for test, rest for training
    test_size = 1000
    X_train_full, X_test_1000, y_train_full, y_test_1000 = train_test_split(
        X_full, y_full, test_size=test_size, random_state=42, shuffle=True
    )

    print(f"Training set size: {X_train_full.shape[0]} entries")
    print(f"Test set size: {X_test_1000.shape[0]} entries")

    # Train model on full training set
    full_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    full_model.fit(X_train_full, y_train_full)

    # Make predictions on 1000-entry test set
    y_pred_1000 = full_model.predict(X_test_1000)

    # Calculate accuracy within $20
    absolute_differences = np.abs(y_pred_1000 - y_test_1000)
    within_20_dollars = absolute_differences <= 20
    accuracy_within_20 = np.mean(within_20_dollars) * 100

    print(f"\n📊 PREDICTION ACCURACY RESULTS:")
    print(f"Total predictions: {len(y_pred_1000)}")
    print(f"Predictions within $20: {np.sum(within_20_dollars)}")
    print(f"Accuracy within $20: {accuracy_within_20:.1f}%")

    # Detailed breakdown
    print(f"\n DETAILED BREAKDOWN:")
    print(f"Predictions within $10: {np.sum(absolute_differences <= 10)} ({(np.sum(absolute_differences <= 10) / len(y_pred_1000) * 100):.1f}%)")
    print(f"Predictions within $20: {np.sum(absolute_differences <= 20)} ({(np.sum(absolute_differences <= 20) / len(y_pred_1000) * 100):.1f}%)")
    print(f"Predictions within $50: {np.sum(absolute_differences <= 50)} ({(np.sum(absolute_differences <= 50) / len(y_pred_1000) * 100):.1f}%)")
    print(f"Predictions within $100: {np.sum(absolute_differences <= 100)} ({(np.sum(absolute_differences <= 100) / len(y_pred_1000) * 100):.1f}%)")

    # Show some examples
    print(f"\n SAMPLE PREDICTIONS:")
    test_results = pd.DataFrame({
        'Actual_Price': y_test_1000,
        'Predicted_Price': y_pred_1000,
        'Difference': absolute_differences,
        'Within_20': within_20_dollars
    })

    # Show 10 examples
    print(test_results.head(10).round(2))

    # Show accuracy by price range
    print(f"\n📊 ACCURACY BY PRICE RANGE:")
    test_results['Price_Range'] = pd.cut(test_results['Actual_Price'], 
                                        bins=[0, 100, 200, 300, 500, 1000, 2000], 
                                        labels=['$0-100', '$100-200', '$200-300', '$300-500', '$500-1000', '$1000+'])

    accuracy_by_range = test_results.groupby('Price_Range')['Within_20'].agg(['count', 'sum', 'mean'])
    accuracy_by_range.columns = ['Total_Predictions', 'Within_20', 'Accuracy_Rate']
    accuracy_by_range['Accuracy_Percentage'] = accuracy_by_range['Accuracy_Rate'] * 100

    print(accuracy_by_range.round(2))

    # Show worst predictions
    print(f"\n❌ WORST PREDICTIONS (Largest errors):")
    worst_predictions = test_results.nlargest(10, 'Difference')
    print(worst_predictions[['Actual_Price', 'Predicted_Price', 'Difference']].round(2))

    # Show best predictions
    print(f"\n✅ BEST PREDICTIONS (Smallest errors):")
    best_predictions = test_results.nsmallest(10, 'Difference')
    print(best_predictions[['Actual_Price', 'Predicted_Price', 'Difference']].round(2))

    # Overall statistics
    print(f"\n OVERALL STATISTICS:")
    print(f"Mean Absolute Error: ${np.mean(absolute_differences):.2f}")
    print(f"Median Absolute Error: ${np.median(absolute_differences):.2f}")
    print(f"Standard Deviation of Error: ${np.std(absolute_differences):.2f}")
    print(f"R² Score: {r2_score(y_test_1000, y_pred_1000):.3f}")

    # Feature importance from full model
    feature_importance_full = pd.DataFrame({
        'feature': X_full.columns,
        'importance': full_model.feature_importances_
    }).sort_values('importance', ascending=False)

    print(f"\n🔍 TOP 10 MOST IMPORTANT FEATURES (Full Model):")
    print(feature_importance_full.head(10))

    # Save results
    full_model_results = {
        'training_size': X_train_full.shape[0],
        'test_size': test_size,
        'accuracy_within_20': float(accuracy_within_20),
        'mean_absolute_error': float(np.mean(absolute_differences)),
        'r2_score': float(r2_score(y_test_1000, y_pred_1000)),
        'within_10_count': int(np.sum(absolute_differences <= 10)),
        'within_20_count': int(np.sum(absolute_differences <= 20)),
        'within_50_count': int(np.sum(absolute_differences <= 50)),
        'within_100_count': int(np.sum(absolute_differences <= 100)),
        'total_features': X_full.shape[1]
    }

    # Ensure results directory exists
    import os
    os.makedirs('results', exist_ok=True)
    
    with open('results/full_model_1000_test_results.json', 'w') as f:
        json.dump(full_model_results, f, indent=2)

    print(f"\n✅ Full model results saved to results/full_model_1000_test_results.json")
    print(f"🎯 Model trained on {X_train_full.shape[0]:,} entries, tested on {test_size} entries")
    
    return full_model_results

def run_predictive_analysis(listings):
    """
    Run complete predictive analysis including both basic and full model training.
    
    Args:
        listings (pd.DataFrame): The listings dataframe
        
    Returns:
        dict: Combined model results
    """
    print("🤖 PREDICTIVE MODELING ANALYSIS")
    print("=" * 60)
    
    # Clean price column if not already done
    if 'price_clean' not in listings.columns and 'price' in listings.columns:
        listings['price_clean'] = listings['price'].str.replace('$', '').str.replace(',', '').astype(float)
    
    # Create clean dataset for modeling
    clean_listings = listings[
        (listings['price_clean'] > 0) & 
        (listings['price_clean'] <= 2000) &  # Remove extreme high prices
        (listings['room_type'] != 'Hotel room')  # Remove problematic hotel rooms
    ].copy()
    
    print(f"Clean dataset for modeling: {len(clean_listings):,} listings")
    
    # Run basic model
    print("\n" + "="*60)
    print("🔮 BASIC MODEL TRAINING")
    print("="*60)
    basic_results = train_basic_model(clean_listings)
    
    # Run full model with 1000-entry test
    print("\n" + "="*60)
    print("🎯 FULL MODEL TRAINING WITH 1000-ENTRY TEST")
    print("="*60)
    full_results = train_full_model_with_1000_test(clean_listings)
    
    # Combine results
    combined_results = {
        'basic_model': basic_results,
        'full_model': full_results,
        'total_clean_listings': len(clean_listings)
    }
    
    print(f"\n🎉 Predictive modeling complete!")
    print(f"📊 Basic model MAE: ${basic_results['mae']:.2f}")
    print(f"📊 Full model accuracy within $20: {full_results['accuracy_within_20']:.1f}%")
    
    return combined_results 