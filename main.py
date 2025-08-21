import matplotlib
matplotlib.use('Agg')  # MUST be before importing matplotlib.pyplot

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import seaborn as sns
import os
import joblib
warnings.filterwarnings('ignore')

# Add for new models
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose

# Global variables
overall_accuracy = None
accuracy_evaluated = False
_cached_data = None
_cached_region_mapping = None


# Create directories if they don't exist
os.makedirs('static/images/forecasts', exist_ok=True)
os.makedirs('models', exist_ok=True)

# --- Crop Recommendation Logic ---
def recommend_crop(avg_temp, avg_ndvi):
    """
    Recommends up to three crops (Rice, Cassava, Maize) based on average forecasted
    Temperature (LST) and NDVI, ranked by suitability.
    """
    high_temp_threshold = 29.0
    moderate_temp_upper = 31.0
    moderate_ndvi_threshold = 0.55
    recommendations = []

    # Evaluate conditions and build ranked list
    if avg_temp > high_temp_threshold:
        if avg_ndvi > moderate_ndvi_threshold:
            recommendations.append(("Cassava", "High temperature and high NDVI are ideal for Cassava."))
        recommendations.append(("Cassava", "High temperature; Cassava is the most drought-tolerant option."))
    else:
        if avg_ndvi > moderate_ndvi_threshold:
            if high_temp_threshold < avg_temp <= moderate_temp_upper:
                recommendations.append(("Rice", "Moderate temperature with high NDVI suggests good water availability, suitable for Rice."))
            if avg_temp <= high_temp_threshold:
                recommendations.append(("Maize", "Moderate/lower temperature with high NDVI, suitable for Maize."))
            recommendations.append(("Rice", "Conditions support high vegetation health, suitable for Rice."))
        else:
            if avg_temp > high_temp_threshold - 1.5:
                recommendations.append(("Cassava", "Warmer conditions, even with moderate NDVI, favor drought-tolerant Cassava."))
            recommendations.append(("Maize", "Moderate/lower temperature with moderate/low NDVI suits Maize."))

    # Ensure at least three recommendations, filling with defaults if needed
    while len(recommendations) < 3:
        if avg_temp > high_temp_threshold and ("Cassava", "Defaulting to Cassava for high temperature.") not in recommendations:
            recommendations.append(("Cassava", "Defaulting to Cassava for high temperature."))
        elif avg_ndvi > moderate_ndvi_threshold and ("Rice", "Defaulting to Rice for high NDVI with non-extreme temp.") not in recommendations:
            recommendations.append(("Rice", "Defaulting to Rice for high NDVI with non-extreme temp."))
        else:
            recommendations.append(("Maize", "Defaulting to Maize for other conditions."))

    # Limit to top 3 and remove duplicates
    seen = set()
    unique_recommendations = []
    for crop, reason in recommendations[:3]:
        if crop not in seen:
            unique_recommendations.append((crop, reason))
            seen.add(crop)
    return unique_recommendations[:3]

def get_least_recommended_crops(avg_temp, avg_ndvi):
    """
    Determines the three least recommended crops based on inverse suitability.
    """
    high_temp_threshold = 29.0
    moderate_temp_upper = 31.0
    moderate_ndvi_threshold = 0.55
    least_recommendations = []

    # Inverse logic for least suitability
    if avg_temp > high_temp_threshold:
        if avg_ndvi < moderate_ndvi_threshold:
            least_recommendations.append(("Rice", "High temperature with low NDVI makes Rice unsuitable due to water needs."))
        least_recommendations.append(("Maize", "High temperature is unfavorable for Maize growth."))
        least_recommendations.append(("Rice", "Extreme heat reduces Rice viability."))
    else:
        if avg_ndvi < moderate_ndvi_threshold:
            least_recommendations.append(("Rice", "Low NDVI indicates insufficient water, unsuitable for Rice."))
            if avg_temp < high_temp_threshold - 1.5:
                least_recommendations.append(("Cassava", "Cooler temperatures with low NDVI are less ideal for Cassava."))
            least_recommendations.append(("Maize", "Low vegetation health with moderate temperature is poor for Maize."))
        else:
            if avg_temp > moderate_temp_upper:
                least_recommendations.append(("Maize", "Excessive temperature is detrimental to Maize."))
            least_recommendations.append(("Cassava", "High NDVI with moderate temperature is less optimal for drought-tolerant Cassava."))
            least_recommendations.append(("Rice", "High NDVI in cooler conditions is less necessary for Rice."))

    # Ensure at least three recommendations, filling with defaults if needed
    while len(least_recommendations) < 3:
        if avg_temp > high_temp_threshold and ("Rice", "Default least suitable due to high heat.") not in least_recommendations:
            least_recommendations.append(("Rice", "Default least suitable due to high heat."))
        elif avg_ndvi < moderate_ndvi_threshold and ("Maize", "Default least suitable due to low vegetation.") not in least_recommendations:
            least_recommendations.append(("Maize", "Default least suitable due to low vegetation."))
        else:
            least_recommendations.append(("Cassava", "Default least suitable due to moderate conditions."))

    # Limit to top 3 least and remove duplicates
    seen = set()
    unique_least_recommendations = []
    for crop, reason in least_recommendations[:3]:
        if crop not in seen:
            unique_least_recommendations.append((crop, reason))
            seen.add(crop)
    return unique_least_recommendations[:3]

# --- Optimized Data Loading ---
def load_data_optimized():
    """
    Load and parse CSV data using pandas for better performance
    """
    try:
        # Use pandas to read CSV directly - much faster than regex parsing
        combined_df = pd.read_csv('combined_with_plants.csv')
        
        # Convert DATE column to datetime
        combined_df['DATE'] = pd.to_datetime(combined_df['DATE'], format='%m/%d/%Y', errors='coerce')
        
        # Drop rows with invalid dates or missing critical data
        combined_df = combined_df.dropna(subset=['DATE', 'RAINFALL_MM', 'NDVI_VALUE', 'LST_VALUE'])
        
        # Ensure numeric columns are properly typed
        numeric_cols = ['RAINFALL_MM', 'NDVI_VALUE', 'LST_VALUE']
        for col in numeric_cols:
            combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce')
        
        combined_df = combined_df.dropna(subset=numeric_cols)
        combined_df = combined_df.reset_index(drop=True)
        
        print(f"✅ Data loaded successfully. Total records: {len(combined_df)}")
        return combined_df
        
    except FileNotFoundError:
        print("❌ Error: File 'combined_with_plants.csv' not found.")
        return None
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None

# --- Model Evaluation ---
def evaluate_crop_recommendation(combined_df, sample_size=1000):
    """
    Evaluate the crop recommendation model using a sample of historical data
    and store the accuracy globally.
    """
    global overall_accuracy
    print("\n" + "=" * 60)
    print("📈 MODEL EVALUATION")
    print("=" * 60)

    # Use a sample for faster evaluation if dataset is large
    if len(combined_df) > sample_size:
        sample_df = combined_df.sample(n=sample_size, random_state=42)
        print(f"Using sample of {sample_size} records for evaluation")
    else:
        sample_df = combined_df

    # Apply recommendation logic to historical data
    predictions = []
    actuals = sample_df['RECOMMENDED'].tolist()

    for _, row in sample_df.iterrows():
        pred, _ = recommend_crop(row['LST_VALUE'], row['NDVI_VALUE'])[0]  # Take first recommendation
        predictions.append(pred)

    # Calculate metrics
    accuracy = accuracy_score(actuals, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        actuals, predictions, average='weighted', zero_division=0
    )
    overall_accuracy = accuracy  # Store accuracy globally

    print(f"Accuracy: {accuracy:.3f}")
    print(f"Precision (Weighted): {precision:.3f}")
    print(f"Recall (Weighted): {recall:.3f}")
    print(f"F1-Score (Weighted): {f1:.3f}")

    # Confusion Matrix
    unique_labels = list(set(actuals + predictions))
    cm = confusion_matrix(actuals, predictions, labels=unique_labels)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=unique_labels, yticklabels=unique_labels)
    plt.title('Confusion Matrix for Crop Recommendation')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    
    # Save the confusion matrix plot
    plt.savefig('static/images/forecasts/confusion_matrix.png')
    # plt.show()

# --- Forecasting System ---
def create_region_mapping(combined_df):
    """
    Create a mapping of GID_2 codes to human-readable region names
    """
    unique_gids = combined_df['GID_2'].unique()
    malawi_regions = {
        'MWI.1.1_1': 'Northern Region - Chitipa',
        'MWI.1.2_1': 'Northern Region - Karonga',
        'MWI.1.3_1': 'Northern Region - Rumphi',
        'MWI.2.1_1': 'Central Region - Kasungu',
        'MWI.2.2_1': 'Central Region - Nkhotakota',
        'MWI.2.3_1': 'Central Region - Ntchisi',
        'MWI.2.4_1': 'Central Region - Dowa',
        'MWI.2.5_1': 'Central Region - Salima',
        'MWI.2.6_1': 'Central Region - Lilongwe',
        'MWI.2.7_1': 'Central Region - Mchinji',
        'MWI.3.1_1': 'Southern Region - Mangochi',
        'MWI.3.2_1': 'Southern Region - Machinga',
        'MWI.3.3_1': 'Southern Region - Zomba',
        'MWI.3.4_1': 'Southern Region - Chiradzulu',
        'MWI.3.5_1': 'Southern Region - Blantyre',
        'MWI.3.6_1': 'Southern Region - Mwanza',
        'MWI.3.7_1': 'Southern Region - Thyolo',
        'MWI.3.8_1': 'Southern Region - Mulanje',
        'MWI.3.9_1': 'Southern Region - Phalombe',
        'MWI.4.1_1': 'Southern Region - Chikwawa',
        'MWI.4.2_1': 'Southern Region - Nsanje',
        'MWI.4.3_1': 'Southern Region - Balaka',
        'MWI.4.4_1': 'Southern Region - Neno',
    }
    region_mapping = {gid: malawi_regions.get(gid, f"Region {gid}") for gid in unique_gids}
    return region_mapping

def get_available_options(combined_df):
    """
    Get available countries and regions for user selection
    """
    combined_df['country_code'] = combined_df['GID_2'].str.slice(0, 3)
    country_mapping = {
        'MWI': 'Malawi',
        'GHA': 'Ghana',
        'UGA': 'Uganda',
        'SEN': 'Senegal',
        'BEN': 'Benin'
    }
    available_countries = {code: country_mapping.get(code, f"Country_{code}") 
                         for code in combined_df['country_code'].unique()}
    return available_countries, create_region_mapping(combined_df)

def display_available_options(available_countries, region_mapping):
    """
    Display available options to the user
    """
    print("🌍 AVAILABLE COUNTRIES:")
    print("=" * 40)
    for code, name in available_countries.items():
        print(f"{code}: {name}")
    
    print("\n📍 AVAILABLE REGIONS (showing first 15):")
    print("=" * 60)
    region_items = list(region_mapping.items())
    for gid, name in region_items[:15]:
        print(f"{gid}: {name}")
    if len(region_mapping) > 15:
        print(f"... and {len(region_mapping) - 15} more regions")
    print(f"\n💡 Tip: Type 'Malawi' for country-level forecast or a GID code for regional analysis")

def get_user_selection(available_countries, region_mapping):
    """
    Get user input for country/region selection
    """
    while True:
        print("\n" + "=" * 60)
        user_input = input("Enter country name (e.g., 'Malawi') or GID code (e.g., 'MWI.1.1_1'): ").strip()
        
        # Check for country match
        country_lower = user_input.lower()
        for code, name in available_countries.items():
            if country_lower == name.lower():
                return {'type': 'country', 'value': code, 'name': name}
        
        # Check for region match
        if user_input in region_mapping:
            return {'type': 'region', 'value': user_input, 'name': region_mapping[user_input]}
        
        # Special case for Malawi
        if country_lower == 'malawi':
            return {'type': 'country', 'value': 'MWI', 'name': 'Malawi'}
        
        print("❌ Invalid selection. Please choose from the available options.")
        display_available_options(available_countries, region_mapping)

def prepare_data_for_forecast(combined_df, selection):
    """
    Prepare data based on user selection with optimized aggregation
    """
    if selection['type'] == 'country':
        country_code = selection['value']
        # Use vectorized operations for filtering
        mask = combined_df['GID_2'].str.startswith(country_code)
        country_data = combined_df[mask]
        
        if country_data.empty:
            print(f"❌ No data found for country code: {country_code}")
            return None
        
        # Optimized aggregation
        aggregated_data = (country_data.groupby('DATE')
                         .agg({'RAINFALL_MM': 'mean', 'NDVI_VALUE': 'mean', 'LST_VALUE': 'mean'})
                         .sort_index())
        
        print(f"✅ Prepared country-level data for {selection['name']}")
        print(f"   Time range: {aggregated_data.index.min()} to {aggregated_data.index.max()}")
        print(f"   Total records: {len(aggregated_data)}")
        return aggregated_data
    else:
        region_gid = selection['value']
        # Use boolean indexing for better performance
        mask = combined_df['GID_2'] == region_gid
        region_data = combined_df[mask].sort_values('DATE')
        
        if region_data.empty:
            print(f"❌ No data found for region: {region_gid}")
            return None
        
        region_data = region_data.set_index('DATE')[['RAINFALL_MM', 'NDVI_VALUE', 'LST_VALUE']]
        print(f"✅ Prepared region-level data for {selection['name']}")
        print(f"   Time range: {region_data.index.min()} to {region_data.index.max()}")
        print(f"   Total records: {len(region_data)}")
        return region_data

def load_or_train_sarimax_model(series, model_name, periods=115):
    """
    Load a pre-trained SARIMAX model if available, otherwise train and save a new one
    """
    model_path = f"models/{model_name}.joblib"
    
    # Try to load existing model
    try:
        model = joblib.load(model_path)
        print(f"✅ Loaded pre-trained model: {model_name}")
        return model
    except:
        print(f"⚠ No pre-trained model found. Training new model: {model_name}")
        print("⏳ This may take a few minutes...")
    
    # Handle missing values
    series = series.dropna()
    
    if len(series) < 24:  # Need at least 2 years of data (assuming 16-day intervals)
        print("⚠ Insufficient data for SARIMAX")
        return None
    
    # Automatically determine seasonal period (assuming 16-day intervals)
    # For 16-day data, seasonal period is approximately 23 (365/16 ≈ 23)
    seasonal_period = 23
    
    # SARIMAX model with seasonal components
    # Order: (p,d,q) - ARIMA parameters
    # Seasonal_order: (P,D,Q,s) - Seasonal parameters
    model = SARIMAX(
        series,
        order=(1, 1, 1),           # Non-seasonal ARIMA(1,1,1)
        seasonal_order=(1, 1, 1, seasonal_period),  # Seasonal SARIMA(1,1,1,23)
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    
    # Fit the model
    print(f"⏳ Fitting SARIMAX model for {model_name}...")
    fitted_model = model.fit(disp=False)
    print(f"✅ Model training completed for {model_name}")
    
    # Save the trained model
    joblib.dump(fitted_model, model_path)
    print(f"✅ Model trained and saved: {model_name}")
    
    return fitted_model

def sarimax_forecast(series, periods=115, model_name="default"):
    """
    SARIMAX forecasting implementation with model saving/loading
    """
    try:
        # Handle missing values
        series = series.dropna()
        
        if len(series) < 24:  # Need at least 2 years of data (assuming 16-day intervals)
            print("⚠ Insufficient data for SARIMAX")
            return None, None, None
        
        # Load or train model
        fitted_model = load_or_train_sarimax_model(series, model_name, periods)
        if fitted_model is None:
            return None, None, None
        
        # Forecast
        forecast_result = fitted_model.get_forecast(steps=periods)
        forecast_mean = forecast_result.predicted_mean
        forecast_ci = forecast_result.conf_int()
        
        # Create forecast series with proper dates
        last_date = series.index[-1]
        future_dates = pd.date_range(last_date + pd.Timedelta(days=16), periods=periods, freq='16D')
        
        forecast_series = pd.Series(forecast_mean.values, index=future_dates)
        lower_bound = pd.Series(forecast_ci.iloc[:, 0].values, index=future_dates)
        upper_bound = pd.Series(forecast_ci.iloc[:, 1].values, index=future_dates)
        
        return forecast_series, lower_bound, upper_bound
        
    except Exception as e:
        print(f"⚠ SARIMAX failed: {e}")
        return None, None, None

def run_forecast(data, selection, forecast_years=5):
    """
    Run the forecasting pipeline using SARIMAX only
    """
    periods = int(365 * forecast_years / 16)
    if periods <= 0:
        periods = 60
    
    print(f"\n🔮 Forecasting for {selection['name']} ({forecast_years} years) using SARIMAX...")
    
    # Create unique model names for each variable and region
    region_id = selection['name'].replace(' ', '_').lower()
    rainfall_forecast, rain_lower, rain_upper = sarimax_forecast(
        data['RAINFALL_MM'], periods, f"rainfall_{region_id}"
    )
    ndvi_forecast, ndvi_lower, ndvi_upper = sarimax_forecast(
        data['NDVI_VALUE'], periods, f"ndvi_{region_id}"
    )
    lst_forecast, lst_lower, lst_upper = sarimax_forecast(
        data['LST_VALUE'], periods, f"lst_{region_id}"
    )
    
    if rainfall_forecast is None or ndvi_forecast is None or lst_forecast is None:
        print("❌ Forecasting failed due to insufficient data or errors.")
        return None, None
    
    forecasts = pd.DataFrame({
        'RAINFALL_MM': rainfall_forecast.values,
        'NDVI_VALUE': ndvi_forecast.values,
        'LST_VALUE': lst_forecast.values
    }, index=rainfall_forecast.index)
    
    confidence_intervals = {
        'rainfall': {'lower': rain_lower, 'upper': rain_upper},
        'ndvi': {'lower': ndvi_lower, 'upper': ndvi_upper},
        'lst': {'lower': lst_lower, 'upper': lst_upper}
    }
    
    return forecasts, confidence_intervals

def plot_results(historical_data, forecasts, ci_dict, selection):
    """
    Plot forecasting results with optimized plotting and save to static/images/forecasts
    """
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 12))
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    
    # Rainfall plot
    ax1.plot(historical_data.index, historical_data['RAINFALL_MM'],
             label='Historical', linewidth=2, color=colors[0], alpha=0.8)
    ax1.plot(forecasts.index, forecasts['RAINFALL_MM'],
             label='Forecast', linewidth=3, color=colors[0])
    ax1.fill_between(forecasts.index, ci_dict['rainfall']['lower'], ci_dict['rainfall']['upper'],
                    color=colors[0], alpha=0.2, label='95% CI')
    ax1.set_ylabel('Rainfall (mm)', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # NDVI plot
    ax2.plot(historical_data.index, historical_data['NDVI_VALUE'],
             label='Historical', linewidth=2, color=colors[1], alpha=0.8)
    ax2.plot(forecasts.index, forecasts['NDVI_VALUE'],
             label='Forecast', linewidth=3, color=colors[1])
    ax2.fill_between(forecasts.index, ci_dict['ndvi']['lower'], ci_dict['ndvi']['upper'],
                    color=colors[1], alpha=0.2, label='95% CI')
    ax2.set_ylabel('NDVI Value', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Temperature plot
    ax3.plot(historical_data.index, historical_data['LST_VALUE'],
             label='Historical', linewidth=2, color=colors[2], alpha=0.8)
    ax3.plot(forecasts.index, forecasts['LST_VALUE'],
             label='Forecast', linewidth=3, color=colors[2])
    ax3.fill_between(forecasts.index, ci_dict['lst']['lower'], ci_dict['lst']['upper'],
                    color=colors[2], alpha=0.2, label='95% CI')
    ax3.set_ylabel('Temperature (°C)', fontweight='bold')
    ax3.set_xlabel('Year', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.suptitle(f'5-Year Forecast for {selection["name"]}\n', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save the plot
    filename = f"static/images/forecasts/forecast_{selection['name'].replace(' ', '_').lower()}.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"✅ Forecast plot saved to: {filename}")
    
    plt.close(fig)  # Close the figure to free memory - CRITICAL for web apps

def generate_summary(forecasts, selection):
    """
    Generate forecast summary, crop recommendation, least recommended crops, and prediction accuracy
    """
    print("\n" + "=" * 60)
    print(f"📊 FORECAST SUMMARY: {selection['name']}")
    print("=" * 60)
    
    avg_rainfall = forecasts['RAINFALL_MM'].mean()
    avg_ndvi = forecasts['NDVI_VALUE'].mean()
    avg_temp = forecasts['LST_VALUE'].mean()
    
    print(f"\n📅 Forecast Period: {forecasts.index[0].strftime('%Y-%m-%d')} to {forecasts.index[-1].strftime('%Y-%m-%d')}")
    print(f"\n🌧  Rainfall Forecast:")
    print(f"   Average: {avg_rainfall:.1f} mm")
    print(f"   Range: {forecasts['RAINFALL_MM'].min():.1f} - {forecasts['RAINFALL_MM'].max():.1f} mm")
    
    print(f"\n🌿 Vegetation Health (NDVI):")
    print(f"   Average: {avg_ndvi:.3f}")
    print(f"   Range: {forecasts['NDVI_VALUE'].min():.3f} - {forecasts['NDVI_VALUE'].max():.3f}")
    
    print(f"\n🌡  Temperature Forecast:")
    print(f"   Average: {avg_temp:.1f}°C")
    print(f"   Range: {forecasts['LST_VALUE'].min():.1f} - {forecasts['LST_VALUE'].max():.1f}°C")
    
    print("\n" + "=" * 60)
    print(f"🌾 TOP 3 CROP RECOMMENDATIONS FOR {selection['name']}")
    print("=" * 60)
    
    crop_recommendations = recommend_crop(avg_temp, avg_ndvi)
    for i, (crop, reason) in enumerate(crop_recommendations, 1):
        print(f"{i}. 🌾 Recommended Crop: {crop}")
        print(f"   🧠 Reason: {reason}")
    
    print("\n" + "=" * 60)
    print(f"🌾 TOP 3 LEAST RECOMMENDED CROPS FOR {selection['name']}")
    print("=" * 60)
    
    least_recommendations = get_least_recommended_crops(avg_temp, avg_ndvi)
    for i, (crop, reason) in enumerate(least_recommendations, 1):
        print(f"{i}. 🌾 Least Recommended Crop: {crop}")
        print(f"   🧠 Reason: {reason}")
    
    print("\n" + "=" * 60)
    print(f"📊 PREDICTION ACCURACY")
    print("=" * 60)
    if overall_accuracy is not None:
        print(f"Overall Accuracy of Crop Recommendations: {overall_accuracy:.3f}")
    else:
        print("Accuracy not available due to insufficient evaluation data.")
    print("=" * 60)

def main():
    """
    Main interactive forecasting system - optimized version using SARIMAX only
    """
    print("🌍 WELCOME TO THE CLIMATE FORECASTING SYSTEM")
    print("=" * 50)
    print("This system predicts Rainfall, Vegetation Health, and Temperature")
    print("for any available country or region for the next 5 years using SARIMAX!")
    print("It also provides crop recommendations and least recommended crops (Rice, Cassava, Maize)!")
    print("=" * 50)
    
    # Load data using optimized method
    combined_df = load_data_optimized()
    if combined_df is None:
        return
    
    # Quick model evaluation with smaller sample
    print("\n⚡ Running quick model evaluation...")
    evaluate_crop_recommendation(combined_df, sample_size=500)
    
    # Get available options
    available_countries, region_mapping = get_available_options(combined_df)
    

    display_available_options(available_countries, region_mapping)
    selection = get_user_selection(available_countries, region_mapping)
    
    data = prepare_data_for_forecast(combined_df, selection)
    if data is None or data.empty:
        print("⚠ Unable to prepare data for forecasting. Please try another selection.")
    
    try:
        forecasts, ci_dict = run_forecast(data, selection)
        if forecasts is None:
            print("❌ Forecasting failed. Please check your data or try another region.")
    except Exception as e:
        print(f"❌ Error during forecasting: {e}")
    
    try:
        plot_results(data, forecasts, ci_dict, selection)
    except Exception as e:
        print(f"⚠ Could not display plot: {e}")
    
    generate_summary(forecasts, selection)
    
    # Save results
    filename = f"static/madecsvs/forecast_{selection['name'].replace(' ', '_').lower()}.csv"
    try:
        forecasts.to_csv(filename)
        print(f"\n💾 Forecast saved to: {filename}")
    except Exception as e:
        print(f"⚠ Could not save forecast to file: {e}")
    
    print("\n" + "=" * 50)


def run_forecast_pipeline(selection_input):
    """
    Run the forecasting pipeline for a given region or country.
    :param selection_input: Dictionary with keys 'type', 'value', 'name'
    :return: forecasts DataFrame, confidence intervals, summary info
    """
    combined_df, _ = get_cached_data()
    if combined_df is None:
        return None, None, "Data loading failed."

    # Prepare data
    data = prepare_data_for_forecast(combined_df, selection_input)
    if data is None or data.empty:
        return None, None, "No data available for the selected region."

    # Run forecast
    forecasts, ci_dict = run_forecast(data, selection_input)
    if forecasts is None:
        return None, None, "Forecasting failed."

    # Generate summary
    avg_rainfall = forecasts['RAINFALL_MM'].mean()
    avg_ndvi = forecasts['NDVI_VALUE'].mean()
    avg_temp = forecasts['LST_VALUE'].mean()

    crop_recommendations = recommend_crop(avg_temp, avg_ndvi)
    least_recommendations = get_least_recommended_crops(avg_temp, avg_ndvi)

    summary = {
        'region': selection_input['name'],
        'avg_rainfall': avg_rainfall,
        'avg_ndvi': avg_ndvi,
        'avg_temp': avg_temp,
        'crop_recommendations': crop_recommendations,
        'least_recommendations': least_recommendations
        # Removed accuracy from here
    }

    return forecasts, ci_dict, summary


from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
import io
import base64

def generate_pdf_report(summary, forecasts, selection):
    """
    Generate a PDF report for the forecast results
    """
    buffer = io.BytesIO()
    
    # Create document
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []
    
    # Title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
        alignment=TA_CENTER,
        textColor=colors.darkblue
    )
    
    story.append(Paragraph("Climate Forecast Report", title_style))
    story.append(Spacer(1, 20))
    
    # Region info
    story.append(Paragraph(f"<b>Region:</b> {summary['region']}", styles['Normal']))
    story.append(Spacer(1, 10))
    
    # Forecast period
    if len(forecasts) > 0:
        start_date = forecasts.index[0].strftime('%Y-%m-%d')
        end_date = forecasts.index[-1].strftime('%Y-%m-%d')
        story.append(Paragraph(f"<b>Forecast Period:</b> {start_date} to {end_date}", styles['Normal']))
        story.append(Spacer(1, 20))
    
    # Statistics table
    story.append(Paragraph("<b>Forecast Statistics</b>", styles['Heading2']))
    story.append(Spacer(1, 10))
    
    data = [
        ['Metric', 'Value'],
        ['Average Rainfall', f"{summary['avg_rainfall']:.2f} mm"],
        ['Average NDVI', f"{summary['avg_ndvi']:.3f}"],
        ['Average Temperature', f"{summary['avg_temp']:.2f} °C"]
    ]
    
    table = Table(data)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 14),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    story.append(table)
    story.append(Spacer(1, 30))
    
    # Crop recommendations
    story.append(Paragraph("<b>Top 3 Recommended Crops</b>", styles['Heading2']))
    story.append(Spacer(1, 10))
    
    for i, (crop, reason) in enumerate(summary['crop_recommendations'], 1):
        story.append(Paragraph(f"<b>{i}. {crop}</b>: {reason}", styles['Normal']))
        story.append(Spacer(1, 5))
    
    story.append(Spacer(1, 20))
    
    # Least recommended crops
    story.append(Paragraph("<b>Top 3 Least Recommended Crops</b>", styles['Heading2']))
    story.append(Spacer(1, 10))
    
    for i, (crop, reason) in enumerate(summary['least_recommendations'], 1):
        story.append(Paragraph(f"<b>{i}. {crop}</b>: {reason}", styles['Normal']))
        story.append(Spacer(1, 5))
    
    story.append(Spacer(1, 30))
    
    # Chart image (if exists)
    chart_path = f"static/images/forecasts/forecast_{selection['name'].replace(' ', '_').lower()}.png"
    if os.path.exists(chart_path):
        try:
            story.append(Paragraph("<b>Forecast Visualization</b>", styles['Heading2']))
            story.append(Spacer(1, 10))
            img = Image(chart_path, width=6*inch, height=4*inch)
            story.append(img)
        except:
            story.append(Paragraph("<i>Chart could not be included in PDF</i>", styles['Normal']))
    
    # Build PDF
    doc.build(story)
    
    buffer.seek(0)
    return buffer

# Flask web application for the forecasting system

from flask import *

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Needed for flash messages

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/forecast')
def forecast():
    return render_template('forecast.html')
def get_cached_data():
    global _cached_data, _cached_region_mapping
    if _cached_data is None:
        _cached_data = load_data_optimized()
        if _cached_data is not None:
            _cached_region_mapping = create_region_mapping(_cached_data)
    return _cached_data, _cached_region_mapping

# Update your Flask route to use cached data:
@app.route('/selection', methods=['GET', 'POST'])
def selection():
    combined_df, region_mapping = get_cached_data()
    
    if combined_df is None:
        flash("Failed to load data", "error")
        return redirect('/selection')
    
    if request.method == 'POST':
        region_code = request.form.get("region")
        
        if not region_code:
            flash("Please select a region", "error")
            return redirect('/selection')

        # Rest of your POST logic...
        # Create region mapping
        region_mapping = create_region_mapping(combined_df)
        
        # Determine if it's a country or region
        if region_code in ['MWI', 'GHA', 'UGA', 'SEN', 'BEN']:  # Country codes
            country_mapping = {
                'MWI': 'Malawi',
                'GHA': 'Ghana',
                'UGA': 'Uganda',
                'SEN': 'Senegal',
                'BEN': 'Benin'
            }
            selection_input = {
                'type': 'country',
                'value': region_code,
                'name': country_mapping.get(region_code, region_code)
            }
        elif region_code in region_mapping:
            selection_input = {
                'type': 'region',
                'value': region_code,
                'name': region_mapping[region_code]
            }
        else:
            flash("Invalid region selected", "error")
            return redirect('/selection')

        # Run forecasting
        forecasts, ci_dict, summary = run_forecast_pipeline(selection_input)

        if forecasts is None:
            flash(f"Forecast failed: {summary}", "error")
            return redirect('/selection')

        # Save plot to static/images/forecasts/
        try:
            plot_results(prepare_data_for_forecast(combined_df, selection_input), forecasts, ci_dict, selection_input)
        except Exception as e:
            flash(f"Could not generate plot: {str(e)}", "error")
            return redirect('/selection')

        # Pass summary to the template
        return render_template('results.html', summary=summary)

    # GET request - show the selection form
    available_countries, region_mapping = get_available_options(combined_df)
    
    return render_template('selection.html', 
                         countries=available_countries, 
                         regions=region_mapping)

@app.route('/forecastselection')
def forecastselection():
    return render_template('forecastselection.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/help')
def help():
    return render_template('help.html')


@app.route('/download_pdf/<region_name>')
def download_pdf(region_name):
    """
    Download PDF report for a specific region
    """
    # Reconstruct selection dict
    combined_df, region_mapping = get_cached_data()
    if combined_df is None:
        flash("Data not available", "error")
        return redirect('/selection')
    
    # Find the region
    selection_input = None
    for gid, name in region_mapping.items():
        if name.replace(' ', '_').lower() == region_name:
            selection_input = {'type': 'region', 'value': gid, 'name': name}
            break
    
    if selection_input is None:
        flash("Region not found", "error")
        return redirect('/selection')
    
    # Prepare data and run forecast (or load existing)
    data = prepare_data_for_forecast(combined_df, selection_input)
    if data is None or data.empty:
        flash("No data available", "error")
        return redirect('/selection')
    
    # Run forecast if needed
    forecasts, ci_dict = run_forecast(data, selection_input)
    if forecasts is None:
        flash("Could not generate forecast", "error")
        return redirect('/selection')
    
    # Generate summary
    avg_rainfall = forecasts['RAINFALL_MM'].mean()
    avg_ndvi = forecasts['NDVI_VALUE'].mean()
    avg_temp = forecasts['LST_VALUE'].mean()
    
    summary = {
        'region': selection_input['name'],
        'avg_rainfall': avg_rainfall,
        'avg_ndvi': avg_ndvi,
        'avg_temp': avg_temp,
        'crop_recommendations': recommend_crop(avg_temp, avg_ndvi),
        'least_recommendations': get_least_recommended_crops(avg_temp, avg_ndvi),
        'accuracy': overall_accuracy
    }
    
    # Generate PDF
    pdf_buffer = generate_pdf_report(summary, forecasts, selection_input)
    
    # Return as download
    region_safe = selection_input['name'].replace(' ', '_').lower()
    return send_file(
        pdf_buffer,
        as_attachment=True,
        download_name=f'forecast_report_{region_safe}.pdf',
        mimetype='application/pdf'
    )

@app.route('/download_csv/<region_name>')
def download_csv(region_name):
    """
    Download CSV forecast data for a specific region
    """
    # Reconstruct selection dict
    combined_df, region_mapping = get_cached_data()
    if combined_df is None:
        flash("Data not available", "error")
        return redirect('/selection')
    
    # Find the region
    selection_input = None
    for gid, name in region_mapping.items():
        if name.replace(' ', '_').lower() == region_name:
            selection_input = {'type': 'region', 'value': gid, 'name': name}
            break
    
    if selection_input is None:
        flash("Region not found", "error")
        return redirect('/selection')
    
    # Prepare data and run forecast
    data = prepare_data_for_forecast(combined_df, selection_input)
    if data is None or data.empty:
        flash("No data available", "error")
        return redirect('/selection')
    
    forecasts, _ = run_forecast(data, selection_input)
    if forecasts is None:
        flash("Could not generate forecast", "error")
        return redirect('/selection')
    
    # Return CSV
    region_safe = selection_input['name'].replace(' ', '_').lower()
    return send_file(
        io.BytesIO(forecasts.to_csv().encode()),
        as_attachment=True,
        download_name=f'forecast_data_{region_safe}.csv',
        mimetype='text/csv'
    )

if __name__ == '__main__':
    app.run(debug=True)