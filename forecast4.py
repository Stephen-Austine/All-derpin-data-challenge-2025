import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

# Load your data
combined_df = pd.read_csv('rainfall_ndvi_lst_cleaned.csv')
combined_df['DATE'] = pd.to_datetime(combined_df['DATE'])

# Create a mapping of GID_2 codes to regions
def create_region_mapping(combined_df):
    """
    Create a mapping of GID_2 codes to human-readable region names
    """
    unique_gids = combined_df['GID_2'].unique()
    
    # Malawi region mapping
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
    
    region_mapping = {}
    for gid in unique_gids:
        if gid in malawi_regions:
            region_mapping[gid] = malawi_regions[gid]
        else:
            # Create readable names from GID codes
            parts = gid.split('_')
            country_code = parts[0][:3]
            region_code = parts[0][4:] if len(parts[0]) > 3 else parts[0]
            region_mapping[gid] = f"{country_code} Region {region_code}"
    
    return region_mapping

def get_available_options(combined_df):
    """
    Get available countries and regions for user selection
    """
    # Extract country codes from GID_2 (first 3 letters before the first dot)
    combined_df['country_code'] = combined_df['GID_2'].str.split('.').str[0]
    
    # Map country codes to full names
    country_mapping = {
        'MWI': 'Malawi',
        'GHA': 'Ghana',
        'UGA': 'Uganda',
        'SEN': 'Senegal',
        'BEN': 'Benin'
    }
    
    available_countries = {}
    for code in combined_df['country_code'].unique():
        if pd.notna(code) and code in country_mapping:
            available_countries[code] = country_mapping[code]
        else:
            available_countries[code] = f"Country {code}"
    
    return available_countries, create_region_mapping(combined_df)

def display_available_options(available_countries, region_mapping):
    """
    Display available options to the user
    """
    print("🌍 AVAILABLE COUNTRIES:")
    print("=" * 40)
    for code, name in available_countries.items():
        print(f"• {name} (code: {code})")
    
    print(f"\n📍 AVAILABLE REGIONS ({len(region_mapping)} total):")
    print("=" * 60)
    for gid, name in list(region_mapping.items())[:15]:  # Show first 15
        print(f"• {gid}: {name}")
    
    if len(region_mapping) > 15:
        print(f"... and {len(region_mapping) - 15} more regions")
    
    print(f"\n💡 TIPS:")
    print("   - Type country name (e.g., 'Malawi') for country-level forecast")
    print("   - Type GID code (e.g., 'MWI.1.1_1') for regional analysis")
    print("   - Type 'list countries' to see available countries")
    print("   - Type 'list regions' to see more regions")
    print("   - Type 'exit' to quit")

def get_user_selection(available_countries, region_mapping):
    """
    Get user input for country/region selection
    """
    while True:
        print("\n" + "=" * 60)
        user_input = input("Enter your choice: ").strip()
        
        # Check for exit command
        if user_input.lower() in ['exit', 'quit', 'q']:
            return None
        
        # Check for list commands
        if user_input.lower() == 'list countries':
            print("\nAvailable Countries:")
            for code, name in available_countries.items():
                print(f"  {code}: {name}")
            continue
        
        if user_input.lower() == 'list regions':
            print(f"\nAvailable Regions ({len(region_mapping)}):")
            for gid, name in region_mapping.items():
                print(f"  {gid}: {name}")
            continue
        
        # Check for country name (case insensitive)
        user_input_lower = user_input.lower()
        for code, name in available_countries.items():
            if user_input_lower == name.lower():
                return {'type': 'country', 'value': code, 'name': name}
        
        # Check for exact GID code match
        if user_input in region_mapping:
            return {'type': 'region', 'value': user_input, 'name': region_mapping[user_input]}
        
        # Check for partial GID code match
        matching_gids = [gid for gid in region_mapping.keys() if user_input in gid]
        if len(matching_gids) == 1:
            gid = matching_gids[0]
            return {'type': 'region', 'value': gid, 'name': region_mapping[gid]}
        elif len(matching_gids) > 1:
            print(f"❌ Multiple regions match '{user_input}':")
            for gid in matching_gids[:5]:  # Show first 5 matches
                print(f"   {gid}: {region_mapping[gid]}")
            if len(matching_gids) > 5:
                print(f"   ... and {len(matching_gids) - 5} more")
            print("Please be more specific.")
            continue
        
        print("❌ Invalid selection. Please choose from the available options.")
        print("   Type 'list countries' or 'list regions' to see available options.")

def prepare_data_for_forecast(combined_df, selection, forecast_years=5):
    """
    Prepare data based on user selection
    """
    if selection['type'] == 'country':
        # Country-level analysis - aggregate all regions for that country
        country_code = selection['value']
        
        # Extract country code from GID_2 (part before first dot)
        combined_df['country_code_from_gid'] = combined_df['GID_2'].str.split('.').str[0]
        country_data = combined_df[combined_df['country_code_from_gid'] == country_code]
        
        if country_data.empty:
            print(f"❌ No data found for country code: {country_code}")
            print("Available country codes:", combined_df['country_code_from_gid'].unique())
            return None
        
        # Aggregate by date
        aggregated_data = country_data.groupby('DATE').agg({
            'RAINFALL_MM': 'mean',
            'NDVI_VALUE': 'mean',
            'LST_VALUE': 'mean'
        }).sort_index()
        
        print(f"✅ Prepared country-level data for {selection['name']}")
        print(f"   Time range: {aggregated_data.index.min()} to {aggregated_data.index.max()}")
        print(f"   Total records: {len(aggregated_data)}")
        print(f"   Regions included: {country_data['GID_2'].nunique()}")
        
        return aggregated_data
        
    else:
        # Region-level analysis
        region_gid = selection['value']
        region_data = combined_df[combined_df['GID_2'] == region_gid].sort_values('DATE')
        
        if region_data.empty:
            print(f"❌ No data found for region: {region_gid}")
            print("Available GID codes sample:", list(combined_df['GID_2'].unique())[:10])
            return None
        
        region_data = region_data.set_index('DATE')[['RAINFALL_MM', 'NDVI_VALUE', 'LST_VALUE']]
        
        print(f"✅ Prepared region-level data for {selection['name']}")
        print(f"   Time range: {region_data.index.min()} to {region_data.index.max()}")
        print(f"   Total records: {len(region_data)}")
        
        return region_data

def prophet_forecast(series, periods=115, yearly_seasonality=True):
    """
    Use Facebook's Prophet for robust forecasting
    """
    from prophet import Prophet
    
    # Handle missing values
    series = series.dropna()
    
    if len(series) < 10:
        print(f"⚠️  Not enough data for forecasting (only {len(series)} points)")
        return None, None, None
    
    prophet_df = pd.DataFrame({
        'ds': series.index,
        'y': series.values
    })
    
    try:
        model = Prophet(
            yearly_seasonality=yearly_seasonality,
            weekly_seasonality=False,
            daily_seasonality=False,
            changepoint_prior_scale=0.05
        )
        
        model.fit(prophet_df)
        future = model.make_future_dataframe(periods=periods, freq='16D')
        forecast = model.predict(future)
        
        forecast_series = forecast.set_index('ds')['yhat'][-periods:]
        confidence_lower = forecast.set_index('ds')['yhat_lower'][-periods:]
        confidence_upper = forecast.set_index('ds')['yhat_upper'][-periods:]
        
        return forecast_series, confidence_lower, confidence_upper
        
    except Exception as e:
        print(f"❌ Prophet forecasting failed: {e}")
        return None, None, None

def run_forecast(data, selection, forecast_years=5):
    """
    Run the forecasting pipeline
    """
    periods = int(365 * forecast_years / 16)  # 16-day intervals
    
    print(f"\n🔮 Forecasting for {selection['name']} ({forecast_years} years)...")
    
    # Forecast each variable
    rainfall_forecast, rain_lower, rain_upper = prophet_forecast(data['RAINFALL_MM'], periods)
    ndvi_forecast, ndvi_lower, ndvi_upper = prophet_forecast(data['NDVI_VALUE'], periods)
    lst_forecast, lst_lower, lst_upper = prophet_forecast(data['LST_VALUE'], periods)
    
    # Check if any forecast failed
    if any(x is None for x in [rainfall_forecast, ndvi_forecast, lst_forecast]):
        print("❌ Forecasting failed for one or more variables")
        return None, None
    
    # Create forecast DataFrame
    forecasts = pd.DataFrame({
        'RAINFALL_MM': rainfall_forecast.values,
        'NDVI_VALUE': ndvi_forecast.values,
        'LST_VALUE': lst_forecast.values
    }, index=rainfall_forecast.index)
    
    return forecasts, {
        'rainfall': {'lower': rain_lower, 'upper': rain_upper},
        'ndvi': {'lower': ndvi_lower, 'upper': ndvi_upper},
        'lst': {'lower': lst_lower, 'upper': lst_upper}
    }

def plot_results(historical_data, forecasts, ci_dict, selection):
    """
    Plot forecasting results
    """
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 12))
    
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    
    # Rainfall
    ax1.plot(historical_data.index, historical_data['RAINFALL_MM'], 
             label='Historical', linewidth=2, color=colors[0], alpha=0.8)
    ax1.plot(forecasts.index, forecasts['RAINFALL_MM'], 
             label='Forecast', linewidth=3, color=colors[0])
    if ci_dict['rainfall']['lower'] is not None:
        ax1.fill_between(forecasts.index, ci_dict['rainfall']['lower'], ci_dict['rainfall']['upper'],
                        color=colors[0], alpha=0.2, label='95% CI')
    ax1.set_ylabel('Rainfall (mm)', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Rainfall Forecast', fontweight='bold')
    
    # NDVI
    ax2.plot(historical_data.index, historical_data['NDVI_VALUE'], 
             label='Historical', linewidth=2, color=colors[1], alpha=0.8)
    ax2.plot(forecasts.index, forecasts['NDVI_VALUE'], 
             label='Forecast', linewidth=3, color=colors[1])
    if ci_dict['ndvi']['lower'] is not None:
        ax2.fill_between(forecasts.index, ci_dict['ndvi']['lower'], ci_dict['ndvi']['upper'],
                        color=colors[1], alpha=0.2, label='95% CI')
    ax2.set_ylabel('NDVI Value', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_title('Vegetation Health Forecast', fontweight='bold')
    
    # LST
    ax3.plot(historical_data.index, historical_data['LST_VALUE'], 
             label='Historical', linewidth=2, color=colors[2], alpha=0.8)
    ax3.plot(forecasts.index, forecasts['LST_VALUE'], 
             label='Forecast', linewidth=3, color=colors[2])
    if ci_dict['lst']['lower'] is not None:
        ax3.fill_between(forecasts.index, ci_dict['lst']['lower'], ci_dict['lst']['upper'],
                        color=colors[2], alpha=0.2, label='95% CI')
    ax3.set_ylabel('Temperature (°C)', fontweight='bold')
    ax3.set_xlabel('Year', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_title('Temperature Forecast', fontweight='bold')
    
    plt.suptitle(f'5-Year Forecast for {selection["name"]}\n', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

def generate_summary(forecasts, selection):
    """
    Generate forecast summary
    """
    print("\n" + "=" * 60)
    print(f"📊 FORECAST SUMMARY: {selection['name']}")
    print("=" * 60)
    
    print(f"\n📅 Forecast Period: {forecasts.index[0].strftime('%Y-%m-%d')} to {forecasts.index[-1].strftime('%Y-%m-%d')}")
    
    print(f"\n🌧️  Rainfall Forecast:")
    print(f"   Average: {forecasts['RAINFALL_MM'].mean():.1f} mm")
    print(f"   Range: {forecasts['RAINFALL_MM'].min():.1f} - {forecasts['RAINFALL_MM'].max():.1f} mm")
    
    print(f"\n🌿 Vegetation Health (NDVI):")
    print(f"   Average: {forecasts['NDVI_VALUE'].mean():.3f}")
    print(f"   Range: {forecasts['NDVI_VALUE'].min():.3f} - {forecasts['NDVI_VALUE'].max():.3f}")
    
    print(f"\n🌡️  Temperature Forecast:")
    print(f"   Average: {forecasts['LST_VALUE'].mean():.1f}°C")
    print(f"   Range: {forecasts['LST_VALUE'].min():.1f} - {forecasts['LST_VALUE'].max():.1f}°C")

def main():
    """
    Main interactive forecasting system
    """
    print("🌍 WELCOME TO THE CLIMATE FORECASTING SYSTEM")
    print("=" * 50)
    print("This system predicts Rainfall, Vegetation Health, and Temperature")
    print("for any available country or region for the next 5 years!")
    
    # Get available options
    available_countries, region_mapping = get_available_options(combined_df)
    
    while True:
        # Display options and get user selection
        display_available_options(available_countries, region_mapping)
        selection = get_user_selection(available_countries, region_mapping)
        
        if selection is None:
            print("Thank you for using the Climate Forecasting System! 👋")
            break
        
        # Prepare data
        data = prepare_data_for_forecast(combined_df, selection)
        if data is None:
            continue
        
        # Run forecast
        forecasts, ci_dict = run_forecast(data, selection)
        
        if forecasts is None:
            print("❌ Forecasting failed. Please try another region.")
            continue
        
        # Plot results
        plot_results(data, forecasts, ci_dict, selection)
        
        # Generate summary
        generate_summary(forecasts, selection)
        
        # Save results
        filename = f"forecast_{selection['name'].replace(' ', '_').replace('/', '_').lower()}.csv"
        forecasts.to_csv(filename)
        print(f"\n💾 Forecast saved to: {filename}")
        
        # Ask if user wants to continue
        print("\n" + "=" * 50)
        continue_choice = input("Would you like to forecast another region? (yes/no): ").strip().lower()
        if continue_choice not in ['yes', 'y', '']:
            print("Thank you for using the Climate Forecasting System! 👋")
            break

# Run the interactive system
if __name__ == "__main__":
    main()