# All-derpin-data-challenge-2025
# Malawi Forecasts - Crop Prediction System

## Overview

Malawi Forecasts is a comprehensive early warning prediction system, designed to use machine learning models to make predictions, on which might be the best crop to plant in a certain region, at a time in a period of 5 years. Leveraging advanced statistical modeling with the SARIMAX (Seasonal AutoRegressive Integrated Moving Average with eXogenous factors) algorithm, this project forecasts key climate variables—rainfall, vegetation health (NDVI), and temperature—over a 5-year horizon. 

Integrated with a Flask-based web application, it provides an intuitive interface for users to explore national or regional forecasts, receive crop recommendations (e.g Rice, Cassava, Maize), and assess least suitable crops based on predicted conditions. The system processes historical data from a CSV file, generates detailed visualizations, and exports results, making it a vital tool for agricultural planning and climate resilience.

## Features

- **Climate Forecasting**: Employs SARIMAX to predict rainfall, NDVI, and temperature with 95% confidence intervals for the next 5 years, based on historical trends and seasonal patterns.
- **Crop Recommendations**: Analyzes forecasted temperature and NDVI to suggest the top three most suitable crops and identify the three least recommended crops, complete with reasoning for each recommendation.
- **Model Evaluation**: Provides a robust evaluation framework, calculating accuracy, precision, recall, F1-score, and visualizing a confusion matrix to assess the crop recommendation model's performance.
- **Interactive Web Application**: Features a Flask-powered interface with pages for home, about, help, region selection, and forecast results, ensuring accessibility for all users.
- **Data Visualization**: Generates high-quality plots comparing historical data with forecasts, including confidence intervals, to aid in decision-making.
- **Data Export**: Automatically saves forecast results to a CSV file (e.g., `forecast_<region>.csv`) for further analysis or record-keeping.
- **User-Friendly Navigation**: Includes intuitive buttons and prompts to guide users through forecasting options and result interpretation.

## Requirements

- **Python 3.x**: Ensure a compatible Python environment is installed.
- **Required Libraries**:
  - `pandas` for data manipulation and analysis
  - `numpy` for numerical computations
  - `matplotlib` and `seaborn` for data visualization
  - `statsmodels` for SARIMAX modeling
  - `scikit-learn` for model evaluation metrics
  - `flask` for web application development

Install dependencies using:
```bash
pip install pandas numpy matplotlib seaborn statsmodels scikit-learn flask
```

## Installation

1. **Clone or Download**: Obtain the project files by cloning the repository or downloading:
   - `main.py`
   - `crop_forecasting_system.py`
   - `templates/*.html`
   - `combined_with_plants.csv`
   - `foreccast3.ipynb`

2. **Data Preparation**: Place the `combined_with_plants.csv` file in the working directory. This file should contain historical climate data with columns for DATE, GID_2, RAINFALL_MM, NDVI_VALUE, LST_VALUE, and optionally RECOMMENDED.

3. **Run the Application**: Launch the Flask app by executing:
   ```bash
   python main.py
   ```

4. **Access the App**: Open a web browser and navigate to `http://127.0.0.1:5000/` to start using the system.

## Usage

### Web Application

- **Home Page**: Visit the root URL to access an introduction to Malawi Forecasts. Choose "Try it out - Forecast Malawi" for a national overview or "Select a specific state" for regional details.
- **Region Selection**: On the selection page, pick a region (Northern, Central, Southern) to generate a tailored forecast. Use the "View Forecast" button to proceed.
- **Forecast Pages**: Displays a loading message followed by forecast results, including plots and summaries. National forecasts cover 3 years, while regional forecasts extend to 5 years.
- **About Page**: Provides background on the system's goals, leveraging advanced analytics and machine learning for resilience.
- **Help Page**: Offers step-by-step guidance on using the website, interpreting results, and contacting support.

### Jupyter Notebook (foreccast3.ipynb)

- Open the notebook in a Jupyter environment to run the backend code interactively.
- Execute cells to load data, evaluate the model, generate forecasts, and visualize results.
- Ideal for developers or analysts wanting to experiment with the code or test modifications.

### Data Processing Workflow

1. **Data Loading**: The system reads `combined_with_plants.csv` and preprocesses it, converting dates and ensuring numeric integrity.
2. **Forecasting**: Applies SARIMAX with a seasonal period of 23 (based on 16-day intervals) to predict future climate variables.
3. **Analysis**: Computes averages, ranges, and crop suitability based on thresholds (e.g., temperature > 29°C favors Cassava).

## Data Format

The `combined_with_plants.csv` file is the backbone of the system, requiring the following structure:

- `UID`: Unique identifier (e.g., MWI.1.1_1_1/1/2022)
- `GID_2`: Region code (e.g., MWI.1.1_1)
- `DATE`: Observation date in mm/dd/yyyy format
- `RAINFALL_MM`: Rainfall in millimeters
- `NDVI_VALUE`: Normalized Difference Vegetation Index (0 to 1)
- `LST_VALUE`: Land Surface Temperature in Celsius
- `RECOMMENDED`: Optional column with actual crop labels (e.g., Rice, Maize, Cassava) for model evaluation

## Output

- **Visualizations**: Three-panel plots displaying historical and forecasted rainfall, NDVI, and temperature, with shaded 95% confidence intervals.
- **Summary Report**: Includes forecast period (e.g., 2025-2030), average and range of values, top and least recommended crops with reasons, and prediction accuracy (if evaluated).
- **CSV Export**: Saves forecast data to a file named `forecast_<region_name>.csv`, preserving the index and predicted values.

## Notes

- **Data Requirements**: At least 24 records (2 years at 16-day intervals) are needed for reliable SARIMAX forecasting.
- **Thresholds**: Crop recommendations use thresholds like 29°C for temperature and 0.55 for NDVI to determine suitability.
- **Scalability**: Currently optimized for Malawi regions (Northern, Central, Southern) but can be extended with additional data.
- **Current Date**: The system reflects data and operations as of 07:58 PM EAT, August 21, 2025.
- **Limitations**: Accuracy depends on data quality and the presence of a RECOMMENDED column for evaluation.

## Contributing

We welcome contributions to enhance Malawi Forecasts! Please:

1. Fork the repository
2. Create a branch for your feature or bug fix
3. Commit changes and submit a pull request with a clear description
4. Adhere to the project's coding standards and include tests if applicable

## License

This project is licensed under the MIT License. See the LICENSE file for full details.

## Acknowledgments

- Data sourced from various climate and agricultural datasets for Malawi
- Inspired by the need for localized climate solutions in Sub-Saharan Africa
- Thanks to the open-source community for tools like Flask, pandas, and statsmodels

## Future Enhancements

- Integrate real-time weather data feeds
- Expand crop options
- Add mobile app support for broader accessibility
- Implement user authentication for personalized forecasts


## Contributers
1. Stephen W. Austine |https://github.com/Stephen-Austine
2. Andy E. Hadulo     |https://github.com/Hadulo
3. George M. Rading   |https://github.com/QazGeo
