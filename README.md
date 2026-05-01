# Malawi Forecasts - Crop Prediction System

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-WebApp-orange.svg)](https://flask.palletsprojects.com/)

> **Hackathon Award - 3rd Place**
>
> This project was developed during the All-Derpin Data Challenge 2025 and secured **3rd place**, earning recognition for its innovative approach to agricultural climate forecasting in Malawi.

---

## Overview

**Malawi Forecasts** is an intelligent early warning crop prediction system designed to help farmers and agricultural planners make data-driven decisions about crop selection. By leveraging advanced time-series forecasting (SARIMAX) and machine learning, the system predicts key climate variables—rainfall, vegetation health (NDVI), and land surface temperature—up to 5 years ahead.

The system provides **crop recommendations** (Rice, Cassava, Maize) based on forecasted conditions, helping stakeholders plan for food security and agricultural resilience in changing climates.

---

## Hackathon Achievement

| Achievement | Details |
|-------------|---------|
| **Position** | 3rd Place |
| **Award** | All-Derpin Data Challenge 2025 |
| **Team** | Stephen W. Austine, Andy E. Hadulo, George M. Rading |

We are proud to have developed a solution that addresses real agricultural challenges in Malawi through data-driven insights.

---

## Key Features

### Climate Forecasting
- **SARIMAX Modeling**: Predicts rainfall, NDVI, and temperature with 95% confidence intervals
- **Multi-horizon forecasts**: National (3 years) and regional (5 years) predictions
- **Auto-training**: Automatically trains and caches models for future use

### Crop Recommendations
- **Intelligent analysis**: Recommends top 3 suitable crops based on forecasted conditions
- **Least recommended**: Identifies crops to avoid under predicted conditions
- **Reasoning**: Provides detailed explanations for each recommendation

### Model Evaluation
- **Comprehensive metrics**: Accuracy, Precision, Recall, F1-Score
- **Confusion matrix**: Visual assessment of model performance
- **Historical validation**: Evaluates against actual crop data

### Interactive Web Application
- **Flask-powered**: Modern, responsive web interface
- **Region selection**: National and regional forecasts
- **Visualizations**: Interactive charts with confidence intervals
- **Export options**: Download forecasts as CSV or PDF reports

---

## Quick Start

### Prerequisites

```bash
# Python 3.x required
python --version

# Install dependencies
pip install pandas numpy matplotlib seaborn statsmodels scikit-learn flask joblib reportlab
```

### Run the Application

```bash
# Start the Flask web server
python main.py

# Open in browser
http://127.0.0.1:5000/
```

### Command-Line Usage

```bash
# Interactive mode
python main.py
```

---

## Project Structure

```
.
├── main.py                    # Main application & Flask server
├── LICENSE                    # MIT License
├── README.md                  # This file
├── combined_with_plants.csv   # Climate & crop data
├── rainfall_ndvi_lst_cleaned.csv
│
├── models/                     # Trained ML models (.joblib)
│   ├── lst_*.joblib           # Temperature models
│   ├── ndvi_*.joblib         # Vegetation models
│   └── rainfall_*.joblib     # Rainfall models
│
├── templates/                  # Flask HTML templates
│   ├── index.html            # Home page
│   ├── selection.html       # Region selection
│   ├── results.html         # Forecast results
│   └── ...
│
├── static/
│   ├── images/              # Generated visualizations
│   └── js/                 # JavaScript assets
│
└── Datasets/               # Raw climate data
    ├── Daily_temp/          # Temperature readings
    ├── GC_org/             # Ground climate data
    └── GC_final/           # Processed datasets
```

---

## Data Format

The system processes climate data with the following structure:

| Column | Description | Example |
|--------|-------------|---------|
| `UID` | Unique identifier | `MWI.1.1_1_1/1/2022` |
| `GID_2` | Region code | `MWI.1.1_1` |
| `DATE` | Observation date | `01/01/2022` |
| `RAINFALL_MM` | Rainfall (mm) | `45.2` |
| `NDVI_VALUE` | Vegetation index | `0.65` |
| `LST_VALUE` | Temperature (°C) | `28.5` |
| `RECOMMENDED` | Actual crop (for eval) | `Rice` |

---

## Usage Examples

### Web Application

1. Visit `http://127.0.0.1:5000/`
2. Choose **National Forecast** or **Select Region**
3. View interactive forecasts with visualizations
4. Download reports in CSV or PDF format

### Programmatic Usage

```python
from main import run_forecast_pipeline, recommend_crop

# Get forecast for a region
selection = {'type': 'region', 'value': 'MWI.2.1_1', 'name': 'Central Region - Kasungu'}
forecasts, ci_dict, summary = run_forecast_pipeline(selection)

# Get crop recommendations
recommendations = recommend_crop(28.5, 0.65)
# Output: [('Maize', '...'), ('Rice', '...'), ('Cassava', '...')]
```

---

## Technology Stack

| Component | Technology |
|-----------|------------|
| **Language** | Python 3.x |
| **Web Framework** | Flask |
| **Forecasting** | SARIMAX (statsmodels) |
| **ML Evaluation** | scikit-learn |
| **Data Processing** | pandas, numpy |
| **Visualization** | matplotlib, seaborn |
| **PDF Generation** | reportlab |

---

## Contributors

| Name | GitHub |
|------|-------|
| Stephen W. Austine | [Stephen-Austine](https://github.com/Stephen-Austine) |
| Andy E. Hadulo | [Hadulo](https://github.com/Hadulo) |
| George M. Rading | [QazGeo](https://github.com/QazGeo) |

---

## Future Enhancements

- [ ] Real-time weather API integration
- [ ] Extended crop database
- [ ] Mobile application
- [ ] User authentication
- [ ] Multi-country support
- [ ] Advanced ensemble forecasting

---

## License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- **All-Derpin Data Challenge 2025** for the opportunity
- Malawi climate data providers
- Open-source community (Flask, pandas, statsmodels)

---

<p align="center">
  <sub>Built with love for Malawi Agriculture</sub>
</p>
