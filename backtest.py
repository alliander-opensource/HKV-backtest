# imports

import pandas as pd 
from datetime import datetime
import openstef
from openstef.data_classes.model_specifications import ModelSpecificationDataClass
from openstef.data_classes.prediction_job import PredictionJobDataClass 
from openstef.pipeline.train_create_forecast_backtest import train_model_and_forecast_back_test
from openstef.feature_engineering.weather_features import calculate_dni, calculate_gti

from tqdm.autonotebook import trange

from get_rcdata import get_rcdataframe, find_nearest

import xarray as xr

def main():

    # open knmi stations
    raycast_data = xr.open_dataset('data/raycast_test_knmi.nc') 
    knmi_stations = pd.read_csv('data/knmi_stations.csv', index_col=0)
    
    # Quantiles
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95]

    # leadtimes 
    leads = [1, 3, 6]

    # backtest parameters
    backtest_iterations = 10
    backtest_horizon = 24
    backtest_folds = 4

    # Create a dataframe with station information
    locations_data = {
        "station": ["sun_heavy", "wind_heavy", "wind_sun_heavy", "consumption_heavy"],
        "lat": [53.445448, 52.515094, 53.325670, 52.30096],
        "lon": [5.7226894, 5.4768915, 5.7532568, 5.04536]
    }
    df_locations = pd.DataFrame(locations_data)

    for lead in leads:
        for i in range(len(df_locations)):
            # Get the station
            station = df_locations["station"][i]
            lat = df_locations["lat"][i]
            lon = df_locations["lon"][i]

            # Find the nearest station
            station_id = find_nearest(stations=knmi_stations, lat=lat, lon=lon)

            # raycast data 
            raycastdf = get_rcdataframe(station=station_id, data=raycast_data, lead=lead).drop(columns=['lead_time'])
            raycastdf.index = raycastdf.index + pd.DateOffset(minutes=15*lead)
            raycastdf.columns = [f"raycast_{lead}h_{col}" for col in raycastdf.columns.get_level_values(1)]
            raycastdf.index = pd.to_datetime(raycastdf.index, utc=True)

            pj = {
                "lat":53.445448,
                "lon":5.7226894,
                "id": 307,
                "name": "Back_test_prediction_job",
                "typ": "demand",
                "model": "xgb",
                "horizon_minutes": 2880,  # How many minutes in the future should be forecasted
                "resolution_minutes": 15,  # In what timestep should be forecasted
                "train_components": 1,
                "sid": "Back_test",
                "created": datetime.now(),
                "description": "",
                "forecast_type": "demand",
                "quantiles": quantiles,
                "hyper_params": {},
                "feature_names": None, 
                "model_type_group": "default",
            }

            input_data=pd.read_csv(f"data/2020_data_{station}.csv", index_col=0, parse_dates=True)
            
            for col in raycastdf.columns:
                input_data[col] = input_data.index.map(raycastdf[col].to_dict())


            calculate_dni(input_data[f"raycast_{lead}h_0.5"], pj)
            input_data[f"dni_raycast_{lead}h_0.5"] = calculate_dni(input_data[f"raycast_{lead}h_0.5"], pj)
            input_data[f"gti_raycast_{lead}h_0.5"] = calculate_gti(input_data[f"raycast_{lead}h_0.5"], pj)

            forecasts: list[pd.DataFrame] = []

            for _ in trange(backtest_iterations):
                backtest_result = train_model_and_forecast_back_test(
                    PredictionJobDataClass(**pj),
                    ModelSpecificationDataClass(**pj),
                    input_data,
                    training_horizons=[backtest_horizon],
                    n_folds=backtest_folds,
                )
                forecast = backtest_result[0]
                forecast = forecast.loc[forecast["horizon"] == backtest_horizon]
                forecast = forecast.drop(
                    columns=[
                        "pid",
                        "customer",
                        "description",
                        "type",
                        "algtype",
                        "tAhead",
                        "horizon",
                    ]
                )
                forecasts.append(forecast)
            
            forecasts_combined = pd.concat(forecasts, axis="columns")
            forecast_median = pd.DataFrame()
            for column in forecasts[0].columns:
                forecast_median[column] = forecasts_combined[column].median(axis="columns")
            forecast_median.to_csv(f"backtest_{station}_{lead}.csv")
            
    return

if __name__ == "__main__":
    main()