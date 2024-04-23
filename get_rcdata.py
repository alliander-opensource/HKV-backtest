import xarray as xr
import numpy as np
from pathlib import Path
import pandas as pd

def find_nearest(lat, lon, stations):
    """
    Find the nearest station to a given lat/lon
    """
    dists = np.sqrt((stations.lat - lat)**2 + (stations.lon - lon)**2)
    return stations.loc[dists.idxmin()].station

def get_rcdataframe(station, data, lead=None):

    if lead is None:
        dataframe = data.irradiance.sel(station=station).to_dataframe().drop(columns=['station', 'x', 'y', 'lat', 'lon']).unstack(level='quantile').astype(float)
        dataframe['lead_time'] = dataframe.index.get_level_values('lead_time') * 15 / 60
        ref_datetimes = dataframe.index.get_level_values('ref_datetime').unique()
        dataframe.index = pd.MultiIndex.from_tuples([
            (pd.to_datetime(date, utc=True),
            pd.to_datetime(date, utc=True) + pd.DateOffset(minutes=j*15))
            for date in ref_datetimes for j in range(24)
        ], names=['ref_datetime', 'valid_datetime'])

    else:
        dataframe = data.irradiance.sel(station=station, lead_time=lead).to_dataframe().drop(columns=['station', 'x', 'y', 'lat', 'lon']).unstack(level='quantile').astype(float)

    return dataframe