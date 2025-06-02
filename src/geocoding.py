# starlink_istanbul/geocoding.py

import pandas as pd
from geopy.geocoders import Nominatim
import time


def geocode_neighborhoods(input_excel: str, output_excel: str, sheet_name: str = None):
    """
    Read an Excel sheet of neighborhoods, use Nominatim to fetch latitude & longitude
    for each (Neighborhood, District). Rate-limit at 1s per request. Save to output.
    """
    df = pd.read_excel(input_excel, sheet_name=sheet_name)
    geolocator = Nominatim(user_agent="istanbul_geocoder")

    def get_coordinates(row):
        query = f"{row['Neighborhood']}, {row['District']}, Istanbul, Turkey"
        try:
            loc = geolocator.geocode(query)
            time.sleep(1)  # avoid throttling
            if loc:
                return pd.Series([loc.latitude, loc.longitude])
        except:
            pass
        return pd.Series([None, None])

    df[["Latitude", "Longitude"]] = df.apply(get_coordinates, axis=1)
    df.to_excel(output_excel, index=False)
    print(f"✅ Geocoding complete. File saved as {output_excel}")
