# starlink_istanbul/mapping.py

import pandas as pd
import folium


def map_service_zones(input_excel: str, output_html: str):
    """
    Load a geocoded neighborhoods file, assign purchasing power and earthquake risk
    by district, define a service zone string, then draw a Folium map with colored
    CircleMarkers for each neighborhood according to its service zone.
    """
    df = pd.read_excel(input_excel)

    high_income = [
        "Beşiktaş", "Kadıköy", "Sarıyer", "Şişli", "Ataşehir", "Beykoz", "Üsküdar", "Bakırköy"
    ]
    low_income = [
        "Esenler", "Sultangazi", "Arnavutköy", "Bağcılar",
        "Esenyurt", "Gaziosmanpaşa", "Sultanbeyli"
    ]
    df["Purchasing_Power"] = df["District"].apply(
        lambda d: "High" if d in high_income else ("Low" if d in low_income else "Medium")
    )

    high_risk = [
        "Avcılar", "Bakırköy", "Küçükçekmece", "Zeytinburnu",
        "Fatih", "Bahçelievler", "Beşiktaş", "Beylikdüzü"
    ]
    df["Earthquake_Risk"] = df["District"].apply(
        lambda d: "High" if d in high_risk else "Low"
    )

    def assign_zone(r):
        if r["Purchasing_Power"] == "High" and r["Earthquake_Risk"] == "High":
            return "Zone 1: High Power + High Risk"
        if r["Purchasing_Power"] == "Low" and r["Earthquake_Risk"] == "High":
            return "Zone 2: Low Power + High Risk"
        if r["Purchasing_Power"] == "High" and r["Earthquake_Risk"] == "Low":
            return "Zone 3: High Power + Low Risk"
        return "Other"

    df["Service_Zone"] = df.apply(assign_zone, axis=1)
    df_clean = df.dropna(subset=["Latitude", "Longitude"])

    zone_colors = {
        "Zone 1: High Power + High Risk": "red",
        "Zone 2: Low Power + High Risk": "orange",
        "Zone 3: High Power + Low Risk": "green",
        "Other": "gray"
    }

    m = folium.Map(location=[41.0082, 28.9784], zoom_start=10)
    for _, row in df_clean.iterrows():
        color = zone_colors[row["Service_Zone"]]
        folium.CircleMarker(
            location=[row["Latitude"], row["Longitude"]],
            radius=4,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            popup=f"{row['Neighborhood']} ({row['District']})<br>{row['Service_Zone']}"
        ).add_to(m)

    m.save(output_html)
    print(f"✅ Service zones map saved as {output_html}")
