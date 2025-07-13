# starlink_istanbul/clustering.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import folium
import numpy as np


def plot_elbow_method(excel_path: str, skiprows: int = 1, k_max: int = 10):
    """
    Load population data, scale features, run KMeans inertia for k=1..k_max,
    then plot the elbow curve with proper error handling.
    """
    try:
        df = pd.read_excel(excel_path, skiprows=skiprows)
        
        # Validate data structure
        if df.shape[1] < 7:
            raise ValueError(f"Expected at least 7 columns, got {df.shape[1]}")
        
        X = df.iloc[:, [4, 5, 6]]  # Population_Density, Latitude, Longitude
        
        # Check for missing values
        if X.isnull().any().any():
            print("⚠️ Warning: Missing values detected, dropping rows...")
            X = X.dropna()
        
        if X.empty:
            raise ValueError("No valid data points after cleaning")
        
        print(f"📊 Using {len(X)} data points for elbow analysis")
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        inertia = []
        K_range = range(1, min(k_max + 1, len(X)))  # Can't have more clusters than data points
        
        for k in K_range:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            km.fit(X_scaled)
            inertia.append(km.inertia_)

        plt.figure(figsize=(8, 5))
        sns.lineplot(x=list(K_range), y=inertia, marker="o")
        plt.title("Elbow Method for Optimal k")
        plt.xlabel("Number of clusters (k)")
        plt.ylabel("Inertia")
        plt.grid(True)
        plt.show()
        
        print("✅ Elbow plot generated successfully")
        
    except Exception as e:
        print(f"❌ Error in elbow method: {e}")
        raise


def cluster_districts_and_map(excel_path: str, skiprows: int = 1, n_clusters: int = 4):
    """
    Load the district-level data (no headers, skiprows),
    run KMeans on Population_Density, Latitude, Longitude,
    then render a Folium map with one CircleMarker per district,
    colored by cluster ID with improved error handling.
    """
    try:
        df = pd.read_excel(excel_path, header=None, skiprows=skiprows)
        
        # Validate data structure
        if df.shape[1] < 7:
            raise ValueError(f"Expected at least 7 columns, got {df.shape[1]}")
        
        df.columns = [
            "Nr_Plates", "District", "Population", "Area_km2",
            "Population_Density", "Latitude", "Longitude"
        ]

        # Clean data
        original_count = len(df)
        df = df.dropna(subset=["Population_Density", "Latitude", "Longitude"])
        
        if df.empty:
            raise ValueError("No valid data points after cleaning")
        
        if len(df) < original_count:
            print(f"⚠️ Dropped {original_count - len(df)} rows with missing data")
        
        # Adjust clusters if needed
        actual_clusters = min(n_clusters, len(df))
        if actual_clusters < n_clusters:
            print(f"⚠️ Reducing clusters from {n_clusters} to {actual_clusters} due to limited data")

        X = df[["Population_Density", "Latitude", "Longitude"]]
        
        # Check for valid coordinate ranges
        if not (40 <= df["Latitude"].mean() <= 42 and 27 <= df["Longitude"].mean() <= 30):
            print("⚠️ Warning: Coordinates seem outside Istanbul area")
        
        km = KMeans(n_clusters=actual_clusters, random_state=42, n_init=10)
        df["Cluster"] = km.fit_predict(X)

        # Create map centered on Istanbul
        center_lat = df["Latitude"].mean()
        center_lon = df["Longitude"].mean()
        
        m = folium.Map(location=[center_lat, center_lon], zoom_start=10)
        colors = ["red", "blue", "green", "orange", "purple", "yellow", "pink", "gray"]

        for _, row in df.iterrows():
            try:
                folium.CircleMarker(
                    location=[row["Latitude"], row["Longitude"]],
                    radius=7,
                    popup=folium.Popup(f"{row['District']}<br>Cluster {row['Cluster']}", max_width=200),
                    color=colors[row["Cluster"] % len(colors)],
                    fill=True,
                    fill_opacity=0.8
                ).add_to(m)
            except Exception as e:
                print(f"⚠️ Warning: Failed to add marker for {row['District']}: {e}")
                continue

        output_path = "../data/istanbul_district_clusters.html"
        m.save(output_path)
        print(f"✅ District clusters map saved as {output_path}")
        print(f"📍 Mapped {len(df)} districts in {actual_clusters} clusters")
        
    except Exception as e:
        print(f"❌ Error in district clustering: {e}")
        raise


def cluster_neighborhoods_and_map(input_excel: str, output_html: str, n_clusters: int = 5):
    """
    Load a geocoded neighborhood file, drop rows without latitude/longitude,
    scale features (Population_Density, Latitude, Longitude), run KMeans,
    and save a Folium map with neighborhoods colored by cluster.
    Enhanced with better error handling and validation.
    """
    try:
        df = pd.read_excel(input_excel)
        
        # Validate required columns
        required_cols = ["Population_Density", "Latitude", "Longitude", "Neighborhood"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        original_count = len(df)
        df = df.dropna(subset=["Latitude", "Longitude", "Population_Density"])
        
        if df.empty:
            raise ValueError("No valid data points after cleaning")
        
        if len(df) < original_count:
            print(f"⚠️ Dropped {original_count - len(df)} rows with missing data")

        # Adjust clusters if needed
        actual_clusters = min(n_clusters, len(df))
        if actual_clusters < n_clusters:
            print(f"⚠️ Reducing clusters from {n_clusters} to {actual_clusters} due to limited data")

        X = df[["Population_Density", "Latitude", "Longitude"]]
        
        # Check for outliers
        for col in X.columns:
            q1, q3 = X[col].quantile([0.25, 0.75])
            iqr = q3 - q1
            outliers = ((X[col] < q1 - 1.5 * iqr) | (X[col] > q3 + 1.5 * iqr)).sum()
            if outliers > 0:
                print(f"⚠️ Found {outliers} outliers in {col}")
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        km = KMeans(n_clusters=actual_clusters, random_state=42, n_init=10)
        df["Cluster"] = km.fit_predict(X_scaled)

        # Create map
        center_lat = df["Latitude"].mean()
        center_lon = df["Longitude"].mean()
        
        m = folium.Map(location=[center_lat, center_lon], zoom_start=10)
        colors = ["red", "blue", "green", "purple", "orange", "yellow", "pink", "gray"]

        success_count = 0
        for _, row in df.iterrows():
            try:
                # Get district name if available
                district_name = row.get("District", "Unknown District")
                
                folium.CircleMarker(
                    location=[row["Latitude"], row["Longitude"]],
                    radius=5,
                    popup=folium.Popup(
                        f"{row['Neighborhood']} ({district_name})<br>Cluster {row['Cluster']}",
                        max_width=250
                    ),
                    color=colors[row["Cluster"] % len(colors)],
                    fill=True,
                    fill_opacity=0.7
                ).add_to(m)
                success_count += 1
            except Exception as e:
                print(f"⚠️ Warning: Failed to add marker for {row['Neighborhood']}: {e}")
                continue

        m.save(output_html)
        print(f"✅ Neighborhood clusters map saved as {output_html}")
        print(f"📍 Successfully mapped {success_count}/{len(df)} neighborhoods in {actual_clusters} clusters")
        
        # Print cluster statistics
        cluster_stats = df.groupby("Cluster").agg({
            "Population_Density": ["mean", "count"],
            "Latitude": "mean",
            "Longitude": "mean"
        }).round(3)
        print("\n📊 Cluster Statistics:")
        print(cluster_stats)
        
    except Exception as e:
        print(f"❌ Error in neighborhood clustering: {e}")
        raise
