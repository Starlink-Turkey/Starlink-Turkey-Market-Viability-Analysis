# starlink_istanbul/clustering.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import folium


def plot_elbow_method(excel_path: str, skiprows: int = 1, k_max: int = 10):
    """
    Load population data, scale features, run KMeans inertia for k=1..k_max,
    then plot the elbow curve.
    """
    df = pd.read_excel(excel_path, skiprows=skiprows)
    X = df.iloc[:, [4, 5, 6]]  # Population_Density, Latitude, Longitude

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    inertia = []
    K_range = range(1, k_max + 1)
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42)
        km.fit(X_scaled)
        inertia.append(km.inertia_)

    plt.figure(figsize=(8, 5))
    sns.lineplot(x=list(K_range), y=inertia, marker="o")
    plt.title("Elbow Method for Optimal k")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Inertia")
    plt.grid(True)
    plt.show()


def cluster_districts_and_map(excel_path: str, skiprows: int = 1, n_clusters: int = 4):
    """
    Load the district-level data (no headers, skiprows),
    run KMeans on Population_Density, Latitude, Longitude,
    then render a Folium map with one CircleMarker per district,
    colored by cluster ID.
    """
    df = pd.read_excel(excel_path, header=None, skiprows=skiprows)
    df.columns = [
        "Nr_Plates", "District", "Population", "Area_km2",
        "Population_Density", "Latitude", "Longitude"
    ]

    X = df[["Population_Density", "Latitude", "Longitude"]]
    km = KMeans(n_clusters=n_clusters, random_state=42)
    df["Cluster"] = km.fit_predict(X)

    m = folium.Map(location=[41.01, 28.95], zoom_start=10)
    colors = ["red", "blue", "green", "orange"]

    for _, row in df.iterrows():
        folium.CircleMarker(
            location=[row["Latitude"], row["Longitude"]],
            radius=7,
            popup=folium.Popup(f"{row['District']}<br>Cluster {row['Cluster']}", max_width=200),
            color=colors[row["Cluster"] % len(colors)],
            fill=True,
            fill_opacity=0.8
        ).add_to(m)

    m.save("data/istanbul_district_clusters.html")
    print("✅ District clusters map saved as istanbul_district_clusters.html")


def cluster_neighborhoods_and_map(input_excel: str, output_html: str, n_clusters: int = 5):
    """
    Load a geocoded neighborhood file, drop rows without latitude/longitude,
    scale features (Population_Density, Latitude, Longitude), run KMeans,
    and save a Folium map with neighborhoods colored by cluster.
    """
    df = pd.read_excel(input_excel)
    df = df.dropna(subset=["Latitude", "Longitude"])

    X = df[["Population_Density", "Latitude", "Longitude"]]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df["Cluster"] = km.fit_predict(X_scaled)

    m = folium.Map(location=[41.0082, 28.9784], zoom_start=10)
    colors = ["red", "blue", "green", "purple", "orange"]

    for _, row in df.iterrows():
        folium.CircleMarker(
            location=[row["Latitude"], row["Longitude"]],
            radius=5,
            popup=folium.Popup(
                f"{row['Neighborhood']} ({row['District']})<br>Cluster {row['Cluster']}",
                max_width=250
            ),
            color=colors[row["Cluster"] % len(colors)],
            fill=True,
            fill_opacity=0.7
        ).add_to(m)

    m.save(output_html)
    print(f"✅ Neighborhood clusters map saved as {output_html}")
