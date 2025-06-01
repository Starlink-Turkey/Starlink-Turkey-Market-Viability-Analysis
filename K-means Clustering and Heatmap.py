#!/usr/bin/env python
# coding: utf-8

# In[ ]:


pip install geopy pandas openpyxl


# # Starlink Research
# ## K-means Clustering and Heatmap Generation for Istanbul

# ### Elbow Method for Optimal K

# In[9]:


# 1. Import libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sns

# 2. Load Excel data (skip header row)
df = pd.read_excel("istanbul_population.xlsx", skiprows=1)

# 3. Select features for clustering: Population Density, Latitude, Longitude
X = df.iloc[:, [4, 5, 6]]  # 4: pop density, 5: lat, 6: long

# 4. Optional: normalize if needed (sklearn KMeans handles scale poorly)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5. Use Elbow Method to find optimal number of clusters
inertia = []
K_range = range(1, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# 6. Plot Elbow Curve
plt.figure(figsize=(8, 5))
sns.lineplot(x=K_range, y=inertia, marker='o')
plt.title("Elbow Method For Optimal k")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Inertia")
plt.grid(True)
plt.show()


# ### District Clustering by Population Density

# ### Visualise Clusters on a Choropleth
# 
# Cluster 0: Likely low-density areas → Potential for expansion
# 
# Cluster 1: Medium-density areas → Growing demand
# 
# Cluster 2: High-density areas → Priority zones for service deployment

# In[2]:


import pandas as pd
import folium
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load the Excel file (no headers)
df = pd.read_excel("istanbul_population.xlsx", header=None, skiprows=1)

# Set correct column names
df.columns = [
    'Nr_Plates', 'District', 'Population', 'Area_km2',
    'Population_Density', 'Latitude', 'Longitude'
]

# --- KMeans clustering ---
X = df[['Population_Density', 'Latitude', 'Longitude']]
kmeans = KMeans(n_clusters=4, random_state=42)
df['Cluster'] = kmeans.fit_predict(X)

# --- Plot on folium map ---
m = folium.Map(location=[41.01, 28.95], zoom_start=10)
colors = ['red', 'blue', 'green', 'orange']

for _, row in df.iterrows():
    folium.CircleMarker(
        location=[row['Latitude'], row['Longitude']],
        radius=7,
        popup=folium.Popup(f"{row['District']}<br>Cluster {row['Cluster']}", max_width=200),
        color=colors[row['Cluster'] % len(colors)],
        fill=True,
        fill_opacity=0.8
    ).add_to(m)

m


# ### Geocoding: Istanbul Neighborhoods Coordinates

# In[3]:


import pandas as pd
from geopy.geocoders import Nominatim
import time

# Load your Excel file
df = pd.read_excel("istanbul_population.xlsx", sheet_name="Neighborhoods")  # adjust sheet name if needed

# Set up geopy
geolocator = Nominatim(user_agent="istanbul_geocoder")

# Function to fetch coordinates
def get_coordinates(row):
    try:
        query = f"{row['Neighborhood']}, {row['District']}, Istanbul, Turkey"
        location = geolocator.geocode(query)
        time.sleep(1)  # avoid throttling
        if location:
            return pd.Series([location.latitude, location.longitude])
    except:
        pass
    return pd.Series([None, None])

# Apply function
df[['Latitude', 'Longitude']] = df.apply(get_coordinates, axis=1)

# Save the results
df.to_excel("istanbul_neighborhoods_with_coords.xlsx", index=False)
print("✅ Geocoding complete. File saved as istanbul_neighborhoods_with_coords.xlsx")


# ### Mapping: Istanbul Neighborhood Clusters
# 
# | Cluster ID | Color     | Geographic Distribution                          | Population Density | Interpretation                                         | Service Priority                    |
# | ---------- | --------- | ------------------------------------------------ | ------------------ | ------------------------------------------------------ | ----------------------------------- |
# | **0**      | 🔴 Red    | Far **north & rural**: Şile, Beykoz, Çatalca     | Low                | **Sparse, rural, forest areas** (low density + remote) | 🔹 Low (hard to serve, few users)   |
# | **1**      | 🔵 Blue   | Dense **western core**: Bağcılar, Esenler        | High               | **Urban cores**, highest density, small area           | 🔺 Highest (need intensive service) |
# | **2**      | 🟢 Green  | Mixed western & south: Küçükçekmece, Esenyurt    | Medium-High        | **Growing suburbs**, diverse density                   | 🔸 Medium-High                      |
# | **3**      | 🟣 Purple | Western periphery: Silivri, Çatalca, rural areas | Low                | **Rural/coastal periphery**, sparse settlements        | 🔹 Low                              |
# | **4**      | 🟡 Yellow | Dense **southern band**: Pendik, Kartal, Maltepe | Medium-High        | **Dense residential belts**, transit corridors         | 🔸 Medium-High                      |
# 

# In[9]:


import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import folium

# Load data
df = pd.read_excel("istanbul_neighborhoods_with_coords.xlsx")

# Drop missing coordinates
df = df.dropna(subset=['Latitude', 'Longitude'])

# Select and scale features
X = df[['Population_Density', 'Latitude', 'Longitude']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# KMeans clustering
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Create folium map
m = folium.Map(location=[41.0082, 28.9784], zoom_start=10)
colors = ['red', 'blue', 'green', 'purple', 'orange']

for _, row in df.iterrows():
    folium.CircleMarker(
        location=[row['Latitude'], row['Longitude']],
        radius=5,
        popup=folium.Popup(f"{row['Neighborhood']} ({row['District']})<br>Cluster {row['Cluster']}", max_width=250),
        color=colors[row['Cluster'] % len(colors)],
        fill=True,
        fill_opacity=0.7
    ).add_to(m)

m.save("istanbul_neighborhood_clusters.html")
print("✅ Map saved as istanbul_neighborhood_clusters.html")


# ### Service Zones According to Earthquake Risks and Income
# 
# Zone colors:
# 
# 🔴 Red: Zone 1 (High Power + High Risk)
# 
# 🟠 Orange: Zone 2 (Low Power + High Risk)
# 
# 🟢 Green: Zone 3 (High Power + Low Risk)
# 
# ⚫ Gray: Other

# In[8]:


import pandas as pd
import folium

# Load the Excel file (replace with your actual file name)
file_path = "istanbul_neighborhoods_with_coords.xlsx"
df = pd.read_excel(file_path)

# Assign purchasing power (proxy by district)
high_income_districts = [
    "Beşiktaş", "Kadıköy", "Sarıyer", "Şişli", "Ataşehir", "Beykoz", "Üsküdar", "Bakırköy"
]
low_income_districts = [
    "Esenler", "Sultangazi", "Arnavutköy", "Bağcılar", "Esenyurt", "Gaziosmanpaşa", "Sultanbeyli"
]
df['Purchasing_Power'] = df['District'].apply(
    lambda d: 'High' if d in high_income_districts else ('Low' if d in low_income_districts else 'Medium')
)

# Assign earthquake risk (proxy by district)
high_risk_districts = [
    "Avcılar", "Bakırköy", "Küçükçekmece", "Zeytinburnu", "Fatih", "Bahçelievler", "Beşiktaş", "Beylikdüzü"
]
df['Earthquake_Risk'] = df['District'].apply(
    lambda d: 'High' if d in high_risk_districts else 'Low'
)

# Assign service zone
def assign_service_zone(row):
    if row['Purchasing_Power'] == 'High' and row['Earthquake_Risk'] == 'High':
        return 'Zone 1: High Power + High Risk'
    elif row['Purchasing_Power'] == 'Low' and row['Earthquake_Risk'] == 'High':
        return 'Zone 2: Low Power + High Risk'
    elif row['Purchasing_Power'] == 'High' and row['Earthquake_Risk'] == 'Low':
        return 'Zone 3: High Power + Low Risk'
    else:
        return 'Other'

df['Service_Zone'] = df.apply(assign_service_zone, axis=1)

# Remove rows with missing coordinates
df_cleaned = df.dropna(subset=['Latitude', 'Longitude'])

# Define zone colors
zone_colors = {
    'Zone 1: High Power + High Risk': 'red',
    'Zone 2: Low Power + High Risk': 'orange',
    'Zone 3: High Power + Low Risk': 'green',
    'Other': 'gray'
}

# Create the Folium map centered on Istanbul
m = folium.Map(location=[41.0082, 28.9784], zoom_start=10)

# Add CircleMarkers
for _, row in df_cleaned.iterrows():
    folium.CircleMarker(
        location=[row['Latitude'], row['Longitude']],
        radius=4,
        color=zone_colors.get(row['Service_Zone'], 'black'),
        fill=True,
        fill_color=zone_colors.get(row['Service_Zone'], 'black'),
        fill_opacity=0.7,
        popup=f"{row['Neighborhood']} ({row['District']})<br>{row['Service_Zone']}"
    ).add_to(m)

# Save map
m.save("istanbul_service_zones_map.html")
print("✅ Map saved as istanbul_service_zones_map.html")


# ## Linear Programming Formulation (Set Cover for Starlink Service Zones)

# \begin{align*}
# \mbox{minimize} \;\;& \sum\limits_{i = 1}^{N} c_{ij} x_{ij} \\
# \mbox{subject to:} \;\;& \sum\limits_{j = 1}^{N} x_{ij} = 1 \;\;\;\; i = 1, 2, \dots, N\\
# \;\;& \sum\limits_{i = 1}^{N} x_{ij} = 1 \;\;\;\; j = 1, 2, \dots, N\\
# \;\;& x_{ij} \in \{0, 1\} \;\;\;\; i = 1, 2, \dots, N; j = 1, 2, \dots, N
# \end{align*}

# In[11]:


import pandas as pd
import numpy as np
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpBinary, PULP_CBC_CMD
from geopy.distance import geodesic

# Load data
df = pd.read_excel("istanbul_neighborhoods_with_coords.xlsx")
service_points = pd.read_excel("istanbul_service_points.xlsx")

# Drop missing coordinates
df = df.dropna(subset=['Latitude', 'Longitude'])

# Parameters
R = 10  # Coverage radius in km (example)
demand_points = df[['Neighborhood', 'District', 'Latitude', 'Longitude', 'Population']].reset_index(drop=True)

# Calculate coverage matrix
coverage = {}
for i, d in demand_points.iterrows():
    for j, s in service_points.iterrows():
        dist = geodesic((d['Latitude'], d['Longitude']), (s['Latitude'], s['Longitude'])).km
        if dist <= R:
            coverage.setdefault(i, []).append(j)

# PuLP Model
model = LpProblem("Starlink_Set_Covering", LpMinimize)
x = {j: LpVariable(f"x_{j}", cat=LpBinary) for j in service_points.index}
y = {i: LpVariable(f"y_{i}", cat=LpBinary) for i in demand_points.index}

# Objective: Minimize site count
model += lpSum(x[j] for j in x)

# Constraint 1: Coverage assignment
for i in y:
    if i in coverage:
        model += y[i] <= lpSum(x[j] for j in coverage[i])
    else:
        model += y[i] == 0  # No coverage

# Constraint 2: ≥95% population covered
total_pop = demand_points['Population'].sum()
model += lpSum(demand_points.loc[i, 'Population'] * y[i] for i in y) >= 0.95 * total_pop

# Solve
solver = PULP_CBC_CMD(msg=True)
model.solve(solver)

# Results
covered_sites = [j for j in x if x[j].varValue > 0.5]
print(f"✅ Optimal sites: {covered_sites}")
print(f"Sites placed at coordinates:\n{service_points.loc[covered_sites, ['Latitude', 'Longitude']]}")

# Greedy heuristic fallback: sort by population covered
site_coverage = []
for j in service_points.index:
    covered_pop = sum(
        demand_points.loc[i, 'Population']
        for i in coverage if j in coverage[i]
    )
    site_coverage.append((j, covered_pop))

site_coverage = sorted(site_coverage, key=lambda x: -x[1])
greedy_sites = [j for j, _ in site_coverage[:5]]

print(f"🛡️ Greedy fallback sites: {greedy_sites}")


# In[ ]:




