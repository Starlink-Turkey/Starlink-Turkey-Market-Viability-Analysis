# starlink_istanbul/mapping.py

import pandas as pd
import folium


def map_service_zones(input_excel: str, output_html: str):
    """
    Load a geocoded neighborhoods file, assign purchasing power and earthquake risk
    by district, define a service zone string, then draw a Folium map with colored
    CircleMarkers for each neighborhood according to its service zone.
    Enhanced with better error handling and validation.
    """
    try:
        df = pd.read_excel(input_excel)
        
        # Validate required columns
        required_cols = ["Latitude", "Longitude", "District", "Neighborhood"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Define district classifications
        high_income = [
            "Beşiktaş", "Kadıköy", "Sarıyer", "Şişli", "Ataşehir", "Beykoz", "Üsküdar", "Bakırköy"
        ]
        low_income = [
            "Esenler", "Sultangazi", "Arnavutköy", "Bağcılar",
            "Esenyurt", "Gaziosmanpaşa", "Sultanbeyli"
        ]
        
        def classify_purchasing_power(district):
            """Classify district by purchasing power"""
            if pd.isna(district):
                return "Unknown"
            return "High" if district in high_income else ("Low" if district in low_income else "Medium")
        
        df["Purchasing_Power"] = df["District"].apply(classify_purchasing_power)

        high_risk = [
            "Avcılar", "Bakırköy", "Küçükçekmece", "Zeytinburnu",
            "Fatih", "Bahçelievler", "Beşiktaş", "Beylikdüzü"
        ]
        
        def classify_earthquake_risk(district):
            """Classify district by earthquake risk"""
            if pd.isna(district):
                return "Unknown"
            return "High" if district in high_risk else "Low"
        
        df["Earthquake_Risk"] = df["District"].apply(classify_earthquake_risk)

        def assign_zone(r):
            """Assign service zone based on purchasing power and earthquake risk"""
            if r["Purchasing_Power"] == "High" and r["Earthquake_Risk"] == "High":
                return "Zone 1: High Power + High Risk"
            elif r["Purchasing_Power"] == "Low" and r["Earthquake_Risk"] == "High":
                return "Zone 2: Low Power + High Risk"
            elif r["Purchasing_Power"] == "High" and r["Earthquake_Risk"] == "Low":
                return "Zone 3: High Power + Low Risk"
            elif r["Purchasing_Power"] == "Medium" and r["Earthquake_Risk"] == "High":
                return "Zone 4: Medium Power + High Risk"
            elif r["Purchasing_Power"] == "Medium" and r["Earthquake_Risk"] == "Low":
                return "Zone 5: Medium Power + Low Risk"
            elif r["Purchasing_Power"] == "Low" and r["Earthquake_Risk"] == "Low":
                return "Zone 6: Low Power + Low Risk"
            else:
                return "Zone 7: Other/Unknown"

        df["Service_Zone"] = df.apply(assign_zone, axis=1)
        
        # Clean data - remove rows with missing coordinates
        original_count = len(df)
        df_clean = df.dropna(subset=["Latitude", "Longitude"])
        
        if df_clean.empty:
            raise ValueError("No valid coordinates found in data")
        
        if len(df_clean) < original_count:
            print(f"⚠️ Dropped {original_count - len(df_clean)} rows with missing coordinates")

        # Validate coordinate ranges for Istanbul
        lat_range = (40.5, 41.5)
        lon_range = (28.0, 30.0)
        
        invalid_coords = (
            (df_clean["Latitude"] < lat_range[0]) | (df_clean["Latitude"] > lat_range[1]) |
            (df_clean["Longitude"] < lon_range[0]) | (df_clean["Longitude"] > lon_range[1])
        ).sum()
        
        if invalid_coords > 0:
            print(f"⚠️ Warning: {invalid_coords} points have coordinates outside expected Istanbul area")

        # Define zone colors
        zone_colors = {
            "Zone 1: High Power + High Risk": "red",
            "Zone 2: Low Power + High Risk": "orange",
            "Zone 3: High Power + Low Risk": "green",
            "Zone 4: Medium Power + High Risk": "coral",
            "Zone 5: Medium Power + Low Risk": "lightgreen",
            "Zone 6: Low Power + Low Risk": "lightblue",
            "Zone 7: Other/Unknown": "gray"
        }

        # Create map centered on data
        center_lat = df_clean["Latitude"].mean()
        center_lon = df_clean["Longitude"].mean()
        
        m = folium.Map(location=[center_lat, center_lon], zoom_start=10)
        
        # Add legend
        legend_html = '''
        <div style="position: fixed; 
                    bottom: 50px; left: 50px; width: 300px; height: 200px; 
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:14px; padding: 10px">
        <p><b>Service Zones Legend</b></p>
        '''
        for zone, color in zone_colors.items():
            legend_html += f'<p><i class="fa fa-circle" style="color:{color}"></i> {zone}</p>'
        legend_html += '</div>'
        m.get_root().html.add_child(folium.Element(legend_html))
        
        success_count = 0
        zone_counts = {}
        
        for _, row in df_clean.iterrows():
            try:
                zone = row["Service_Zone"]
                color = zone_colors.get(zone, "gray")
                
                # Count zones
                zone_counts[zone] = zone_counts.get(zone, 0) + 1
                
                folium.CircleMarker(
                    location=[row["Latitude"], row["Longitude"]],
                    radius=4,
                    color=color,
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.7,
                    popup=folium.Popup(
                        f"{row['Neighborhood']} ({row['District']})<br>"
                        f"{zone}<br>"
                        f"Purchasing Power: {row['Purchasing_Power']}<br>"
                        f"Earthquake Risk: {row['Earthquake_Risk']}",
                        max_width=300
                    )
                ).add_to(m)
                success_count += 1
            except Exception as e:
                print(f"⚠️ Warning: Failed to add marker for {row.get('Neighborhood', 'Unknown')}: {e}")
                continue

        m.save(output_html)
        print(f"✅ Service zones map saved as {output_html}")
        print(f"📍 Successfully mapped {success_count}/{len(df_clean)} neighborhoods")
        
        # Print zone statistics
        print("\n📊 Service Zone Distribution:")
        for zone, count in sorted(zone_counts.items()):
            percentage = (count / success_count) * 100
            print(f"  {zone}: {count} neighborhoods ({percentage:.1f}%)")
        
        return df_clean, zone_counts
        
    except Exception as e:
        print(f"❌ Error in service zone mapping: {e}")
        raise
