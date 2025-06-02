# starlink_istanbul/src/main.py

from clustering import (
    plot_elbow_method,
    cluster_districts_and_map,
    cluster_neighborhoods_and_map
)
from geocoding import geocode_neighborhoods
from mapping import map_service_zones
from setcover import solve_set_cover


if __name__ == "__main__":
    # File paths (adjust if needed)
    district_excel = "../data/istanbul_population.xlsx"
    neighborhood_input = "../data/istanbul_population.xlsx"   # sheet “Neighborhoods”
    neighborhood_geocoded = "../data/istanbul_neighborhoods_with_coords.xlsx"
    service_points_excel = "../data/istanbul_service_points.xlsx"

    # 1) Elbow method plot (optional)
    plot_elbow_method(excel_path=district_excel, skiprows=1, k_max=10)

    # 2) Cluster districts → Folium map
    cluster_districts_and_map(excel_path=district_excel, skiprows=1, n_clusters=4)

    # 3) Geocode neighborhoods (only run once)
    # geocode_neighborhoods(
    #     input_excel=neighborhood_input,
    #     output_excel=neighborhood_geocoded,
    #     sheet_name="Neighborhoods"
    # )

    # 4) Cluster neighborhoods → Folium map
    cluster_neighborhoods_and_map(
        input_excel=neighborhood_geocoded,
        output_html="../data/istanbul_neighborhood_clusters.html",
        n_clusters=5
    )

    # 5) Map service zones → Folium map
    map_service_zones(
        input_excel=neighborhood_geocoded,
        output_html="../data/istanbul_service_zones_map.html"
    )

    # 6) Solve set-cover for service points
    solve_set_cover(
        neighborhoods_excel=neighborhood_geocoded,
        service_points_excel=service_points_excel,
        radius_km=10.0
    )
