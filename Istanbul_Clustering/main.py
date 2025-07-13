# starlink_istanbul/src/main.py

from clustering import (
    plot_elbow_method,
    cluster_districts_and_map,
    cluster_neighborhoods_and_map
)
from geocoding import geocode_neighborhoods
from mapping import map_service_zones
from setcover import solve_set_cover, validate_coverage_parameters


def run_analysis_pipeline():
    """
    Run the complete Starlink Istanbul analysis pipeline with proper error handling
    """
    print("🚀 Starting Starlink Istanbul Analysis Pipeline")
    print("=" * 60)
    
    # File paths (adjust if needed)
    district_excel = "../data/istanbul_population.xlsx"
    neighborhood_input = "../data/istanbul_population.xlsx"   # sheet "Neighborhoods"
    neighborhood_geocoded = "../data/istanbul_neighborhoods_with_coords.xlsx"
    service_points_excel = "../data/istanbul_service_points.xlsx"

    try:
        # 1) Elbow method plot (optional)
        print("\n📊 Step 1: Generating Elbow Method Plot...")
        plot_elbow_method(excel_path=district_excel, skiprows=1, k_max=10)

        # 2) Cluster districts → Folium map
        print("\n🗺️ Step 2: Clustering Districts...")
        cluster_districts_and_map(excel_path=district_excel, skiprows=1, n_clusters=4)

        # 3) Geocode neighborhoods (only run once - commented out to avoid API calls)
        print("\n📍 Step 3: Geocoding (skipped - already done)")
        # geocode_neighborhoods(
        #     input_excel=neighborhood_input,
        #     output_excel=neighborhood_geocoded,
        #     sheet_name="Neighborhoods"
        # )

        # 4) Cluster neighborhoods → Folium map
        print("\n🏘️ Step 4: Clustering Neighborhoods...")
        cluster_neighborhoods_and_map(
            input_excel=neighborhood_geocoded,
            output_html="../data/istanbul_neighborhood_clusters.html",
            n_clusters=5
        )

        # 5) Map service zones → Folium map
        print("\n🎯 Step 5: Mapping Service Zones...")
        map_service_zones(
            input_excel=neighborhood_geocoded,
            output_html="../data/istanbul_service_zones_map.html"
        )

        # 6) Solve set-cover for service points with different parameters
        print("\n🔧 Step 6: Solving Set Cover Optimization...")
        
        # Test different scenarios as mentioned in verification
        scenarios = [
            {"radius": 0.49, "coverage": 0.95, "name": "Optimal Link Budget"},
            {"radius": 5.0, "coverage": 0.95, "name": "Conservative Coverage"},
            {"radius": 10.0, "coverage": 0.85, "name": "Reduced Requirements"},
            {"radius": 10.0, "coverage": 0.95, "name": "Standard Requirements"},
        ]
        
        for scenario in scenarios:
            print(f"\n--- Testing {scenario['name']} ---")
            try:
                validate_coverage_parameters(scenario["radius"], scenario["coverage"])
                result = solve_set_cover(
                    neighborhoods_excel=neighborhood_geocoded,
                    service_points_excel=service_points_excel,
                    radius_km=scenario["radius"],
                    population_coverage=scenario["coverage"]
                )
                if result:
                    chosen_sites, coverage_achieved = result
                    print(f"✅ Scenario successful: {len(chosen_sites)} sites, {coverage_achieved:.2%} coverage")
                else:
                    print(f"❌ Scenario failed")
            except Exception as e:
                print(f"❌ Error in scenario {scenario['name']}: {e}")
            print("-" * 40)
        
        print("\n🎉 Analysis Pipeline Complete!")
        
    except Exception as e:
        print(f"❌ Pipeline failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_analysis_pipeline()
