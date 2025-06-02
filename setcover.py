# starlink_istanbul/setcover.py

import pandas as pd
from geopy.distance import geodesic
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpBinary, PULP_CBC_CMD


def solve_set_cover(
    neighborhoods_excel: str,
    service_points_excel: str,
    radius_km: float = 10.0
):
    """
    Load neighborhood demand points (with Population, Latitude, Longitude) and candidate
    service points. Build a coverage matrix for any demand i if within radius_km of service j.
    Then solve a binary set-cover LP that ensures ≥95% of total population is covered.
    Print chosen sites and a greedy fallback.
    """
    demand_df = pd.read_excel(neighborhoods_excel)
    service_df = pd.read_excel(service_points_excel)

    demand_df = demand_df.dropna(subset=["Latitude", "Longitude"])
    coverage = {}

    # Build coverage dict: { demand_idx: [service_idx, ...], ... }
    for i, d in demand_df.iterrows():
        for j, s in service_df.iterrows():
            dist = geodesic(
                (d["Latitude"], d["Longitude"]),
                (s["Latitude"], s["Longitude"])
            ).km
            if dist <= radius_km:
                coverage.setdefault(i, []).append(j)

    model = LpProblem("Starlink_Set_Covering", LpMinimize)
    x_vars = {j: LpVariable(f"x_{j}", cat=LpBinary) for j in service_df.index}
    y_vars = {i: LpVariable(f"y_{i}", cat=LpBinary) for i in demand_df.index}

    # Objective: minimize number of sites
    model += lpSum(x_vars[j] for j in x_vars)

    # If a demand i is covered, then y_i ≤ sum(x_j for j in coverage[i])
    for i in y_vars:
        if i in coverage:
            model += y_vars[i] <= lpSum(x_vars[j] for j in coverage[i])
        else:
            model += y_vars[i] == 0

    total_pop = demand_df["Population"].sum()
    # Cover at least 95% of total population
    model += lpSum(demand_df.loc[i, "Population"] * y_vars[i] for i in y_vars) >= 0.95 * total_pop

    solver = PULP_CBC_CMD(msg=False)
    model.solve(solver)

    chosen = [j for j in x_vars if x_vars[j].varValue > 0.5]
    print(f"✅ Optimal sites: {chosen}")
    print("Coordinates of chosen sites:")
    print(service_df.loc[chosen, ["Latitude", "Longitude"]])

    # Greedy fallback (top 5 by population covered)
    site_coverage = []
    for j in service_df.index:
        covered_pop = sum(
            demand_df.loc[i, "Population"]
            for i in coverage
            if j in coverage[i]
        )
        site_coverage.append((j, covered_pop))

    greedy = [j for j, _ in sorted(site_coverage, key=lambda x: -x[1])[:5]]
    print(f"🛡️ Greedy fallback sites: {greedy}")
