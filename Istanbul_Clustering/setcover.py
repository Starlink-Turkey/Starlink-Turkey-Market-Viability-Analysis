# starlink_istanbul/setcover.py

import pandas as pd
import numpy as np
from geopy.distance import geodesic
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpBinary, PULP_CBC_CMD, LpStatus


def solve_set_cover(
    neighborhoods_excel: str,
    service_points_excel: str,
    radius_km: float = 10.0,
    population_coverage: float = 0.95
):
    """
    Load neighborhood demand points (with Population, Latitude, Longitude) and candidate
    service points. Build a coverage matrix for any demand i if within radius_km of service j.
    Then solve a binary set-cover LP that ensures ≥population_coverage of total population is covered.
    Print chosen sites and a greedy fallback.
    
    Fixed issues:
    - Added proper constraint validation
    - Fixed population coverage logic
    - Added infeasibility checks
    - Improved coverage matrix validation
    """
    print(f"🔍 Loading data and solving set cover with {radius_km}km radius, {population_coverage*100}% coverage...")
    
    # Load and validate data
    try:
        demand_df = pd.read_excel(neighborhoods_excel)
        service_df = pd.read_excel(service_points_excel)
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # Clean data
    demand_df = demand_df.dropna(subset=["Latitude", "Longitude", "Population"])
    service_df = service_df.dropna(subset=["Latitude", "Longitude"])
    
    print(f"📊 Loaded {len(demand_df)} demand points and {len(service_df)} service points")
    
    # Validate population data
    if demand_df["Population"].sum() == 0:
        print("❌ Error: Total population is zero")
        return
        
    # Build coverage matrix with validation
    coverage = {}
    total_coverage_stats = {}
    
    for i, d in demand_df.iterrows():
        coverage[i] = []
        for j, s in service_df.iterrows():
            try:
                dist = geodesic(
                    (d["Latitude"], d["Longitude"]),
                    (s["Latitude"], s["Longitude"])
                ).km
                if dist <= radius_km:
                    coverage[i].append(j)
            except Exception as e:
                print(f"⚠️ Warning: Distance calculation failed for demand {i}, service {j}: {e}")
                continue
    
    # Calculate coverage statistics for each service point
    for j in service_df.index:
        covered_pop = sum(
            demand_df.loc[i, "Population"]
            for i in coverage
            if j in coverage[i]
        )
        total_coverage_stats[j] = covered_pop
    
    # Check if problem is feasible
    max_possible_coverage = 0
    covered_demands = set()
    
    # Calculate maximum possible coverage using all service points
    for i in coverage:
        if len(coverage[i]) > 0:  # If demand point i can be covered by at least one service point
            covered_demands.add(i)
            max_possible_coverage += demand_df.loc[i, "Population"]
    
    total_pop = demand_df["Population"].sum()
    max_coverage_ratio = max_possible_coverage / total_pop
    
    print(f"📈 Maximum possible coverage: {max_coverage_ratio:.2%} of population")
    print(f"🎯 Required coverage: {population_coverage:.2%} of population")
    
    if max_coverage_ratio < population_coverage:
        print(f"❌ Problem is infeasible! Maximum possible coverage ({max_coverage_ratio:.2%}) < required ({population_coverage:.2%})")
        print("💡 Suggestions:")
        print(f"   - Increase radius from {radius_km}km")
        print(f"   - Reduce required coverage from {population_coverage:.2%}")
        print(f"   - Add more service points")
        return
    
    # Create and solve the optimization model
    model = LpProblem("Starlink_Set_Covering", LpMinimize)
    
    # Decision variables
    x_vars = {j: LpVariable(f"x_{j}", cat=LpBinary) for j in service_df.index}
    
    # Objective: minimize number of sites
    model += lpSum(x_vars[j] for j in x_vars)
    
    # Coverage constraints: each demand point can be covered by at least one selected service point
    # We use a different formulation: for each demand point, the sum of covering service points >= 1
    # But we only enforce this for demand points we want to cover
    
    # First, let's identify which demand points we must cover to meet population target
    # We'll use a more flexible approach: cover enough population, not necessarily all points
    
    # Add constraint: total covered population >= target
    covered_population = 0
    for i in demand_df.index:
        if i in coverage and len(coverage[i]) > 0:
            # This demand point can be covered
            # Add constraint: if any service covering this point is selected, count this population
            # Use indicator constraints: if any x_j for j in coverage[i] is 1, then this population counts
            
            # Create indicator variable for whether demand i is covered
            z_i = LpVariable(f"z_{i}", cat=LpBinary)
            
            # Constraint: z_i <= sum of all service points that can cover i
            model += z_i <= lpSum(x_vars[j] for j in coverage[i])
            
            # Add to covered population sum
            covered_population += demand_df.loc[i, "Population"] * z_i
    
    # Population coverage constraint
    model += covered_population >= population_coverage * total_pop
    
    # Solve the model
    print("🔧 Solving optimization model...")
    solver = PULP_CBC_CMD(msg=False)
    status = model.solve(solver)
    
    # Check solution status
    if LpStatus[status] != 'Optimal':
        print(f"❌ Optimization failed with status: {LpStatus[status]}")
        if LpStatus[status] == 'Infeasible':
            print("💡 The problem is infeasible with current parameters")
        return
    
    # Extract solution
    chosen = [j for j in x_vars if x_vars[j].varValue and x_vars[j].varValue > 0.5]
    
    if not chosen:
        print("❌ No sites selected in optimal solution")
        return
    
    print(f"✅ Optimal solution found: {len(chosen)} sites selected")
    print(f"🏢 Optimal sites: {chosen}")
    
    # Calculate actual coverage achieved
    actual_covered_pop = 0
    covered_demand_points = set()
    
    for i in coverage:
        if any(j in chosen for j in coverage[i]):
            actual_covered_pop += demand_df.loc[i, "Population"]
            covered_demand_points.add(i)
    
    actual_coverage_ratio = actual_covered_pop / total_pop
    print(f"📊 Actual coverage achieved: {actual_coverage_ratio:.2%} of population")
    print(f"📍 Covered {len(covered_demand_points)} out of {len(demand_df)} demand points")
    
    print("\n📋 Coordinates of chosen sites:")
    chosen_sites = service_df.loc[chosen, ["Latitude", "Longitude"]].copy()
    chosen_sites["Coverage_Population"] = [total_coverage_stats[j] for j in chosen]
    print(chosen_sites)
    
    # Greedy fallback solution
    print(f"\n🛡️ Greedy fallback solution (top 5 by population covered):")
    site_coverage = []
    for j in service_df.index:
        covered_pop = total_coverage_stats.get(j, 0)
        site_coverage.append((j, covered_pop))
    
    greedy = [j for j, _ in sorted(site_coverage, key=lambda x: -x[1])[:5]]
    
    # Calculate actual unique coverage for greedy solution (avoid double counting)
    greedy_covered_pop = 0
    greedy_covered_points = set()
    
    for i in coverage:
        if any(j in greedy for j in coverage[i]):
            greedy_covered_pop += demand_df.loc[i, "Population"]
            greedy_covered_points.add(i)
    
    greedy_ratio = greedy_covered_pop / total_pop
    
    print(f"Greedy sites: {greedy}")
    print(f"Greedy coverage: {greedy_ratio:.2%} of population")
    print(f"Greedy covered points: {len(greedy_covered_points)} out of {len(demand_df)} demand points")
    
    return chosen, actual_coverage_ratio


def validate_coverage_parameters(radius_km: float, population_coverage: float):
    """
    Validate that the coverage parameters are reasonable
    """
    if radius_km <= 0:
        raise ValueError("Radius must be positive")
    if not 0 < population_coverage <= 1:
        raise ValueError("Population coverage must be between 0 and 1")
    if radius_km > 50:  # Reasonable upper bound for Starlink coverage
        print(f"⚠️ Warning: Radius {radius_km}km seems very large for satellite internet coverage")
    if population_coverage < 0.5:
        print(f"⚠️ Warning: Coverage target {population_coverage:.1%} seems low for a viable service")
