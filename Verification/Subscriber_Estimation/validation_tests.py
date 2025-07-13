#!/usr/bin/env python3
"""
Comprehensive Validation Tests for Starlink Subscriber Estimation Model
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, LeaveOneOut
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class SubscriberEstimationValidator:
    def __init__(self):
        self.df = None
        self.turkey_data = None
        self.df_model = None
        self.features = None
        self.enhanced_features = None
        self.target = 'Estimated_Subscribers'
        self.subscriber_data = {
            'AFR': 401 * 1000,
            'ASI': 970 * 1000,
            'NAm': 2423 * 1000,
            'EUR': 620 * 1000,
            'OCE': 390 * 1000,
            'SAm': 748 * 1000
        }
        self.load_and_prepare_data()
        
    def load_and_prepare_data(self):
        """Load and prepare data exactly as in main.py"""
        print("📊 Loading and preparing subscriber estimation data...")
        
        # Load data (exactly as in main.py)
        self.df = pd.read_csv('data.csv', sep=';')
        
        # Convert European decimal format and handle missing values
        numeric_columns = [
            'Population (in millions)',
            'GDP per capita, PPP (current international $) [2023]',
            'Individuals using the Internet (% of population) [2023]',
            'InternetCost_BroadbandCostPerMonth_USD_2024',
            'Starlink Price'
        ]
        
        for col in numeric_columns:
            if col in self.df.columns:
                self.df[col] = self.df[col].astype(str).str.replace(',', '.')
                self.df[col] = self.df[col].replace(['#N/A', 'no data', ''], np.nan)
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        
        # Rename columns
        self.df.rename(columns={
            'Population (in millions)': 'Population_millions',
            'GDP per capita, PPP (current international $) [2023]': 'GDP_per_capita',
            'Individuals using the Internet (% of population) [2023]': 'Internet_penetration',
            'InternetCost_BroadbandCostPerMonth_USD_2024': 'Broadband_cost_USD',
            'Starlink Price': 'Starlink_price_USD'
        }, inplace=True)
        
        # Country to continent mapping (as in main.py)
        country_to_continent = {
            'BEN': 'AFR', 'BWA': 'AFR', 'GHA': 'AFR', 'KEN': 'AFR', 'MDG': 'AFR', 
            'MOZ': 'AFR', 'NGA': 'AFR', 'RWA': 'AFR', 'SLE': 'AFR', 'SWZ': 'AFR',
            'TLS': 'AFR', 'ZMB': 'AFR', 'ZWE': 'AFR',
            'IDN': 'ASI', 'JPN': 'ASI', 'MDV': 'ASI', 'MYS': 'ASI', 'PHL': 'ASI', 'MNG': 'ASI',
            'BRB': 'NAm', 'CAN': 'NAm', 'CRI': 'NAm', 'DOM': 'NAm', 'HND': 'NAm', 
            'JAM': 'NAm', 'MEX': 'NAm', 'PAN': 'NAm', 'TTO': 'NAm', 'USA': 'NAm',
            'ALB': 'EUR', 'AUT': 'EUR', 'BEL': 'EUR', 'BGR': 'EUR', 'CHE': 'EUR', 
            'CYP': 'EUR', 'DEU': 'EUR', 'DNK': 'EUR', 'ESP': 'EUR', 'EST': 'EUR',
            'FIN': 'EUR', 'FRA': 'EUR', 'GBR': 'EUR', 'GEO': 'EUR', 'GRC': 'EUR',
            'HRV': 'EUR', 'HUN': 'EUR', 'IRL': 'EUR', 'ITA': 'EUR', 'LTU': 'EUR',
            'LUX': 'EUR', 'LVA': 'EUR', 'MDA': 'EUR', 'MKD': 'EUR', 'MLT': 'EUR',
            'NLD': 'EUR', 'NOR': 'EUR', 'POL': 'EUR', 'PRT': 'EUR', 'ROU': 'EUR',
            'SVN': 'EUR', 'SWE': 'EUR', 'TUR': 'EUR', 'UKR': 'EUR',
            'AUS': 'OCE', 'NZL': 'OCE', 'SLB': 'OCE', 'VUT': 'OCE', 'WSM': 'OCE',
            'ARG': 'SAm', 'BRA': 'SAm', 'CHL': 'SAm', 'COL': 'SAm', 'ECU': 'SAm',
            'PER': 'SAm', 'PRY': 'SAm', 'SLV': 'SAm', 'URY': 'SAm'
        }
        
        self.df['Country Code'] = self.df['Country Code'].map(country_to_continent).fillna(self.df['Country Code'])
        
        # Separate Turkey and Starlink-available countries
        self.turkey_data = self.df[self.df['Country Name'] == 'Turkiye'].copy()
        df_starlink_available = self.df[self.df['Starlink Available'] == 'yes'].copy()
        
        # Fill Turkey's missing broadband cost
        self.turkey_data['Broadband_cost_USD'] = self.turkey_data['Broadband_cost_USD'].fillna(12.35)
        
        # Calculate continental populations and apportion subscribers
        continental_population = df_starlink_available.groupby('Country Code')['Population_millions'].sum().to_dict()
        
        def apportion_subscribers(row):
            continent_code = row['Country Code']
            country_population = row['Population_millions']
            total_cont_pop = continental_population.get(continent_code, 0)
            total_cont_subs = self.subscriber_data.get(continent_code, 0)
            
            if total_cont_pop == 0:
                return 0
            
            population_share = country_population / total_cont_pop
            estimated_subs = total_cont_subs * population_share
            return estimated_subs
        
        df_starlink_available = df_starlink_available.dropna(subset=['Population_millions'])
        df_starlink_available['Estimated_Subscribers'] = df_starlink_available.apply(apportion_subscribers, axis=1)
        
        # Define features
        self.features = [
            'Population_millions',
            'GDP_per_capita',
            'Internet_penetration',
            'Broadband_cost_USD',
            'Starlink_price_USD'
        ]
        
        # Prepare model data
        self.df_model = df_starlink_available.dropna(subset=self.features + [self.target])
        
        # Create enhanced features
        self.df_model['GDP_per_Internet_user'] = self.df_model['GDP_per_capita'] / (self.df_model['Internet_penetration'] / 100)
        self.df_model['Affordability_Index'] = self.df_model['GDP_per_capita'] / self.df_model['Broadband_cost_USD']
        self.df_model['Market_Size'] = self.df_model['Population_millions'] * (self.df_model['Internet_penetration'] / 100)
        
        self.enhanced_features = self.features + ['GDP_per_Internet_user', 'Affordability_Index', 'Market_Size']
        
        print(f"✅ Data prepared: {len(self.df_model)} countries for modeling")
        print(f"📊 Continental subscriber totals: {sum(self.subscriber_data.values()):,}")
        
    def test_data_quality(self):
        """Test data quality and subscriber apportionment logic"""
        print("\n" + "="*60)
        print("🔍 DATA QUALITY & APPORTIONMENT VALIDATION")
        print("="*60)
        
        # Check subscriber data integrity
        print("1. 📊 Subscriber Data Validation:")
        total_subscribers = sum(self.subscriber_data.values())
        print(f"   Total global subscribers: {total_subscribers:,}")
        
        # Verify apportionment adds up correctly
        modeled_total = self.df_model['Estimated_Subscribers'].sum()
        print(f"   Total apportioned subscribers: {modeled_total:,.0f}")
        print(f"   Difference: {abs(total_subscribers - modeled_total):,.0f} ({abs(total_subscribers - modeled_total)/total_subscribers*100:.1f}%)")
        
        if abs(total_subscribers - modeled_total) / total_subscribers < 0.05:
            print("   ✅ Subscriber apportionment is accurate")
        else:
            print("   ⚠️ Significant discrepancy in subscriber apportionment")
        
        # Check continental distribution
        print("\n2. 🌍 Continental Distribution:")
        continental_subs = self.df_model.groupby('Country Code')['Estimated_Subscribers'].sum()
        for continent, expected in self.subscriber_data.items():
            actual = continental_subs.get(continent, 0)
            print(f"   {continent}: Expected {expected:,}, Actual {actual:,.0f} ({actual/expected*100:.1f}%)")
        
        # Data quality checks
        print("\n3. 📈 Data Quality Checks:")
        missing_data = self.df_model[self.features].isnull().sum()
        print(f"   Missing values in features: {missing_data.sum()} total")
        
        # Check for outliers
        print("\n4. 🔍 Outlier Analysis:")
        outlier_threshold = self.df_model[self.target].quantile(0.95)
        outliers = self.df_model[self.df_model[self.target] > outlier_threshold]
        print(f"   Outlier threshold (95th percentile): {outlier_threshold:,.0f}")
        print(f"   Number of outliers: {len(outliers)}")
        
        for _, row in outliers.iterrows():
            print(f"     {row['Country Name']}: {row[self.target]:,.0f} subscribers")
        
        # Check for extreme values
        print("\n5. ⚠️ Extreme Value Detection:")
        extreme_checks = [
            (self.df_model['Population_millions'] > 300, "Population > 300M"),
            (self.df_model['GDP_per_capita'] > 100000, "GDP > $100k"),
            (self.df_model['Internet_penetration'] > 99, "Internet > 99%"),
            (self.df_model['Broadband_cost_USD'] > 200, "Broadband > $200"),
            (self.df_model['Starlink_price_USD'] > 150, "Starlink > $150")
        ]
        
        for condition, description in extreme_checks:
            count = condition.sum()
            if count > 0:
                countries = list(self.df_model[condition]['Country Name'])
                print(f"   {description}: {count} countries - {countries}")
            else:
                print(f"   {description}: ✅ No extreme values")
    
    def test_model_assumptions(self):
        """Test modeling assumptions and approach validity"""
        print("\n" + "="*60)
        print("🧮 MODELING ASSUMPTIONS VALIDATION")
        print("="*60)
        
        # 1. Population-based apportionment assumption
        print("1. 📊 Population-Based Apportionment Validity:")
        
        # Test if subscriber density varies significantly by country characteristics
        self.df_model['Subscriber_Density'] = self.df_model['Estimated_Subscribers'] / self.df_model['Population_millions']
        
        # Check correlation with economic factors
        density_gdp_corr = np.corrcoef(self.df_model['Subscriber_Density'], self.df_model['GDP_per_capita'])[0,1]
        density_internet_corr = np.corrcoef(self.df_model['Subscriber_Density'], self.df_model['Internet_penetration'])[0,1]
        
        print(f"   Subscriber density vs GDP correlation: {density_gdp_corr:.3f}")
        print(f"   Subscriber density vs Internet penetration correlation: {density_internet_corr:.3f}")
        
        if abs(density_gdp_corr) > 0.3 or abs(density_internet_corr) > 0.3:
            print("   ⚠️ Population-based apportionment may be oversimplified")
            print("   💡 Consider economic factors in apportionment")
        else:
            print("   ✅ Population-based apportionment seems reasonable")
        
        # 2. Feature correlation analysis
        print("\n2. 🔗 Feature Correlation Analysis:")
        corr_matrix = self.df_model[self.features].corr()
        
        # Find high correlations
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = abs(corr_matrix.iloc[i, j])
                if corr_val > 0.7:
                    high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_val))
        
        if high_corr_pairs:
            print("   ⚠️ High correlations detected:")
            for feat1, feat2, corr_val in high_corr_pairs:
                print(f"     {feat1} ↔ {feat2}: {corr_val:.3f}")
        else:
            print("   ✅ No severe multicollinearity detected")
        
        # 3. Target variable distribution
        print("\n3. 📈 Target Variable Analysis:")
        skewness = stats.skew(self.df_model[self.target])
        kurtosis = stats.kurtosis(self.df_model[self.target])
        
        print(f"   Skewness: {skewness:.3f}")
        print(f"   Kurtosis: {kurtosis:.3f}")
        
        if abs(skewness) > 2:
            print("   ⚠️ Highly skewed target variable - consider log transformation")
        else:
            print("   ✅ Target variable distribution is reasonable")
        
        # 4. Enhanced features validity
        print("\n4. 🔧 Enhanced Features Validation:")
        
        # Check for infinite values
        inf_counts = {}
        for feature in self.enhanced_features:
            inf_count = np.isinf(self.df_model[feature]).sum()
            inf_counts[feature] = inf_count
            if inf_count > 0:
                print(f"   ⚠️ {feature}: {inf_count} infinite values")
        
        if sum(inf_counts.values()) == 0:
            print("   ✅ No infinite values in enhanced features")
        
        # Check feature ranges
        print("\n   Enhanced feature ranges:")
        for feature in ['GDP_per_Internet_user', 'Affordability_Index', 'Market_Size']:
            if feature in self.df_model.columns:
                min_val = self.df_model[feature].min()
                max_val = self.df_model[feature].max()
                print(f"     {feature}: {min_val:.1f} - {max_val:.1f}")
    
    def test_model_performance(self):
        """Test model performance and detect overfitting"""
        print("\n" + "="*60)
        print("🎯 MODEL PERFORMANCE & OVERFITTING DETECTION")
        print("="*60)
        
        # Prepare clean data
        X = self.df_model[self.enhanced_features].replace([np.inf, -np.inf], np.nan).dropna()
        y = self.df_model[self.target].loc[X.index]
        
        print(f"Clean dataset size: {len(X)} countries")
        
        # Test different models
        models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Pipeline([('scaler', StandardScaler()), ('ridge', Ridge(alpha=1.0))]),
            'ElasticNet': Pipeline([('scaler', StandardScaler()), ('elastic', ElasticNet(alpha=1.0))]),
            'Random Forest': RandomForestRegressor(n_estimators=50, random_state=42, max_depth=5)
        }
        
        print("\n1. 📊 Cross-Validation Performance:")
        print(f"{'Model':<20} {'CV R²':<10} {'CV Std':<10} {'Train R²':<10} {'Overfitting':<12}")
        print("-" * 70)
        
        results = {}
        for name, model in models.items():
            # Cross-validation scores
            cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
            
            # Training score
            model.fit(X, y)
            train_score = model.score(X, y)
            
            # Detect overfitting
            overfitting = train_score - cv_scores.mean()
            overfitting_severity = "High" if overfitting > 0.2 else "Medium" if overfitting > 0.1 else "Low"
            
            results[name] = {
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'train_score': train_score,
                'overfitting': overfitting
            }
            
            print(f"{name:<20} {cv_scores.mean():.4f} {cv_scores.std():.4f} {train_score:.4f} {overfitting_severity:<12}")
        
        # 2. Leave-One-Out Cross-Validation for small dataset
        print("\n2. 🔄 Leave-One-Out Cross-Validation:")
        if len(X) <= 30:  # Only for small datasets
            loo_model = Pipeline([('scaler', StandardScaler()), ('ridge', Ridge(alpha=1.0))])
            loo_scores = cross_val_score(loo_model, X, y, cv=LeaveOneOut(), scoring='r2')
            print(f"   LOO R² mean: {loo_scores.mean():.4f}")
            print(f"   LOO R² std: {loo_scores.std():.4f}")
        else:
            print("   Skipping LOO CV (dataset too large)")
        
        # 3. Prediction accuracy on different data splits
        print("\n3. 🎯 Train/Test Split Analysis:")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        best_model = Pipeline([('scaler', StandardScaler()), ('ridge', Ridge(alpha=1.0))])
        best_model.fit(X_train, y_train)
        
        train_pred = best_model.predict(X_train)
        test_pred = best_model.predict(X_test)
        
        train_mae = mean_absolute_error(y_train, train_pred)
        test_mae = mean_absolute_error(y_test, test_pred)
        train_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(y_test, test_pred)
        
        print(f"   Training R²: {train_r2:.4f}, MAE: {train_mae:,.0f}")
        print(f"   Testing R²: {test_r2:.4f}, MAE: {test_mae:,.0f}")
        print(f"   Performance degradation: {train_r2 - test_r2:.4f}")
        
        if train_r2 - test_r2 > 0.3:
            print("   ⚠️ Significant overfitting detected")
        else:
            print("   ✅ Model generalizes reasonably well")
        
        # 4. Check for perfect predictions (suspicious)
        print("\n4. 🚨 Suspicious Performance Detection:")
        for name, result in results.items():
            if result['train_score'] > 0.99:
                print(f"   ⚠️ {name}: Suspiciously high training score ({result['train_score']:.4f})")
            if result['overfitting'] > 0.5:
                print(f"   ⚠️ {name}: Severe overfitting detected ({result['overfitting']:.4f})")
        
        return results
    
    def test_turkey_prediction_validity(self):
        """Validate Turkey prediction against comparable countries and business logic"""
        print("\n" + "="*60)
        print("🇹🇷 TURKEY PREDICTION VALIDATION")
        print("="*60)
        
        # Prepare Turkey data
        turkey_enhanced = self.turkey_data.copy()
        turkey_enhanced['GDP_per_Internet_user'] = turkey_enhanced['GDP_per_capita'] / (turkey_enhanced['Internet_penetration'] / 100)
        turkey_enhanced['Affordability_Index'] = turkey_enhanced['GDP_per_capita'] / turkey_enhanced['Broadband_cost_USD']
        turkey_enhanced['Market_Size'] = turkey_enhanced['Population_millions'] * (turkey_enhanced['Internet_penetration'] / 100)
        
        print("Turkey's Profile:")
        print(f"   Population: {turkey_enhanced['Population_millions'].iloc[0]:.1f} million")
        print(f"   GDP per capita: ${turkey_enhanced['GDP_per_capita'].iloc[0]:,.0f}")
        print(f"   Internet penetration: {turkey_enhanced['Internet_penetration'].iloc[0]:.1f}%")
        print(f"   Broadband cost: ${turkey_enhanced['Broadband_cost_USD'].iloc[0]:.2f}")
        
        # Make predictions with different models
        X = self.df_model[self.enhanced_features].replace([np.inf, -np.inf], np.nan).dropna()
        y = self.df_model[self.target].loc[X.index]
        
        models = {
            'Ridge': Pipeline([('scaler', StandardScaler()), ('ridge', Ridge(alpha=1.0))]),
            'ElasticNet': Pipeline([('scaler', StandardScaler()), ('elastic', ElasticNet(alpha=1.0))]),
            'Random Forest': RandomForestRegressor(n_estimators=50, random_state=42, max_depth=5)
        }
        
        print("\n🎯 Turkey Predictions by Model:")
        turkey_predictions = {}
        for name, model in models.items():
            model.fit(X, y)
            turkey_X = turkey_enhanced[self.enhanced_features].replace([np.inf, -np.inf], np.nan)
            prediction = model.predict(turkey_X)[0]
            turkey_predictions[name] = prediction
            print(f"   {name}: {prediction:,.0f} subscribers")
        
        # Calculate prediction statistics
        pred_values = list(turkey_predictions.values())
        mean_pred = np.mean(pred_values)
        std_pred = np.std(pred_values)
        
        print(f"\n📊 Prediction Statistics:")
        print(f"   Mean prediction: {mean_pred:,.0f}")
        print(f"   Standard deviation: {std_pred:,.0f}")
        print(f"   Coefficient of variation: {std_pred/mean_pred*100:.1f}%")
        
        if std_pred/mean_pred > 0.3:
            print("   ⚠️ High prediction uncertainty - models disagree significantly")
        else:
            print("   ✅ Reasonable prediction consistency across models")
        
        # Compare with similar countries
        print("\n🔍 Comparison with Similar Countries:")
        
        # Find countries with similar characteristics
        turkey_pop = turkey_enhanced['Population_millions'].iloc[0]
        turkey_gdp = turkey_enhanced['GDP_per_capita'].iloc[0]
        turkey_internet = turkey_enhanced['Internet_penetration'].iloc[0]
        
        similar_countries = self.df_model[
            (abs(self.df_model['Population_millions'] - turkey_pop) < 50) |
            (abs(self.df_model['GDP_per_capita'] - turkey_gdp) < 15000) |
            (abs(self.df_model['Internet_penetration'] - turkey_internet) < 10)
        ].copy()
        
        if len(similar_countries) > 0:
            similar_countries['Similarity_Score'] = (
                1 / (1 + abs(similar_countries['Population_millions'] - turkey_pop) / turkey_pop) +
                1 / (1 + abs(similar_countries['GDP_per_capita'] - turkey_gdp) / turkey_gdp) +
                1 / (1 + abs(similar_countries['Internet_penetration'] - turkey_internet) / turkey_internet)
            )
            
            top_similar = similar_countries.nlargest(5, 'Similarity_Score')
            
            print("   Most similar countries:")
            for _, row in top_similar.iterrows():
                penetration_rate = (row['Estimated_Subscribers'] / row['Population_millions'] / 1000000) * 100
                print(f"     {row['Country Name']}: {row['Estimated_Subscribers']:,.0f} subs ({penetration_rate:.3f}% penetration)")
            
            # Calculate Turkey's penetration rate based on predictions
            turkey_penetration_rates = {}
            for model_name, prediction in turkey_predictions.items():
                penetration = (prediction / turkey_pop / 1000000) * 100
                turkey_penetration_rates[model_name] = penetration
            
            avg_turkey_penetration = np.mean(list(turkey_penetration_rates.values()))
            avg_similar_penetration = np.mean(
                [(row['Estimated_Subscribers'] / row['Population_millions'] / 1000000) * 100 
                 for _, row in top_similar.iterrows()]
            )
            
            print(f"\n📈 Penetration Rate Analysis:")
            print(f"   Turkey predicted penetration: {avg_turkey_penetration:.3f}%")
            print(f"   Similar countries average: {avg_similar_penetration:.3f}%")
            print(f"   Relative penetration: {avg_turkey_penetration/avg_similar_penetration*100:.0f}%")
        
        # Business logic validation
        print("\n✅ Business Logic Validation:")
        
        # Check if prediction is reasonable as percentage of population
        for model_name, prediction in turkey_predictions.items():
            penetration = (prediction / turkey_pop / 1000000) * 100
            if penetration > 1.0:
                print(f"   ⚠️ {model_name}: High penetration rate ({penetration:.3f}%)")
            elif penetration < 0.01:
                print(f"   ⚠️ {model_name}: Very low penetration rate ({penetration:.3f}%)")
            else:
                print(f"   ✅ {model_name}: Reasonable penetration rate ({penetration:.3f}%)")
        
        # Check against European average
        european_countries = self.df_model[self.df_model['Country Code'] == 'EUR']
        if len(european_countries) > 0:
            european_avg_subs = european_countries['Estimated_Subscribers'].mean()
            european_avg_penetration = np.mean(
                [(row['Estimated_Subscribers'] / row['Population_millions'] / 1000000) * 100 
                 for _, row in european_countries.iterrows()]
            )
            
            print(f"\n🇪🇺 European Context:")
            print(f"   Average European subscribers: {european_avg_subs:,.0f}")
            print(f"   Average European penetration: {european_avg_penetration:.3f}%")
            print(f"   Turkey vs European average: {mean_pred/european_avg_subs*100:.0f}%")
        
        return turkey_predictions
    
    def test_sensitivity_analysis(self):
        """Test prediction sensitivity to input parameters"""
        print("\n" + "="*60)
        print("🎛️ SENSITIVITY ANALYSIS")
        print("="*60)
        
        # Prepare model and Turkey baseline
        X = self.df_model[self.enhanced_features].replace([np.inf, -np.inf], np.nan).dropna()
        y = self.df_model[self.target].loc[X.index]
        
        model = Pipeline([('scaler', StandardScaler()), ('ridge', Ridge(alpha=1.0))])
        model.fit(X, y)
        
        # Baseline Turkey prediction
        turkey_baseline = self.turkey_data.copy()
        turkey_baseline['GDP_per_Internet_user'] = turkey_baseline['GDP_per_capita'] / (turkey_baseline['Internet_penetration'] / 100)
        turkey_baseline['Affordability_Index'] = turkey_baseline['GDP_per_capita'] / turkey_baseline['Broadband_cost_USD']
        turkey_baseline['Market_Size'] = turkey_baseline['Population_millions'] * (turkey_baseline['Internet_penetration'] / 100)
        
        baseline_prediction = model.predict(turkey_baseline[self.enhanced_features])[0]
        
        print(f"Baseline Turkey prediction: {baseline_prediction:,.0f} subscribers")
        
        # Test sensitivity to each parameter
        sensitivity_tests = [
            ('Population_millions', [0.8, 0.9, 1.0, 1.1, 1.2]),
            ('GDP_per_capita', [0.7, 0.85, 1.0, 1.15, 1.3]),
            ('Internet_penetration', [0.8, 0.9, 1.0, 1.1, 1.2]),
            ('Broadband_cost_USD', [0.5, 0.75, 1.0, 1.5, 2.0]),
        ]
        
        print("\n📊 Parameter Sensitivity:")
        for param, multipliers in sensitivity_tests:
            print(f"\n{param}:")
            for mult in multipliers:
                turkey_test = turkey_baseline.copy()
                turkey_test[param] = turkey_test[param] * mult
                
                # Recalculate enhanced features
                turkey_test['GDP_per_Internet_user'] = turkey_test['GDP_per_capita'] / (turkey_test['Internet_penetration'] / 100)
                turkey_test['Affordability_Index'] = turkey_test['GDP_per_capita'] / turkey_test['Broadband_cost_USD']
                turkey_test['Market_Size'] = turkey_test['Population_millions'] * (turkey_test['Internet_penetration'] / 100)
                
                prediction = model.predict(turkey_test[self.enhanced_features])[0]
                change_pct = (prediction - baseline_prediction) / baseline_prediction * 100
                
                print(f"   {mult:+.0%}: {prediction:,.0f} ({change_pct:+.1f}%)")
    
    def run_all_tests(self):
        """Run all validation tests"""
        print("🚀 STARTING COMPREHENSIVE SUBSCRIBER ESTIMATION VALIDATION")
        print("="*60)
        
        self.test_data_quality()
        self.test_model_assumptions()
        model_results = self.test_model_performance()
        turkey_predictions = self.test_turkey_prediction_validity()
        self.test_sensitivity_analysis()
        
        print("\n" + "="*60)
        print("✅ ALL VALIDATION TESTS COMPLETED")
        print("="*60)
        
        # Summary recommendations
        print("\n📋 SUMMARY RECOMMENDATIONS:")
        
        # Data quality assessment
        total_subscribers = sum(self.subscriber_data.values())
        modeled_total = self.df_model['Estimated_Subscribers'].sum()
        apportionment_error = abs(total_subscribers - modeled_total) / total_subscribers
        
        if apportionment_error < 0.05:
            print("✅ Data apportionment is accurate")
        else:
            print("⚠️ Review subscriber apportionment methodology")
        
        # Model performance assessment
        best_cv_score = max(model_results.values(), key=lambda x: x['cv_mean'])['cv_mean']
        if best_cv_score > 0.6:
            print("✅ Model has good predictive power")
        else:
            print("⚠️ Model has limited predictive power - consider additional features")
        
        # Overfitting assessment
        max_overfitting = max(model_results.values(), key=lambda x: x['overfitting'])['overfitting']
        if max_overfitting > 0.3:
            print("⚠️ Significant overfitting detected - use simpler models")
        else:
            print("✅ Overfitting is under control")
        
        # Turkey prediction assessment
        pred_values = list(turkey_predictions.values())
        prediction_uncertainty = np.std(pred_values) / np.mean(pred_values)
        
        if prediction_uncertainty > 0.3:
            print("⚠️ High prediction uncertainty - use ensemble methods")
        else:
            print("✅ Turkey predictions are reasonably consistent")
        
        # Final recommendation
        mean_prediction = np.mean(pred_values)
        print(f"\n🇹🇷 Turkey Subscriber Estimate: {mean_prediction:,.0f}")
        print("   Use this as baseline estimate with appropriate confidence intervals")
        print("   Consider market research to validate assumptions")

if __name__ == "__main__":
    validator = SubscriberEstimationValidator()
    validator.run_all_tests() 