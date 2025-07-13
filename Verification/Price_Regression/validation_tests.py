#!/usr/bin/env python3
"""
Comprehensive Validation Tests for Starlink Price Regression Model
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import cross_val_score, LeaveOneOut
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

class StarLinkPriceValidator:
    def __init__(self):
        self.df = None
        self.model = None
        self.X = None
        self.y = None
        self.load_and_prepare_data()
        
    def load_and_prepare_data(self):
        """Load and prepare the data exactly as in main.py"""
        print("📊 Loading and preparing data...")
        
        # Load data
        self.df = pd.read_csv('Data.csv', delimiter=';')
        
        # Data preprocessing (exactly as in main.py)
        self.df['GDP_PPP'] = self.df['GDP per capita, PPP (current international $) [2023]'] \
            .str.replace(',', '.').astype(float)
        self.df['Internet_Usage'] = self.df['Individuals using the Internet (% of population) [2023]'] \
            .str.replace(',', '.').astype(float) / 100
        self.df['BB_Price'] = self.df['InternetCost_BroadbandCostPerMonth_USD_2024'] \
            .str.replace(',', '.').astype(float)
        self.df['Starlink_Price'] = self.df['Starlink Price']
        self.df['Available'] = self.df['Starlink Available'].str.lower()
        
        # Filter and prepare regression data
        self.df = self.df[self.df['Available'] == 'yes'].dropna(subset=['GDP_PPP', 'Internet_Usage', 'BB_Price', 'Starlink_Price'])
        
        self.X = pd.DataFrame({
            'const': 1,
            'log_GDP': np.log(self.df['GDP_PPP']),
            'log_BBPrice': np.log(self.df['BB_Price']),
            'IntUsage': self.df['Internet_Usage']
        })
        self.y = np.log(self.df['Starlink_Price'])
        
        # Fit model
        self.model = sm.OLS(self.y, self.X).fit()
        
        print(f"✅ Data loaded: {len(self.df)} countries with Starlink availability")
        
    def test_data_quality(self):
        """Test data quality and integrity"""
        print("\n" + "="*60)
        print("🔍 DATA QUALITY VALIDATION")
        print("="*60)
        
        # Check for missing values
        missing_data = self.df[['GDP_PPP', 'Internet_Usage', 'BB_Price', 'Starlink_Price']].isnull().sum()
        print(f"Missing values per column:\n{missing_data}")
        
        # Check for outliers using IQR method
        print("\n📊 Outlier Detection (IQR Method):")
        for col in ['GDP_PPP', 'Internet_Usage', 'BB_Price', 'Starlink_Price']:
            q1 = self.df[col].quantile(0.25)
            q3 = self.df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers = self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)][col]
            print(f"  {col}: {len(outliers)} outliers detected")
            if len(outliers) > 0:
                print(f"    Outlier countries: {list(self.df[self.df[col].isin(outliers)]['Country Name'])}")
        
        # Data range validation
        print("\n📈 Data Range Validation:")
        print(f"  GDP PPP range: ${self.df['GDP_PPP'].min():,.0f} - ${self.df['GDP_PPP'].max():,.0f}")
        print(f"  Internet Usage range: {self.df['Internet_Usage'].min():.1%} - {self.df['Internet_Usage'].max():.1%}")
        print(f"  Broadband Price range: ${self.df['BB_Price'].min():.2f} - ${self.df['BB_Price'].max():.2f}")
        print(f"  Starlink Price range: ${self.df['Starlink_Price'].min():.0f} - ${self.df['Starlink_Price'].max():.0f}")
        
        # Check for extreme values
        extreme_checks = [
            (self.df['GDP_PPP'] < 1000, "GDP PPP < $1,000"),
            (self.df['GDP_PPP'] > 200000, "GDP PPP > $200,000"),
            (self.df['Internet_Usage'] < 0.1, "Internet Usage < 10%"),
            (self.df['BB_Price'] > 500, "Broadband Price > $500"),
            (self.df['Starlink_Price'] < 30, "Starlink Price < $30"),
            (self.df['Starlink_Price'] > 150, "Starlink Price > $150")
        ]
        
        print("\n⚠️  Extreme Value Checks:")
        for condition, description in extreme_checks:
            count = condition.sum()
            if count > 0:
                countries = list(self.df[condition]['Country Name'])
                print(f"  {description}: {count} countries - {countries}")
            else:
                print(f"  {description}: ✅ No extreme values")
    
    def test_model_assumptions(self):
        """Test regression model assumptions"""
        print("\n" + "="*60)
        print("🧮 MODEL ASSUMPTIONS VALIDATION")
        print("="*60)
        
        # Get residuals and fitted values
        residuals = self.model.resid
        fitted = self.model.fittedvalues
        
        # 1. Linearity Test
        print("1. 📈 Linearity Test:")
        # Check if log transformation improved linearity
        linear_corr = np.corrcoef(np.exp(self.y), np.exp(fitted))[0,1]
        print(f"   Correlation between actual and predicted prices: {linear_corr:.3f}")
        if linear_corr > 0.4:
            print("   ✅ Reasonable linear relationship")
        else:
            print("   ❌ Weak linear relationship")
        
        # 2. Normality of Residuals
        print("\n2. 📊 Normality of Residuals:")
        shapiro_stat, shapiro_p = stats.shapiro(residuals)
        print(f"   Shapiro-Wilk test: statistic={shapiro_stat:.3f}, p-value={shapiro_p:.3f}")
        if shapiro_p > 0.05:
            print("   ✅ Residuals are normally distributed")
        else:
            print("   ⚠️  Residuals may not be normally distributed")
        
        # 3. Homoscedasticity (constant variance)
        print("\n3. 📏 Homoscedasticity Test:")
        # Breusch-Pagan test would be ideal, but let's use a simpler approach
        # Split residuals into two groups and compare variances
        n_half = len(residuals) // 2
        var1 = np.var(residuals[:n_half])
        var2 = np.var(residuals[n_half:])
        f_stat = max(var1, var2) / min(var1, var2)
        print(f"   Variance ratio test: F-statistic={f_stat:.3f}")
        if f_stat < 3:
            print("   ✅ Homoscedasticity assumption likely satisfied")
        else:
            print("   ⚠️  Potential heteroscedasticity detected")
        
        # 4. Independence (Durbin-Watson test)
        print("\n4. 🔄 Independence Test:")
        try:
            from statsmodels.stats.diagnostic import durbin_watson
            dw_stat = durbin_watson(residuals)
            print(f"   Durbin-Watson statistic: {dw_stat:.3f}")
            if 1.5 <= dw_stat <= 2.5:
                print("   ✅ No significant autocorrelation")
            else:
                print("   ⚠️  Potential autocorrelation detected")
        except Exception as e:
            print(f"   ⚠️  Could not compute Durbin-Watson statistic: {e}")
            print("   Assuming independence for cross-sectional data")
        
        # 5. Multicollinearity
        print("\n5. 🔗 Multicollinearity Check:")
        # Calculate correlation matrix for predictors (excluding constant)
        X_no_const = self.X.drop('const', axis=1)
        corr_matrix = X_no_const.corr()
        print("   Correlation matrix between predictors:")
        print(corr_matrix.round(3))
        
        # Check for high correlations
        high_corr = (corr_matrix.abs() > 0.8) & (corr_matrix.abs() < 1.0)
        if high_corr.any().any():
            print("   ⚠️  High correlation detected between predictors")
        else:
            print("   ✅ No severe multicollinearity detected")
    
    def test_model_performance(self):
        """Test model performance and accuracy"""
        print("\n" + "="*60)
        print("🎯 MODEL PERFORMANCE VALIDATION")
        print("="*60)
        
        # Basic performance metrics
        print("1. 📊 Basic Performance Metrics:")
        print(f"   R-squared: {self.model.rsquared:.3f}")
        print(f"   Adjusted R-squared: {self.model.rsquared_adj:.3f}")
        print(f"   F-statistic: {self.model.fvalue:.3f}")
        print(f"   P-value (F-test): {self.model.f_pvalue:.6f}")
        
        # Interpretation
        if self.model.rsquared > 0.5:
            print("   ✅ Strong explanatory power")
        elif self.model.rsquared > 0.3:
            print("   ⚠️  Moderate explanatory power")
        else:
            print("   ❌ Weak explanatory power")
        
        # Cross-validation
        print("\n2. 🔄 Cross-Validation Performance:")
        # Convert to sklearn format for cross-validation
        X_sklearn = self.X.drop('const', axis=1).values
        y_sklearn = self.y.values
        
        # Leave-One-Out Cross-Validation
        loo = LeaveOneOut()
        lr = LinearRegression()
        loo_scores = cross_val_score(lr, X_sklearn, y_sklearn, cv=loo, scoring='r2')
        
        print(f"   Leave-One-Out R² mean: {loo_scores.mean():.3f}")
        print(f"   Leave-One-Out R² std: {loo_scores.std():.3f}")
        
        # Prediction accuracy metrics
        y_pred = self.model.fittedvalues
        mae = mean_absolute_error(self.y, y_pred)
        rmse = np.sqrt(mean_squared_error(self.y, y_pred))
        
        print(f"   Mean Absolute Error (log scale): {mae:.3f}")
        print(f"   Root Mean Square Error (log scale): {rmse:.3f}")
        
        # Convert to original scale for interpretation
        y_orig = np.exp(self.y)
        y_pred_orig = np.exp(y_pred)
        mae_orig = mean_absolute_error(y_orig, y_pred_orig)
        rmse_orig = np.sqrt(mean_squared_error(y_orig, y_pred_orig))
        
        print(f"   Mean Absolute Error (USD): ${mae_orig:.2f}")
        print(f"   Root Mean Square Error (USD): ${rmse_orig:.2f}")
        
        # Prediction intervals
        print("\n3. 📏 Prediction Confidence:")
        residual_std = np.std(self.model.resid)
        print(f"   Residual standard error: {residual_std:.3f}")
        print(f"   Typical prediction error: ±{residual_std * 1.96:.3f} (log scale)")
        print(f"   Typical prediction error: ±{(np.exp(residual_std * 1.96) - 1) * 100:.1f}% (original scale)")
    
    def test_turkey_prediction(self):
        """Validate Turkey-specific prediction"""
        print("\n" + "="*60)
        print("🇹🇷 TURKEY PREDICTION VALIDATION")
        print("="*60)
        
        # Turkey's input values
        turkey_gdp = 42326.16
        turkey_bb_price = 11.0
        turkey_internet_usage = 0.86
        
        print("Turkey's Economic Indicators:")
        print(f"   GDP PPP: ${turkey_gdp:,.2f}")
        print(f"   Broadband Price: ${turkey_bb_price:.2f}")
        print(f"   Internet Usage: {turkey_internet_usage:.1%}")
        
        # Make prediction
        turkey_input = pd.DataFrame({
            'const': [1],
            'log_GDP': [np.log(turkey_gdp)],
            'log_BBPrice': [np.log(turkey_bb_price)],
            'IntUsage': [turkey_internet_usage]
        })
        
        pred_log_price = self.model.predict(turkey_input)[0]
        pred_price = np.exp(pred_log_price)
        
        # Calculate prediction interval
        pred_se = self.model.get_prediction(turkey_input).se_mean[0]
        pred_interval_lower = np.exp(pred_log_price - 1.96 * pred_se)
        pred_interval_upper = np.exp(pred_log_price + 1.96 * pred_se)
        
        print(f"\n🎯 Turkey Price Prediction:")
        print(f"   Predicted Price: ${pred_price:.2f}")
        print(f"   95% Confidence Interval: ${pred_interval_lower:.2f} - ${pred_interval_upper:.2f}")
        
        # Compare with similar countries
        print("\n🔍 Comparison with Similar Countries:")
        # Find countries with similar GDP PPP
        similar_gdp = self.df[abs(self.df['GDP_PPP'] - turkey_gdp) < 10000].copy()
        if len(similar_gdp) > 0:
            print("   Countries with similar GDP PPP:")
            for _, row in similar_gdp.iterrows():
                print(f"     {row['Country Name']}: GDP ${row['GDP_PPP']:,.0f}, Starlink ${row['Starlink_Price']:.0f}")
        
        # Find countries with similar broadband prices
        similar_bb = self.df[abs(self.df['BB_Price'] - turkey_bb_price) < 5].copy()
        if len(similar_bb) > 0:
            print("   Countries with similar Broadband Prices:")
            for _, row in similar_bb.iterrows():
                print(f"     {row['Country Name']}: BB ${row['BB_Price']:.2f}, Starlink ${row['Starlink_Price']:.0f}")
        
        # Validate prediction reasonableness
        print("\n✅ Prediction Reasonableness Check:")
        avg_price = self.df['Starlink_Price'].mean()
        if abs(pred_price - avg_price) < 2 * self.df['Starlink_Price'].std():
            print(f"   ✅ Predicted price (${pred_price:.2f}) is within reasonable range")
        else:
            print(f"   ⚠️  Predicted price (${pred_price:.2f}) may be unusual")
        
        print(f"   Global average Starlink price: ${avg_price:.2f}")
        print(f"   Turkey prediction vs average: {((pred_price - avg_price) / avg_price * 100):+.1f}%")
    
    def test_sensitivity_analysis(self):
        """Test sensitivity to input parameters"""
        print("\n" + "="*60)
        print("🎛️  SENSITIVITY ANALYSIS")
        print("="*60)
        
        # Base Turkey values
        base_gdp = 42326.16
        base_bb_price = 11.0
        base_internet_usage = 0.86
        
        # Test GDP sensitivity
        print("1. 📈 GDP Sensitivity Analysis:")
        gdp_scenarios = [base_gdp * 0.8, base_gdp * 0.9, base_gdp, base_gdp * 1.1, base_gdp * 1.2]
        for gdp in gdp_scenarios:
            turkey_input = pd.DataFrame({
                'const': [1],
                'log_GDP': [np.log(gdp)],
                'log_BBPrice': [np.log(base_bb_price)],
                'IntUsage': [base_internet_usage]
            })
            pred_price = np.exp(self.model.predict(turkey_input)[0])
            change = (gdp - base_gdp) / base_gdp * 100
            print(f"   GDP {change:+.0f}%: ${pred_price:.2f}")
        
        # Test Broadband Price sensitivity
        print("\n2. 💰 Broadband Price Sensitivity Analysis:")
        bb_scenarios = [base_bb_price * 0.5, base_bb_price * 0.75, base_bb_price, base_bb_price * 1.5, base_bb_price * 2.0]
        for bb_price in bb_scenarios:
            turkey_input = pd.DataFrame({
                'const': [1],
                'log_GDP': [np.log(base_gdp)],
                'log_BBPrice': [np.log(bb_price)],
                'IntUsage': [base_internet_usage]
            })
            pred_price = np.exp(self.model.predict(turkey_input)[0])
            change = (bb_price - base_bb_price) / base_bb_price * 100
            print(f"   BB Price {change:+.0f}%: ${pred_price:.2f}")
        
        # Test Internet Usage sensitivity
        print("\n3. 🌐 Internet Usage Sensitivity Analysis:")
        internet_scenarios = [0.6, 0.7, 0.8, 0.86, 0.9, 0.95]
        for internet_usage in internet_scenarios:
            turkey_input = pd.DataFrame({
                'const': [1],
                'log_GDP': [np.log(base_gdp)],
                'log_BBPrice': [np.log(base_bb_price)],
                'IntUsage': [internet_usage]
            })
            pred_price = np.exp(self.model.predict(turkey_input)[0])
            change = (internet_usage - base_internet_usage) / base_internet_usage * 100
            print(f"   Internet Usage {internet_usage:.0%} ({change:+.0f}%): ${pred_price:.2f}")
    
    def run_all_tests(self):
        """Run all validation tests"""
        print("🚀 STARTING COMPREHENSIVE VALIDATION TESTS")
        print("="*60)
        
        self.test_data_quality()
        self.test_model_assumptions()
        self.test_model_performance()
        self.test_turkey_prediction()
        self.test_sensitivity_analysis()
        
        print("\n" + "="*60)
        print("✅ ALL VALIDATION TESTS COMPLETED")
        print("="*60)
        
        # Summary recommendations
        print("\n📋 SUMMARY RECOMMENDATIONS:")
        
        # Model quality assessment
        r2 = self.model.rsquared
        if r2 > 0.3:
            print("✅ Model has reasonable explanatory power")
        else:
            print("⚠️  Model has limited explanatory power - consider additional variables")
        
        # Coefficient significance
        significant_vars = (self.model.pvalues < 0.05).sum()
        if significant_vars > 1:
            print("✅ Multiple variables are statistically significant")
        else:
            print("⚠️  Few variables are statistically significant")
        
        # Prediction reliability
        residual_std = np.std(self.model.resid)
        if residual_std < 0.3:
            print("✅ Predictions have reasonable precision")
        else:
            print("⚠️  Predictions have high uncertainty")
        
        # Turkey-specific assessment
        turkey_input = pd.DataFrame({
            'const': [1],
            'log_GDP': [np.log(42326.16)],
            'log_BBPrice': [np.log(11.0)],
            'IntUsage': [0.86]
        })
        turkey_pred = np.exp(self.model.predict(turkey_input)[0])
        
        print(f"\n🇹🇷 Turkey Price Prediction: ${turkey_pred:.2f}")
        print("   This prediction should be used as a starting point for pricing strategy")
        print("   Consider local market conditions, competition, and regulatory factors")

if __name__ == "__main__":
    validator = StarLinkPriceValidator()
    validator.run_all_tests() 