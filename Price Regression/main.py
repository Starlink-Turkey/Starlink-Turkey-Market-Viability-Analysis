import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load and preprocess dataset
df = pd.read_csv('Data.csv', delimiter=';')

df['GDP_PPP'] = df['GDP per capita, PPP (current international $) [2023]'] \
    .str.replace(',', '.').astype(float)
df['Internet_Usage'] = df['Individuals using the Internet (% of population) [2023]'] \
    .str.replace(',', '.').astype(float) / 100
df['BB_Price'] = df['InternetCost_BroadbandCostPerMonth_USD_2024'] \
    .str.replace(',', '.').astype(float)
df['Starlink_Price'] = df['Starlink Price']
df['Available'] = df['Starlink Available'].str.lower()

# Filter to markets where Starlink is available
df = df[df['Available'] == 'yes'].dropna(subset=['GDP_PPP', 'Internet_Usage', 'BB_Price', 'Starlink_Price'])

# Prepare regression variables
X = pd.DataFrame({
    'const': 1,
    'log_GDP': np.log(df['GDP_PPP']),
    'log_BBPrice': np.log(df['BB_Price']),
    'IntUsage': df['Internet_Usage']
})
y = np.log(df['Starlink_Price'])

# Fit the log-linear regression model
model = sm.OLS(y, X).fit()

# Display regression summary
print(model.summary())

# Predict for Turkey
turkey_input = pd.DataFrame({
    'const': [1],
    'log_GDP': [np.log(42326.16)],
    'log_BBPrice': [np.log(11)],
    'IntUsage': [0.86]
})
pred_log_price = model.predict(turkey_input)[0]
pred_price = np.exp(pred_log_price)

print(f"\nPredicted Starlink monthly price for Turkey: ${pred_price:.2f} USD")
