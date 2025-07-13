# Price Regression Analysis - README

## Overview
This directory contains a regression analysis to predict optimal Starlink pricing for Turkey based on economic indicators from 76 countries where Starlink is available.

## Files
- `main.py` - Main regression analysis script
- `validation_tests.py` - Comprehensive validation testing suite
- `validation_report.md` - Detailed analysis and business recommendations
- `Data.csv` - Country-level economic and pricing data
- `Country Data Excel.xlsx` - Source Excel file with additional data

## Quick Results

### Turkey Price Prediction: **$56.11 USD**
- 95% Confidence Interval: $51.42 - $61.23
- 2.4% below global average ($57.50)
- Competitive positioning in Turkish market

### Model Performance
- **R² = 0.185** (18.5% of price variation explained)
- **RMSE = $13.08** (reasonable accuracy for strategic planning)
- **F-test p-value = 0.002** (statistically significant overall)

## Key Findings

### ✅ Strengths
- Reasonable prediction accuracy for business planning
- Turkey prediction aligns with similar countries (Bulgaria $55, Croatia $57, Romania $52)
- Good data quality with 76 countries and no missing values
- Statistically significant model overall

### ⚠️ Limitations
- Low explanatory power (R² = 0.185)
- No individual predictors are statistically significant
- High multicollinearity between GDP and Internet Usage (r = 0.931)
- Non-normal residuals suggest missing variables

### 📊 Sensitivity Analysis
- **Internet Usage**: Most important factor (±10% usage → ±$1.65 price change)
- **Broadband Price**: Moderate impact (±50% price → ±$1.32 change)
- **GDP**: Low impact (±20% GDP → ±$0.25 price change)

## Usage Instructions

### Run Basic Analysis
```bash
python main.py
```

### Run Comprehensive Validation
```bash
python validation_tests.py
```

### Required Dependencies
```bash
pip install pandas numpy statsmodels scikit-learn scipy matplotlib seaborn
```

## Business Recommendations

1. **Use $56.11 as baseline price** for Turkey market entry
2. **Consider pricing range of $51-$61** based on confidence interval
3. **Conduct additional market research** to validate local factors:
   - Regulatory environment
   - Local competition
   - Consumer purchasing power
   - Currency fluctuations

4. **Monitor Internet Usage trends** as key price sensitivity factor
5. **Position competitively** - slightly below global average

## Validation Results Summary

The comprehensive validation testing revealed:
- ✅ **Data Quality**: Clean dataset with proper preprocessing
- ✅ **Model Assumptions**: Most assumptions satisfied (linearity, homoscedasticity)
- ⚠️ **Normality**: Residuals not normally distributed
- ⚠️ **Multicollinearity**: High correlation between predictors
- ✅ **Business Logic**: Prediction aligns with comparable countries
- ✅ **Sensitivity**: Reasonable response to input changes

## Conclusion

The model provides a **solid starting point** for Turkey pricing strategy but should be supplemented with local market research. The predicted price of **$56.11 USD** positions Starlink competitively while maintaining reasonable profit margins.

**Next Steps:**
1. Use $56.11 as baseline in business planning
2. Conduct local market research to validate assumptions
3. Consider additional factors (regulation, competition, currency)
4. Monitor model performance with new data as markets evolve 