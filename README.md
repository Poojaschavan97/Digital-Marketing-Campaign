# 📊 Digital Marketing Campaign Analysis

### End-to-End Data Science Project
EDA · Feature Engineering · Preprocessing · Modeling · Evaluation · Business Insights

---

## 📢 Problem Statement

The objective of this project is to analyze digital marketing campaign performance and identify the key factors influencing customer conversion. The dataset includes customer demographics, campaign details, engagement metrics, and website interaction data.

**Goals:**
- Evaluate the effectiveness of different marketing channels and campaign types
- Identify customer segments most likely to convert
- Understand how advertising spend and engagement drive conversion
- Build a predictive model to flag high-conversion-probability customers
- Translate findings into actionable business recommendations

---

## 👨‍💼 Business Questions

1. Which marketing channel delivers the highest conversion rate and ROI?
2. How does advertising spend (AdSpend) impact conversion outcomes?
3. Which customer segments (Age, Gender, Income) are most likely to convert?
4. How does website engagement (TimeOnSite, PagesPerVisit, WebsiteVisits) relate to conversion?
5. Where in the customer journey (funnel) does the biggest drop-off occur?

---

## 🗂️ Dataset Overview

| Column | Description |
|---|---|
| CustomerID | Unique identifier assigned to each customer |
| Age | Age of the customer |
| Gender | Gender of the customer |
| Income | Income level indicating purchasing power |
| CampaignChannel | Marketing channel used (Email, Social Media, Search Ads, etc.) |
| CampaignType | Type of campaign (Awareness, Conversion, Retargeting, etc.) |
| AdSpend | Amount spent on advertising for the campaign/customer |
| ClickThroughRate (CTR) | Percentage of users who clicked on the advertisement |
| ConversionRate | Historical conversion rate of the customer |
| WebsiteVisits | Number of visits to the website |
| PagesPerVisit | Average pages viewed per visit |
| TimeOnSite | Average time spent on site |
| SocialShares | Number of times content was shared on social media |
| EmailOpens / EmailClicks | Email engagement metrics |
| PreviousPurchases | Number of prior purchases made by the customer |
| LoyaltyPoints | Accumulated loyalty program points |
| Conversion | Target variable — whether the customer converted (1) or not (0) |

- **Rows:** 8,000 &nbsp;|&nbsp; **Columns:** 20
- No missing values or duplicate records found
- `AdvertisingPlatform` and `AdvertisingTool` were dropped (zero variance — single unique value)

---

## 🔍 Project Workflow

### 1. Data Gathering & Quality Checks
- Verified data types, checked for missing values and duplicates
- Scanned categorical columns for anomalies and rare categories

### 2. Exploratory Data Analysis (EDA)

**Univariate Analysis**
- Outlier detection via boxplots across numerical features
- Distribution analysis via histograms and countplots

<!-- 📌 Insert visualization: Boxplots of numerical features -->
<!-- ![Outlier Analysis](images/boxplots.png) -->

<!-- 📌 Insert visualization: Distribution histograms -->
<!-- ![Feature Distributions](images/histograms.png) -->

**Bivariate Analysis (vs. Conversion)**
- Conversion rate by Campaign Channel and Campaign Type
- AdSpend vs. Conversion
- Age distribution by Conversion
- Previous Purchases vs. Conversion
- Gender vs. Conversion
- Correlation heatmap of numerical features

<!-- 📌 Insert visualization: Conversion rate by channel/campaign type -->
<!-- ![Conversion by Channel](images/conversion_by_channel.png) -->

<!-- 📌 Insert visualization: Correlation heatmap -->
<!-- ![Correlation Heatmap](images/correlation_heatmap.png) -->

**Key EDA Findings**
- Class imbalance: ~87.7% of customers converted → requires SMOTE + AUC/F1 evaluation instead of raw accuracy
- Channel alone is not a strong differentiator — conversion rates are similar across channels
- `PreviousPurchases` and `LoyaltyPoints` show the strongest positive correlation with conversion

### 3. Feature Engineering

Five new features were engineered to capture behavior not directly present in the raw data:

| Feature | Description |
|---|---|
| `Email_CTR` | Ratio of email clicks to opens — measures email content effectiveness |
| `AdEfficiency` | Conversion rate per unit of ad spend — captures ROI efficiency |
| `EngagementScore` | Weighted composite of website visits, pages/visit, time on site, and social shares |
| `CustomerValue` | Weighted combination of loyalty points and previous purchases |
| `AgeGroup` | Binned age brackets (18–30, 31–45, 46–60, 61+) |

### 4. Preprocessing Pipeline
- Numerical features scaled with `StandardScaler`
- Categorical features encoded with `OneHotEncoder`
- Combined via `ColumnTransformer` for a single, reusable pipeline
- Stratified train-test split to preserve class balance

### 5. Handling Class Imbalance
- Applied **SMOTE** (Synthetic Minority Oversampling) inside an `imblearn` pipeline — fit only on training data to prevent data leakage

<!-- 📌 Insert visualization: Class distribution before/after SMOTE -->
<!-- ![SMOTE Before/After](images/smote_balance.png) -->

### 6. Model Building & Comparison

Models evaluated using 5-fold stratified cross-validation on ROC-AUC, F1, and Accuracy:

- Logistic Regression
- Random Forest
- Gradient Boosting
- **XGBoost** ✅ (best performer)

<!-- 📌 Insert visualization: Model comparison bar charts (ROC-AUC / F1 / Accuracy) -->
<!-- ![Model Comparison](images/model_comparison.png) -->

### 7. Hyperparameter Tuning
- `GridSearchCV` over `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`
- Optimized for ROC-AUC using a custom scorer

### 8. Final Model Evaluation

Final tuned **XGBoost** model evaluated on the held-out test set:

- ROC-AUC Score
- F1 Score
- Confusion Matrix
- ROC Curve

<!-- 📌 Insert visualization: Confusion matrix, ROC curve -->
<!-- ![Model Evaluation](images/final_model_evaluation.png) -->

### 9. Feature Importance & SHAP Analysis
- Built-in XGBoost feature importances
- SHAP summary plot for interpretability — shows direction and magnitude of each feature's impact on conversion

<!-- 📌 Insert visualization: SHAP summary plot -->
<!-- ![SHAP Summary](images/shap_summary.png) -->

**Key Findings**
- `LoyaltyPoints` and `PreviousPurchases` are the strongest predictors — existing customers convert far more reliably
- The engineered `EngagementScore` ranks highly, validating the feature engineering approach
- `ConversionRate` and `ClickThroughRate` matter more than raw ad spend volume

### 10. Business Insights & Funnel Analysis
- Average AdSpend by channel
- Conversion rate by age group
- Conversion funnel: Total Customers → Website Visitors → Email Openers → Email Clickers → Converted, with stage-to-stage drop-off rates

<!-- 📌 Insert visualization: Conversion funnel chart -->
<!-- ![Conversion Funnel](images/conversion_funnel.png) -->

<!-- 📌 Insert visualization: AdSpend by channel / Conversion by age group -->
<!-- ![Business Insights Dashboard](images/business_insights.png) -->

---

## 💡 Business Recommendations

| Finding | Recommendation |
|---|---|
| Loyal customers convert 3x more | Prioritize retention and loyalty programs over pure acquisition spend |
| Largest funnel drop-off is between website visit and email open | Invest in email subject-line testing and re-engagement campaigns |
| Ad spend alone isn't a strong predictor | Shift budget allocation toward high-engagement channels rather than high-spend channels |
| Engagement metrics outperform demographic ones | Use behavior-based (not just demographic) targeting for campaigns |

---

## 🛠️ Tech Stack

- **Data Handling:** pandas, numpy
- **Visualization:** matplotlib, seaborn
- **Preprocessing & Modeling:** scikit-learn, imbalanced-learn (SMOTE), XGBoost
- **Interpretability:** SHAP
- **Environment:** Jupyter Notebook

---

## 📁 Project Structure

```
digital-marketing-campaign-analysis/
│
├── Digital_Marketing.ipynb          # Main analysis notebook
├── digital_marketing_campaign_dataset.csv   # Raw dataset
├── digital_marketing_model.pkl      # Exported model pipeline (preprocessor + XGBoost)
├── images/                          # Visualizations referenced in this README
└── README.md
```

---

## ✅ Project Summary

**End-to-End Pipeline Completed:**
1. Problem framing with clear business questions
2. Thorough EDA — distributions, outliers, bivariate analysis, correlation
3. Feature engineering — 5 new meaningful features created
4. Preprocessing pipeline — StandardScaler + OneHotEncoder via ColumnTransformer
5. Class imbalance handled via SMOTE (leakage-safe, inside a pipeline)
6. Model comparison across 4 algorithms with cross-validation
7. Hyperparameter tuning of the best model (XGBoost)
8. Model interpretability via feature importance and SHAP
9. Business insights and funnel analysis translated into actionable recommendations
10. Final model exported for reuse

---

## 🚀 How to Run

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn xgboost shap
   ```
3. Open `Digital_Marketing.ipynb` in Jupyter Notebook / JupyterLab
4. Run all cells in order

---

