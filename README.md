# Lloyds Banking Group - Transactional Data Analytics 
A data science project simulating real-world banking analytics use cases: fraud detection, customer segmentation, and predictive balance forecasting using machine learning.

## Problem Overview
Retail banks generate millions of transactions daily - and within that volume lie two of the most commercially significant analytical challenges in financial services:
* Understanding who their customers are and how to serve them differently based on their financial behavior.
* Forecasting how customer balances will evolve to support proactive financial planning and risk management.

Despite the availability of rich transactional data, there is often limited structured analysis that translates raw transaction behavior to actionable customer intelligence.

This project delivers an end-to-end transactional analytics solution built on simulated UK retail banking data, demonstrating how machine learning can address both challenges within a cohesive pipeline.

The project is structured around two interconnected analytical workstreams:
1. **Customer Segmentation** - clustering 1200+ customers into behaviorally distinct segments using Hierarchical Clustering, validated against K-Means, DBSCAN and GMM with PCA and t-SNE for visualization
2. **Future Balance Prediction** - forecasting customer account balances using regression models (Linear Regression, Random Forest, XGBoost), with SHAP-based interpretability.

## Business Objectives
* Segment 1200+ customers into behaviorally distinct groups based on transaction patterns, balance behavior and temporal activity.
* Identify and profile each customer segment to support targeted product and engagement strategies.
* Forecast future account balances at the individual customer level to enable proactive risk and product decisions.
* Identify the key behavioral drivers of balance movement through interpretable modelling.
* Deliver model-backed insights to support risk, product, and customer teams.

## Data Source & Context
The dataset simulates a year's worth of retail banking transactions across multiple customer accounts, modelled on typical UK retail banking transaction flows - including merchant spend, peer-to-peer transfers, and timestamped account activity.

| **Column** | **Description** |
|------------|-----------------|
| `Date` | Transaction date (DD/MM/YYYY) |
| `Timestamp` | Time of transaction |
| `Account No` | Unique customer account identifier |
| `Balance` | Account balance post-transaction |
| `Amount` | Transaction amount (positive=credit, negative=debit) |
| `Third Party Account No` | Counterpart account for P2P transfers |
| `Third Party Name` | Merchant or entity name for retail transactions |

### Key Characteristics
* Includes both merchant transactions (e.g., Amazon, Boots, Dorothy Perkins) and peer-to-peer transfers between accounts
* Right-skewed transaction amount distribution with a concentration of low-value activity and a tail of high-value outliers
* No pre-assigned segment labels - customer groups are derived entirely from behavioral patterns in the data

## Analytical Approach
### Customer Segmentation
**Notebook:** Customer Segmentation.ipynb

#### Data Understanding & Preparation
* Null value assessment across all columns; rows with missing values in key fields (`Date`, `Timestamp`, `Account No`, `Balance`, `Amount`, `Third Party Name`) were dropped.
* `Third Party Account No` removed as it as not analytically relevant to customer-level segmentation.
* Duplicate record check performed to ensure each transaction was counted once.

Clean data is especially important in unsupervised learning - clustering algorithms have no error signal to work against, so noise and missing values directly distort the cluster boundaries that emerge. The conservative approach to dropping incomplete records reflects this sensitivity.

#### Exploratory Data Analysis
* Balance distribution found to be approximately normal with a rightward skew and significant high-balance outliers - indicating a subset of high-net-worth or high-activity accounts.
* Transaction amount distribution found to be highly right-skewed, confirming that most transactions are small in value with occasional large outliers consistent with bulk or high-value spend.
* Box plots used to visually confirm the outlier structure in both dimensions before feature engineering.

#### Feature Engineering
Customer-level features were aggregated from the raw transaction data to create asingle profile per account - the unit of analysis for clustering:
* **Transaction aggregates**: total transaction amount, average transaction amount, transaction count.
* **Balance metrics**: average balance, minimum balance, maximum balance
* **Temporal patterns**: most common transaction hour, day and month - encoded using **cyclical sine/cosine transformation** to preserve the circular nature of time.

The cyclical encoding step is deliberate feature engineering choice that linear models and clustering algorithms would otherwise get wrong - treating time as a raw integer causes artificial distance between values that are behaviorally adjacent.

#### Clustering Algorithm Comparison
Four clustering approaches were evaluated to identify the best fit for this dataset:
* **K-Means** - Elbow method and Silhouette Score analysis suggested an optimal k of 4; however, final evaluation scores were poor indicating the algorithm struggled with feature space structure.
* **DBSCAN** - Returned only noise labels across all tested parameter combinations, confirming it is unsuitable for this dataset's density characteristics.
* **Gaussian Mixture Model (GMM)** - Negative Silhouette Score (-0.06) and very high Davies-Boulding Index (44.73) confirmed significant cluster overlap and poor fit.
* **Hierarchical Clustering** - Achieved the strongest results by a significant margin, producing compact, well-separated clusters confirmed visually via dendrogram.

The algorithm comparison is not just due diligence - it surfaces a meaningful finding: the customer behavioral space in this dataset has a hierarchical structure that K-Means and density-based methods cannot capture. Agglomerative clustering's bottom-up merging process is better suited to datasets where clusters are nested or unevenly sized.

#### Dimensionality Reduction & Visualization
* **PCA** applied to project clusters into 2D - revealed vertical alignment with limited inter-cluster separation, indicating high variance in one principal component but limited discriminatory power across the full feature space.
* **t-SNE** applied as a complementary view - produced well-separated cluster groupings, confirming that the Hierarchical Clustering structure is real and non-linear in nature.

The contrast between PCA and t-SNE is instructive: PCA's linear projection can obscure cluster quality that actually exists in non-linear feature relationships. The t-SNE result provides stronger visual confirmation that the four segments are genuinely distinct in behavioral terms.

#### Customer Segment Profiles
Four distinct customer segments were identified and profiled:
| **Segment** | **Profile** | **Characteristics** |
|-------------|-------------|---------------------|
| **Cluster 0** | High-Value Frequent Customers | High balance, moderate-to-high transaction volume, evening activity peak |
| **Cluster 1** | Moderate Engagement, Low Balance | Average transactions, lower balance, price-sensitive behavior |
| **Cluster 2** | Low Engagement Customers | Fewer transactions, lower balances, minimal platform activity |
| **Cluster 3** | Occasional High-Value Spenders | Higher average transaction amounts but lower frequency, evening peak |

### Future Balance Prediction
**Notebook** Predicting Future Balance Using Machine Learning.ipynb

#### Data Understanding & Preparation
* Missing `Balance` values filled via forward-fill grouped by account, preserving time-series continuity; median imputation applied as a fallback.
* Missing `Amount` values filled with 0, reflecting a no-transaction assumption.
* Unknown third-party accounts and merchant names standardised as `"Unknown"`.

Forward-filling balance values by account - rather than across the whole dataset - is a deliberate design choice. Carrying forward a different customer's balance would introduce false signal into the model, since accounts behave independently. Grouping by account before imputation ensures that continuity is maintained only where it is analytically valid.

#### Feature Engineering
* `Date` and `Timestamp` combined into a unified `Datetime` column for consistent temporal handling.
* Time-based features extracted: `Year`, `Month`, `Day`, `Hour`, `DayOfWeek`.
* Dataset sorted chronologically to maintain time-series integrity.
* Target variable `Future_Balance` created via `.shift(-1)` - representing the next recorded balance for each account.

The `.shift(-1)` approach frames this as a supervised learning probelm: given everything known about a transaction at time *t*, predict the resulting balance at time *t+1*. This is a practical forecasting frame for banking — it mirrors how balance-based product triggers would operate in a live environment, where decisions are made in the moment before the next event occurs.

#### ML Model Training & Comparison
Three regression models were trained on an 80/20 train-test split:
* **Linear Regression** — used as a performance baseline to establish the ceiling of a purely linear explanation.
* **Random Forest Regressor** (100 estimators) — ensemble method to capture non-linear interactions between balance, amount, and time features.
* **XGBoost Regressor** (200 estimators, learning rate 0.1) — gradient boosting for stronger sequential error correction and predictive accuracy

The progression from Linear Regression to Random Forest to XGBoost is intentional — each step tests whether additional model complexity is justified by measurable performance gains. The consistent improvement in R² across all three confirms that balance dynamics contain non-linear structure that simpler models cannot capture.

#### Hyperparameter Tuning
The XGBoost model was further tuned with 300 estimators, learning rate 0.05, max depth 6, and subsampling of 0.8 — balancing model complexity against overfitting risk. The lower learning rate paired with more estimators allows the model to correct errors more gradually, producing more stable predictions on unseen data.

#### Interpretability — SHAP Analysis
SHAP values were computed for the tuned XGBoost model to explain the directional contribution of each feature at the individual prediction level — critical in a banking context where regulatory expectations increasingly require models to be interpretable, not just accurate.

#### Time Series Forecasting - ARIMA, SARIMA & ETS
To extend analysis to a true forward-looking forecast, the balance data was resampled to monthly frequency and tested for stationarity and seasonality before fitting time series models:
* **Stationarity testing** — ACF and PACF plots generated and an Augmented Dickey-Fuller (ADF) test run to assess whether differencing was required before modelling.
* **Seasonal decomposition** — trend, seasonality, and residual components isolated to understand the underlying structure of monthly balance behavior before model selection.
* **ARIMA** — grid search over (p, d, q) parameter combinations using AIC as the selection criterion; the best-fit model produced a flat forecast, confirming ARIMA cannot capture the trend and seasonal variation present in this dataset.
* **SARIMA** — extended ARIMA with seasonal order parameters, grid-searched across all viable combinations; captured both trend and seasonal components, producing a more realistic forward-looking forecast than ARIMA.
* **ETS (Holt-Winters Exponential Smoothing)** — fitted with additive trend and seasonality components; produced the best visual fit to observed balance fluctuations, adapting well to recent movements by weighting recent observations more heavily.

The three-model comparison was deliberate — ARIMA's failure to capture trend and seasonality is an informative result, not just a dead end. It confirmed that the balance series has meaningful temporal structure that requires seasonal modelling, and that ETS's adaptive weighting makes it the most suitable approach for a dataset with irregular fluctuations.

**Final 12-Month Forecast**
The ETS model was used to generate a 12-month forward-looking balance forecast, visualised against observed historical data. The forecast closely tracks the directional trajectory of balance movement, though it smooths out extreme short-term volatility — an expected trade-off when producing monthly-level planning outputs from a transaction-level dataset.

### Key Insights
* **Hierarchical Clustering outperformed all other methods tested**, achieving a Silhouette Score of 0.558 and a Davies-Bouldin Index of 0.509 — confirming that customer behavioral segments in this dataset have a nested, hierarchical structure that K-Means, DBSCAN, and GMM could not capture. The algorithm comparison was not just procedural; it revealed something meaningful about the nature of the data.
* **Four behaviorally distinct customer segments emerged**, each with clear and actionable profiles. High-value frequent customers, moderate-engagement low-balance customers, low-engagement dormant accounts, and occasional high-value spenders represent meaningfully different relationships with the bank — and therefore different product and engagement strategies.
* **Temporal patterns — particularly transaction hour — are consistent within segments**, with all four clusters peaking in the evening. This suggests that time-of-day is a behavioral constant across the customer base rather than a differentiator between segments, but it does confirm that evening is the primary window for customer engagement regardless of segment.
* **Current balance is the dominant predictor of future balance**, contributing the highest SHAP importance scores in the forecasting model by a significant margin. This reflects strong autocorrelation in account trajectories — a customer's balance today is the most reliable signal of their balance tomorrow, with direct implications for how balance-based product triggers should be designed.
* **Transaction amount has limited standalone predictive power** for balance forecasting. As a single point-in-time value rather than an aggregated behavioral measure, it does not carry sufficient signal on its own. Models built around cumulative spend patterns or rolling averages would likely improve on this in a production setting.
* **Temporal features contribute marginally but meaningfully** to balance prediction, capturing seasonal and intra-day spending cycles — payroll timing, weekend behavior, end-of-month pressure — that matter at the portfolio level even if their per-prediction impact is small.

### Model Performance Summary
#### Customer Segmentation - Clustering Comparison
| **Algorithm** | **Silhouette Score ↑** | **Davies-Bouldin Index ↓** | **Outcome** |
|---------------|-----------------------|----------------------------|-------------|
| **K-Means** | -0.09 | 46.64 | Poor fit |
| **DBSCAN** | Failed | Failed | Not suitable |
| **GMM** | -0.06 | 44.73 | Poor fit |
| **Hierarchical Clustering** | **0.558** | **0.509** | **Best — selected** |

#### Time Series Forecasting — Model Comparison
| **Model** | **Performance** | **Strengths** | **Outcome**|
|-----------|-----------------|---------------|------------|
| **ARIMA** | Poor — flat forecast | Works for stationary series | Cannot handle trend or seasonality |
| **SARIMA** | Good | Captures trend & seasonal components | Better fit, needs sufficient data | 
| **ETS** | **Best** | **Adapts well to fluctuations, weights recent data** | **Selected for 12-month forecast** |

| **Model** | **MAE** | **MSE** | **RMSE** | **R²** |
|-----------|---------|---------|----------|--------|
| **Linear Regression** | 729.48 | 1,549,730.62 | 1,244.88 | 0.8306 |
| **Random Forest** | 680.00 | 1,211,531.94 | 1,100.70 | 0.8676 |
| **XGBoost (Initial)** | 643.54 | 1,141,166.78 | 1,068.25 | 0.8753 |
| **XGBoost (Tuned)** | **646.22** | **1,136,706.68** | **1,066.16** | **0.8758** | 

The tuned XGBoost model achieved the strongest overall forecasting performance, explaining approximately 88% of the variance in future account balances. MSE and RMSE improvements over the initial XGBoost confirm that tuning reduced extreme prediction errors, though the marginal MAE increase reflects a minor trade-off in average absolute error.

### Decision Support Use Cases
This analysis enables stakeholders to:
* Design targeted retention, reactivation, and upsell strategies for each of the four identified customer segments.
* Prioritise high-value frequent customers for premium product offerings and VIP engagement programs.
* Identify low-engagement accounts early and trigger reactivation campaigns before they churn.
* Forecast individual account balances to inform overdraft, credit, and savings product decisions.
* Identify the behavioral drivers of balance movement to personalise financial guidance at scale.
* Track how segment membership and balance trajectories evolve over time as a measure of strategy effectiveness.

### Tools & Technology
* **Python** (Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn)
* **Machine Learning Models**
* **Statsmodels** (ARIMA, SARIMA, ETS)
* **SHAP**
* **Jupyter Notebook**
* **Git & GitHub**

### Business Impact & Next Steps
The two workstreams in this project address problems that sit at the core of retail banking analytics — and the methodological choices made throughout are designed to be translatable into real operational contexts, not just demonstrative of technical capability.
* **Customer Segmentation** - The four segments identified through Hierarchical Clustering give the bank a structured, data-driven lens for personalising customer engagement at scale. In practice, each segment maps to a distinct commercial strategy: Cluster 0 (high-value frequent customers) warrants premium retention investment; Cluster 1 (moderate engagement, low balance) presents an opportunity for balance-building products like fixed savings accounts or cashback incentives; Cluster 2 (low engagement) is a reactivation priority where targeted outreach could recover dormant relationships before they fully lapse; and Cluster 3 (occasional high-value spenders) is best served through exclusive event-based offers timed to their natural transaction patterns. The rigorous algorithm comparison — testing K-Means, DBSCAN, GMM, and Hierarchical Clustering — adds methodological credibility to the segmentation output and ensures the chosen model is justified rather than assumed.
* **Balance Forecasting** - The 12-month forward-looking balance forecast has direct operational application across three product areas: overdraft risk management (flagging accounts on a downward trajectory before they breach limits), proactive savings nudges (identifying accounts with growing balances that may be ready for higher-yield products), and credit pre-qualification (using balance stability as an input signal alongside traditional credit metrics). The SHAP interpretability layer ensures these interventions are explainable — a regulatory and governance requirement that is increasingly non-negotiable in financial services. In a production environment, the monthly aggregated forecasts would be the primary output surfaced to product and planning teams, with the transaction-level model running as the underlying engine.

#### Future Enhancements:
* **Segment tracking over time** — Monitor how customers move between segments month-on-month to identify early signals of churn, financial distress, or upgrade potential.
* **Personalisation engine integration** — Feed segment labels directly into a product recommendation engine to automate offer targeting based on behavioral profile.
* **Behavioral feature enrichment** — Incorporate rolling aggregates (30-day average spend, transaction velocity, merchant diversity score) to deepen both the segmentation and forecasting signal beyond point-in-time values.
* **Portfolio-level forecasting** — Extend the balance prediction framework from individual accounts to segment and portfolio aggregation, enabling treasury and liquidity planning use cases.

### Author
Nandhitha Sivakumar
Focused on analytics-driven business decision-making
