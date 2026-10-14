# Multiple Linear Regression & Feature Selection Explorer (AIoT-DIC2)

An interactive, high-performance web dashboard and REST API for comparing **5 Feature Selection Algorithms** (Lasso L1, Forward SFS, RFE, ElasticNet, and Exhaustive Best Subset) across feature subsets $k=1\dots 5$ on the **50 Startups** benchmark dataset.

![Dashboard Preview](dashboard_demo.png)

---

## 🚀 How to Run Locally with Flask

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Start the Flask Application
```bash
python app.py
```

### 3. Open in Browser
Navigate to: [http://localhost:5000/](http://localhost:5000/)

---

## 🌐 Flask REST API Endpoints

- **`GET /`**: Renders the main interactive regression explorer dashboard.
- **`GET /api/models`**: Returns the complete 5-algorithm feature selection evaluation data ($k=1\dots 5$).
- **`POST /api/predict`**: Live inference endpoint calculating predicted profit and formula contribution breakdown:
  ```json
  {
    "algo": "Lasso",
    "k": 2,
    "rd_spend": 100000,
    "marketing_spend": 150000,
    "admin_spend": 120000,
    "state": "California"
  }
  ```
- **`GET /api/summary`**: Returns summary metrics across all algorithms.

---

## ✨ Key Web App Features

- **Algorithm Switcher**: Switch dynamically between Lasso (L1), Forward SFS, RFE, ElasticNet, Exhaustive Search, or Compare All 5 Simultaneously.
- **Interactive Feature Count Slider ($k=1\dots 5$)**: Instantly observe model metrics and equation changes across different subset sizes.
- **Dual-Panel Responsive Charts**: Interactive Chart.js visualizations for *"RMSE by Number of Features"* and *"R-squared by Number of Features"* with sweet-spot highlighting and hover tooltips.
- **Dynamic Regression Equation**: Live mathematical formula updating in real-time as predictors are added or removed.
- **StatsUnlock-Style Coefficients Table**: Full inferential statistics including $\beta$ coefficients, Standard Error, $t$-values, $p$-values, 95% Confidence Intervals, and VIF (Variance Inflation Factor).
- **Live Startup Profit Predictor**: Interactive input sliders for R&D Spend, Administration, Marketing Spend, and State with instant profit calculations and contribution waterfall bars.
- **Top 10 Outliers & Residuals Diagnostic Table**: Detailed residual analysis for out-of-sample test cases.

---

## 📊 Core Empirical Findings

### 1. The 2-Feature Sweet Spot ($k=2$)
Across all 5 feature selection algorithms, the model consistently converges on `['R&D Spend', 'Marketing Spend']` as the optimal subset:
- **Minimum Test RMSE**: **$8,198.80** (lowest error across all tested models)
- **Peak Test $R^2$**: **0.9474** (explains 94.74% of startup profit variation)
- **R&D Dominance**: Alone, R&D spend accounts for over 94.6% of profit variation ($k=1$: $R^2=0.9465$, $\text{RMSE} = \$8,274.87$). Adding Marketing Spend provides complementary positive predictive lift.

### 2. The 5-Feature Overfitting Cliff ($k=5$)
Adding redundant variables (Administration and location dummies `State_Florida`, `State_New York`) introduces noise and multicollinearity:
- **Test RMSE Spikes**: from **$8,198.80** to **$9,137.99** (a **+$939 penalty!**)
- **$R^2$ Degrades**: drops to **0.9347**
- **Principle of Parsimony (Occam's Razor)**: The simpler 2-variable model outperforms the full 5-variable model in predictive accuracy, interpretability, and generalizability.

---

## 🔬 Comparison of the 5 Feature Selection Algorithms

All 5 methods were benchmarked across $k=1\dots 5$ features on the out-of-sample test set:

| Feature Selection Algorithm | $k=1$ (RMSE) | $k=2$ (RMSE) | $k=3$ (RMSE) | $k=4$ (RMSE) | $k=5$ (RMSE) | Best Feature Subset |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Lasso (L1 Ranking)** | $8,274.87 | **$8,198.80** ★ | $8,309.06 | $8,409.92 | $9,137.99 | `['R&D Spend', 'Marketing Spend']` |
| **Forward SFS** | $8,274.87 | **$8,198.80** ★ | $8,803.78 | $8,997.20 | $9,137.99 | `['R&D Spend', 'Marketing Spend']` |
| **RFE** | $8,274.87 | **$8,198.80** ★ | $8,803.78 | $9,068.54 | $9,137.99 | `['R&D Spend', 'Marketing Spend']` |
| **ElasticNet Ranking** | $8,274.87 | **$8,198.80** ★ | $8,803.78 | $8,997.20 | $9,137.99 | `['R&D Spend', 'Marketing Spend']` |
| **Exhaustive (Best Subset)** | $8,274.87 | **$8,198.80** ★ | $8,171.65 | $8,178.37 | $9,137.99 | `['R&D Spend', 'Admin', 'State_FL']` |

---

## 📁 Repository Structure

```
├── dashboard_demo.png                  # Dashboard screenshot in README
├── app.py                              # Flask Web Application (WSGI entrypoint & REST API)
├── api/
│   └── index.py                        # Vercel Serverless Function entry point
├── templates/
│   └── index.html                      # StatsUnlock-style Jinja2 web template
├── static/
│   └── model_data.js                   # Static data asset for instant frontend fallback
├── model_data.json                     # Comprehensive model statistics, weights, and tests
├── vercel.json                         # Vercel deployment configuration
├── requirements.txt                    # Project dependencies (Flask, NumPy, Pandas, etc.)
├── main.py                             # CRISP-DM CLI multiple linear regression script
├── feature_progression_analysis.py     # Script generating k=1..5 metrics and charts
├── all_5_algorithms_comparison_plot.png# Comparison chart of all 5 algorithms
└── README.md                           # Documentation and project manual
```

---

## ☁️ Deploying to Vercel

This repository is configured for zero-config Vercel Python deployment:

1. Push changes to GitHub:
   ```bash
   git add .
   git commit -m "feat: complete Flask MLR web app"
   git push origin main
   ```
2. In [Vercel](https://vercel.com/new), select **Import Git Repository**.
3. Click **Deploy**. Vercel will build the serverless Flask app using `api/index.py` and route all requests through `vercel.json`.

---

## 🙏 Acknowledgments & References

- **Design & Layout Reference**: [StatsUnlock Multiple Linear Regression Calculator](https://statsunlock.com/multiple-linear-regression-calculator/)
- **Dataset**: [50 Startups Dataset (Avik Jain 100-Days-Of-ML-Code)](https://raw.githubusercontent.com/Avik-Jain/100-Days-Of-ML-Code/master/datasets/50_Startups.csv)
- **Core Libraries**: Flask, Scikit-learn, Statsmodels, SciPy, NumPy, Pandas, Chart.js
