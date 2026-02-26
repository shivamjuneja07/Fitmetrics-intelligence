# FitMetrics Intelligence
### Gym Analytics and Predictive Modeling Platform

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4+-orange?logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![CI](https://github.com/yourusername/fitmetrics-intelligence/actions/workflows/ci.yml/badge.svg)

A senior-level end-to-end data science project combining business analytics with machine learning on gym member fitness data. This project covers exploratory data analysis, three predictive models, member segmentation, an interactive dashboard, and a professional PDF report.

---

## Project Overview

This project analyzes 973 gym member records to uncover behavioral patterns, predict caloric output, classify experience levels, and segment members into actionable personas. The analysis is designed to be directly applicable to gym operators, fitness product teams, and health analytics professionals.

**Three ML Models:**
- Random Forest Regression for calorie burn prediction (R-squared = 0.972)
- Random Forest Classification for experience level (Accuracy = 91.8%)
- K-Means Clustering for member segmentation (4 behavioral personas)

---

## Repository Structure

```
fitmetrics-intelligence/
│
├── data/
│   ├── gym_members_exercise_tracking.csv   # Raw dataset (973 members)
│   └── cluster_profiles.csv               # Generated cluster summary
│
├── notebooks/
│   └── fitmetrics_full_analysis.ipynb     # End-to-end Jupyter analysis
│
├── scripts/
│   ├── generate_visuals.py                # EDA and model visualizations
│   ├── generate_pdf.py                    # PDF report generation
│   └── streamlit_dashboard.py            # Interactive Streamlit dashboard
│
├── visuals/                               # All generated charts (PNG)
│
├── dashboard/                             # Dashboard assets
│
├── .github/
│   └── workflows/
│       └── ci.yml                         # GitHub Actions CI pipeline
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Key Insights

| Finding | Detail |
|---|---|
| Top calorie driver | Session Duration (most predictive feature) |
| Highest calorie workout | HIIT and Cardio consistently outperform |
| Fat reduction pattern | Each additional training day reduces avg fat by ~1.8% |
| Hydration correlation | Higher water intake linked to lower body fat |
| Expert vs Beginner calorie gap | Experts burn ~38% more calories per session |

---

## Model Performance

| Model | Algorithm | Metric | Score |
|---|---|---|---|
| Calorie Prediction | Random Forest Regressor | R-squared | 0.972 |
| Calorie Prediction | Random Forest Regressor | MAE | 36.2 kcal |
| Experience Classification | Random Forest Classifier | Accuracy | 91.8% |
| Member Segmentation | K-Means (k=4) | PCA Variance Explained | 51.4% |

---

## Member Personas (Clustering Output)

| Segment | Profile | Business Action |
|---|---|---|
| High Performers | Long sessions, low fat, high calories | Premium tier upsell |
| Casual Members | Low frequency, shorter sessions | Retention campaigns |
| Intensive Trainers | Very high BPM, short sessions | HIIT class bundles |
| Balanced Athletes | Consistent across all metrics | Loyalty rewards |

---

## Visuals

### Caloric Output by Workout Type
![Calories by Workout]([visuals/01_calories_by_workout.png](https://github.com/shivamjuneja07/Fitmetrics-intelligence/blob/main/01_calories_by_workout.png))

### Session Duration vs Calories (by Experience Level)
![Duration vs Calories](visuals/02_duration_vs_calories.png)

### Feature Correlation Matrix
![Correlation](visuals/03_correlation_heatmap.png)

### Member Segmentation via K-Means
![Segmentation](visuals/07_member_segmentation.png)

### Feature Importance: Calorie Prediction
![Feature Importance](visuals/08_calorie_feature_importance.png)

### Experience Level Classification Confusion Matrix
![Confusion Matrix](visuals/09_experience_confusion_matrix.png)

---

## Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/fitmetrics-intelligence.git
cd fitmetrics-intelligence
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Generate all visuals and models
```bash
python scripts/generate_visuals.py
```

### 4. Launch the interactive dashboard
```bash
streamlit run scripts/streamlit_dashboard.py
```

### 5. Open the Jupyter notebook
```bash
jupyter notebook notebooks/fitmetrics_full_analysis.ipynb
```

---

## Tech Stack

- **Data Processing:** pandas, numpy
- **Visualization:** matplotlib, seaborn
- **Machine Learning:** scikit-learn
- **Dashboard:** Streamlit, Plotly
- **Reporting:** ReportLab
- **CI/CD:** GitHub Actions

---

## Author

**Shivam**
Business Analyst and Data Scientist

---

## License

This project is licensed under the MIT License.
