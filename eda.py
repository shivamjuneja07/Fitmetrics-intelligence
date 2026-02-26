"""
eda.py
------
FitMetrics Intelligence - Exploratory Data Analysis Module

Runs the full EDA pipeline and saves all visualisations to visuals/.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from preprocess import run_pipeline

VISUALS_DIR = os.path.join(os.path.dirname(__file__), '..', 'visuals')
os.makedirs(VISUALS_DIR, exist_ok=True)

PALETTE = ['#2C4A6E', '#4A7BA7', '#6BA3BE', '#A8C5DA', '#D4E6F1']
ACCENT = '#2C4A6E'
sns.set_theme(style='whitegrid', font='DejaVu Sans')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = '#F9FBFC'


def plot_overview(df):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Member Overview: Demographics and Activity', fontsize=16, fontweight='bold', color=ACCENT)

    wt = df['Workout_Type'].value_counts()
    axes[0, 0].bar(wt.index, wt.values, color=PALETTE[:4], edgecolor='white', linewidth=1.5)
    axes[0, 0].set_title('Workout Type Distribution', fontweight='bold', color=ACCENT)
    axes[0, 0].set_ylabel('Number of Members')
    for i, v in enumerate(wt.values):
        axes[0, 0].text(i, v + 3, str(v), ha='center', fontsize=10, color=ACCENT)

    el = df['Experience_Label'].value_counts()
    axes[0, 1].pie(el.values, labels=el.index, colors=PALETTE[:3], autopct='%1.1f%%',
                   startangle=90, wedgeprops={'edgecolor': 'white', 'linewidth': 2})
    axes[0, 1].set_title('Experience Level Breakdown', fontweight='bold', color=ACCENT)

    axes[1, 0].hist(df['Age'], bins=20, color=PALETTE[1], edgecolor='white', linewidth=0.8)
    axes[1, 0].axvline(df['Age'].mean(), color=PALETTE[0], linestyle='--', linewidth=2,
                       label=f"Mean: {df['Age'].mean():.0f}")
    axes[1, 0].set_title('Age Distribution', fontweight='bold', color=ACCENT)
    axes[1, 0].set_xlabel('Age')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].legend()

    gender_wt = df.groupby(['Workout_Type', 'Gender']).size().unstack()
    gender_wt.plot(kind='bar', ax=axes[1, 1], color=[PALETTE[0], PALETTE[2]], edgecolor='white')
    axes[1, 1].set_title('Gender Split by Workout Type', fontweight='bold', color=ACCENT)
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].tick_params(axis='x', rotation=0)
    axes[1, 1].legend(title='Gender')

    plt.tight_layout()
    plt.savefig(os.path.join(VISUALS_DIR, '01_overview_dashboard.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("[eda] Saved 01_overview_dashboard.png")


def plot_calories_performance(df):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Calorie Burn and Performance Analysis', fontsize=16, fontweight='bold', color=ACCENT)

    order = df.groupby('Workout_Type')['Calories_Burned'].median().sort_values(ascending=False).index
    sns.violinplot(data=df, x='Workout_Type', y='Calories_Burned', order=order,
                   hue='Workout_Type', palette=PALETTE[:4], ax=axes[0, 0], inner='quartile', legend=False)
    axes[0, 0].set_title('Calories Burned by Workout Type', fontweight='bold', color=ACCENT)
    axes[0, 0].set_xlabel('')
    axes[0, 0].set_ylabel('Calories Burned')

    sc = axes[0, 1].scatter(df['Session_Duration (hours)'], df['Calories_Burned'],
                             c=df['Experience_Level'], cmap='Blues', alpha=0.6, s=30, edgecolors='none')
    axes[0, 1].set_title('Session Duration vs Calories Burned', fontweight='bold', color=ACCENT)
    axes[0, 1].set_xlabel('Session Duration (hours)')
    axes[0, 1].set_ylabel('Calories Burned')
    plt.colorbar(sc, ax=axes[0, 1], label='Experience Level')

    sns.boxplot(data=df, x='Experience_Label', y='Calories_Burned',
                order=['Beginner', 'Intermediate', 'Expert'],
                hue='Experience_Label', palette=PALETTE[:3], ax=axes[1, 0], legend=False)
    axes[1, 0].set_title('Calories by Experience Level', fontweight='bold', color=ACCENT)
    axes[1, 0].set_xlabel('')
    axes[1, 0].set_ylabel('Calories Burned')

    hr_data = df.groupby('Experience_Label')[['Resting_BPM', 'Avg_BPM', 'Max_BPM']].mean()
    hr_data = hr_data.loc[['Beginner', 'Intermediate', 'Expert']]
    x = np.arange(3)
    w = 0.25
    axes[1, 1].bar(x - w, hr_data['Resting_BPM'], w, label='Resting BPM', color=PALETTE[3])
    axes[1, 1].bar(x, hr_data['Avg_BPM'], w, label='Avg BPM', color=PALETTE[1])
    axes[1, 1].bar(x + w, hr_data['Max_BPM'], w, label='Max BPM', color=PALETTE[0])
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(['Beginner', 'Intermediate', 'Expert'])
    axes[1, 1].set_title('Heart Rate Profile by Experience Level', fontweight='bold', color=ACCENT)
    axes[1, 1].set_ylabel('BPM')
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(VISUALS_DIR, '02_calories_performance.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("[eda] Saved 02_calories_performance.png")


def plot_correlation_bmi(df):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Feature Correlations and BMI Analysis', fontsize=16, fontweight='bold', color=ACCENT)

    num_cols = ['Age', 'Weight (kg)', 'Height (m)', 'Max_BPM', 'Avg_BPM', 'Resting_BPM',
                'Session_Duration (hours)', 'Calories_Burned', 'Fat_Percentage',
                'Water_Intake (liters)', 'Workout_Frequency (days/week)', 'Experience_Level', 'BMI']
    corr = df[num_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, cmap='Blues', annot=True, fmt='.2f',
                linewidths=0.5, ax=axes[0], cbar_kws={'shrink': 0.8}, annot_kws={'size': 7})
    axes[0].set_title('Correlation Matrix', fontweight='bold', color=ACCENT)
    axes[0].tick_params(axis='x', rotation=45, labelsize=8)
    axes[0].tick_params(axis='y', labelsize=8)

    for gender, color in zip(['Male', 'Female'], [PALETTE[0], PALETTE[2]]):
        subset = df[df['Gender'] == gender]['BMI']
        axes[1].hist(subset, bins=25, alpha=0.6, color=color, label=gender, edgecolor='white')
    axes[1].set_title('BMI Distribution by Gender', fontweight='bold', color=ACCENT)
    axes[1].set_xlabel('BMI')
    axes[1].set_ylabel('Count')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(VISUALS_DIR, '03_correlation_bmi.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("[eda] Saved 03_correlation_bmi.png")


def print_summary(df):
    print("\n===== EDA SUMMARY =====")
    print(f"Total members: {len(df):,}")
    print(f"Avg calories burned per session: {df['Calories_Burned'].mean():.0f}")
    print(f"Avg session duration: {df['Session_Duration (hours)'].mean():.2f} hours")
    print(f"Avg BMI: {df['BMI'].mean():.2f}")
    print(f"Most common workout: {df['Workout_Type'].mode()[0]}")
    print("=======================\n")


def run_eda():
    df = run_pipeline()
    print_summary(df)
    plot_overview(df)
    plot_calories_performance(df)
    plot_correlation_bmi(df)
    print("[eda] All visuals saved.")
    return df


if __name__ == '__main__':
    run_eda()
