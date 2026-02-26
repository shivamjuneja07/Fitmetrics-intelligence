"""
preprocess.py
-------------
FitMetrics Intelligence - Data Preprocessing Module

Loads raw gym member data, validates schema, engineers features,
and outputs a cleaned dataset ready for EDA and modeling.
"""

import pandas as pd
import numpy as np
import os

RAW_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'gym_members_exercise_tracking.csv')
CLEAN_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'gym_members_clean.csv')

REQUIRED_COLS = [
    'Age', 'Gender', 'Weight (kg)', 'Height (m)', 'Max_BPM', 'Avg_BPM',
    'Resting_BPM', 'Session_Duration (hours)', 'Calories_Burned',
    'Workout_Type', 'Fat_Percentage', 'Water_Intake (liters)',
    'Workout_Frequency (days/week)', 'Experience_Level', 'BMI'
]


def load_data(path: str = RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    print(f"[preprocess] Loaded {len(df):,} rows x {len(df.columns)} columns")
    return df


def validate(df: pd.DataFrame) -> pd.DataFrame:
    initial = len(df)
    df = df.dropna(subset=REQUIRED_COLS)
    df = df[df['Age'].between(10, 100)]
    df = df[df['BMI'].between(10, 60)]
    df = df[df['Calories_Burned'] > 0]
    df = df[df['Session_Duration (hours)'] > 0]
    print(f"[preprocess] Validation: {initial - len(df)} rows removed | {len(df):,} rows retained")
    return df.reset_index(drop=True)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    # HR reserve (Karvonen method)
    df['HR_Reserve'] = df['Max_BPM'] - df['Resting_BPM']

    # Intensity index: avg BPM as % of max
    df['HR_Intensity_%'] = (df['Avg_BPM'] / df['Max_BPM'] * 100).round(2)

    # Calories per hour
    df['Calories_Per_Hour'] = (df['Calories_Burned'] / df['Session_Duration (hours)']).round(2)

    # Weekly caloric load
    df['Weekly_Caloric_Load'] = (df['Calories_Burned'] * df['Workout_Frequency (days/week)']).round(2)

    # BMI category
    def bmi_cat(bmi):
        if bmi < 18.5:
            return 'Underweight'
        elif bmi < 25:
            return 'Normal'
        elif bmi < 30:
            return 'Overweight'
        return 'Obese'

    df['BMI_Category'] = df['BMI'].apply(bmi_cat)

    # Experience label
    df['Experience_Label'] = df['Experience_Level'].map({1: 'Beginner', 2: 'Intermediate', 3: 'Expert'})

    # Encoded categoricals for modeling
    df['Gender_enc'] = (df['Gender'] == 'Male').astype(int)
    df['WorkoutType_enc'] = df['Workout_Type'].astype('category').cat.codes

    print(f"[preprocess] Feature engineering complete. Columns: {len(df.columns)}")
    return df


def save(df: pd.DataFrame, path: str = CLEAN_PATH) -> None:
    df.to_csv(path, index=False)
    print(f"[preprocess] Saved clean data to: {path}")


def run_pipeline() -> pd.DataFrame:
    df = load_data()
    df = validate(df)
    df = engineer_features(df)
    save(df)
    return df


if __name__ == '__main__':
    run_pipeline()
