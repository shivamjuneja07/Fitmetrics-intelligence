"""
dashboard/app.py
----------------
FitMetrics Intelligence - Interactive Streamlit Dashboard

Run with:
    streamlit run dashboard/app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
from preprocess import run_pipeline

# ---- Page config ----
st.set_page_config(
    page_title="FitMetrics Intelligence",
    page_icon="🏋️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---- Custom CSS ----
st.markdown("""
<style>
    .main { background-color: #F9FBFC; }
    .metric-card {
        background-color: white;
        border-left: 4px solid #2C4A6E;
        padding: 16px;
        border-radius: 8px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.06);
        margin-bottom: 12px;
    }
    h1 { color: #2C4A6E; }
    h2, h3 { color: #2C4A6E; }
    .stSelectbox label { font-weight: 600; color: #2C4A6E; }
</style>
""", unsafe_allow_html=True)

PALETTE = ['#2C4A6E', '#4A7BA7', '#6BA3BE', '#A8C5DA', '#D4E6F1']


@st.cache_data
def load_data():
    return run_pipeline()


df = load_data()

# ---- Sidebar ----
st.sidebar.image("https://img.icons8.com/color/96/gym.png", width=60)
st.sidebar.title("FitMetrics Intelligence")
st.sidebar.markdown("---")

gender_filter = st.sidebar.multiselect("Gender", options=df['Gender'].unique(), default=list(df['Gender'].unique()))
workout_filter = st.sidebar.multiselect("Workout Type", options=df['Workout_Type'].unique(),
                                         default=list(df['Workout_Type'].unique()))
exp_filter = st.sidebar.multiselect("Experience Level", options=['Beginner', 'Intermediate', 'Expert'],
                                     default=['Beginner', 'Intermediate', 'Expert'])
age_range = st.sidebar.slider("Age Range", int(df['Age'].min()), int(df['Age'].max()),
                               (int(df['Age'].min()), int(df['Age'].max())))

exp_map = {'Beginner': 1, 'Intermediate': 2, 'Expert': 3}
exp_levels = [exp_map[e] for e in exp_filter]

filtered = df[
    (df['Gender'].isin(gender_filter)) &
    (df['Workout_Type'].isin(workout_filter)) &
    (df['Experience_Level'].isin(exp_levels)) &
    (df['Age'].between(*age_range))
]

st.sidebar.markdown(f"**{len(filtered):,}** members selected")

# ---- Header ----
st.title("🏋️ FitMetrics Intelligence")
st.markdown("*Gym Analytics and Predictive Modeling Platform*")
st.markdown("---")

# ---- KPI Row ----
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Total Members", f"{len(filtered):,}")
with col2:
    st.metric("Avg Calories/Session", f"{filtered['Calories_Burned'].mean():.0f} kcal")
with col3:
    st.metric("Avg Session Duration", f"{filtered['Session_Duration (hours)'].mean():.2f} hrs")
with col4:
    st.metric("Avg BMI", f"{filtered['BMI'].mean():.1f}")
with col5:
    st.metric("Avg Workout Frequency", f"{filtered['Workout_Frequency (days/week)'].mean():.1f} days/wk")

st.markdown("---")

# ---- Tab Navigation ----
tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Performance", "Health Metrics", "Predictive Insights"])

with tab1:
    col_a, col_b = st.columns(2)

    with col_a:
        fig = px.histogram(filtered, x='Age', nbins=20, color_discrete_sequence=[PALETTE[1]],
                           title='Age Distribution')
        fig.update_layout(bargap=0.05, plot_bgcolor='#F9FBFC')
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        wt_counts = filtered['Workout_Type'].value_counts().reset_index()
        wt_counts.columns = ['Workout_Type', 'Count']
        fig = px.bar(wt_counts, x='Workout_Type', y='Count', color='Workout_Type',
                     color_discrete_sequence=PALETTE, title='Workout Type Distribution')
        fig.update_layout(showlegend=False, plot_bgcolor='#F9FBFC')
        st.plotly_chart(fig, use_container_width=True)

    col_c, col_d = st.columns(2)

    with col_c:
        el_counts = filtered['Experience_Label'].value_counts().reset_index()
        el_counts.columns = ['Experience', 'Count']
        fig = px.pie(el_counts, names='Experience', values='Count',
                     color_discrete_sequence=PALETTE, title='Experience Level Breakdown',
                     hole=0.4)
        st.plotly_chart(fig, use_container_width=True)

    with col_d:
        gender_wt = filtered.groupby(['Workout_Type', 'Gender']).size().reset_index(name='Count')
        fig = px.bar(gender_wt, x='Workout_Type', y='Count', color='Gender',
                     barmode='group', title='Gender Split by Workout Type',
                     color_discrete_sequence=[PALETTE[0], PALETTE[2]])
        fig.update_layout(plot_bgcolor='#F9FBFC')
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    col_a, col_b = st.columns(2)

    with col_a:
        fig = px.violin(filtered, x='Workout_Type', y='Calories_Burned', color='Workout_Type',
                        box=True, color_discrete_sequence=PALETTE,
                        title='Calories Burned by Workout Type')
        fig.update_layout(showlegend=False, plot_bgcolor='#F9FBFC')
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        fig = px.scatter(filtered, x='Session_Duration (hours)', y='Calories_Burned',
                         color='Experience_Label', opacity=0.6,
                         color_discrete_map={'Beginner': PALETTE[3], 'Intermediate': PALETTE[1], 'Expert': PALETTE[0]},
                         title='Session Duration vs Calories Burned')
        fig.update_layout(plot_bgcolor='#F9FBFC')
        st.plotly_chart(fig, use_container_width=True)

    hr_data = filtered.groupby('Experience_Label')[['Resting_BPM', 'Avg_BPM', 'Max_BPM']].mean().reset_index()
    fig = go.Figure()
    for col, color in zip(['Resting_BPM', 'Avg_BPM', 'Max_BPM'], [PALETTE[3], PALETTE[1], PALETTE[0]]):
        fig.add_trace(go.Bar(x=hr_data['Experience_Label'], y=hr_data[col], name=col,
                             marker_color=color))
    fig.update_layout(barmode='group', title='Heart Rate Profile by Experience Level',
                      plot_bgcolor='#F9FBFC', yaxis_title='BPM')
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    col_a, col_b = st.columns(2)

    with col_a:
        fig = px.histogram(filtered, x='BMI', color='Gender', nbins=30, barmode='overlay',
                           opacity=0.7, color_discrete_sequence=[PALETTE[0], PALETTE[2]],
                           title='BMI Distribution by Gender')
        fig.add_vline(x=18.5, line_dash='dot', line_color='gray', annotation_text='Underweight')
        fig.add_vline(x=25, line_dash='dash', line_color='gray', annotation_text='Overweight')
        fig.add_vline(x=30, line_dash='dashdot', line_color='gray', annotation_text='Obese')
        fig.update_layout(plot_bgcolor='#F9FBFC')
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        fig = px.scatter(filtered, x='BMI', y='Fat_Percentage', color='Experience_Label',
                         opacity=0.6,
                         color_discrete_map={'Beginner': PALETTE[3], 'Intermediate': PALETTE[1], 'Expert': PALETTE[0]},
                         title='BMI vs Body Fat Percentage')
        fig.update_layout(plot_bgcolor='#F9FBFC')
        st.plotly_chart(fig, use_container_width=True)

    fig = px.box(filtered, x='BMI_Category', y='Calories_Burned', color='BMI_Category',
                 color_discrete_sequence=PALETTE,
                 category_orders={'BMI_Category': ['Underweight', 'Normal', 'Overweight', 'Obese']},
                 title='Calorie Burn Distribution by BMI Category')
    fig.update_layout(showlegend=False, plot_bgcolor='#F9FBFC')
    st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.subheader("Calorie Burn Estimator")
    st.markdown("Use the sliders below to estimate how many calories a member would burn in a session.")

    c1, c2, c3 = st.columns(3)
    with c1:
        age_in = st.slider("Age", 18, 59, 35)
        weight_in = st.slider("Weight (kg)", 40.0, 130.0, 75.0)
        height_in = st.slider("Height (m)", 1.50, 2.10, 1.75)
    with c2:
        duration_in = st.slider("Session Duration (hours)", 0.5, 3.0, 1.0)
        freq_in = st.slider("Workout Frequency (days/week)", 1, 7, 4)
        exp_in = st.selectbox("Experience Level", ['Beginner', 'Intermediate', 'Expert'])
    with c3:
        workout_in = st.selectbox("Workout Type", ['Cardio', 'Strength', 'Yoga', 'HIIT'])
        avg_bpm_in = st.slider("Avg BPM", 100, 180, 140)
        water_in = st.slider("Water Intake (liters)", 1.5, 4.0, 2.5)

    exp_num = {'Beginner': 1, 'Intermediate': 2, 'Expert': 3}[exp_in]
    bmi_calc = weight_in / (height_in ** 2)
    calories_simple = (duration_in * freq_in * avg_bpm_in * 0.5 * exp_num)

    st.markdown(f"""
    <div class='metric-card'>
        <h3>Estimated Calories: <span style='color:#2C4A6E'>{calories_simple:.0f} kcal</span></h3>
        <p>Calculated BMI: <strong>{bmi_calc:.1f}</strong> | Experience: <strong>{exp_in}</strong></p>
        <p><em>Note: This is a simplified estimate. Use the trained model (models.py) for full ML-based predictions.</em></p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.markdown("<p style='text-align:center; color:#6B7280; font-size:13px'>FitMetrics Intelligence | Built by Shivam</p>",
            unsafe_allow_html=True)
