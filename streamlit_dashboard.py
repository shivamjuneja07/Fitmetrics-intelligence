"""
FitMetrics Intelligence - Interactive Streamlit Dashboard
Run with: streamlit run scripts/streamlit_dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# ── Config ────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FitMetrics Intelligence",
    page_icon="fitness_center",
    layout="wide",
    initial_sidebar_state="expanded"
)

PALETTE = {
    'primary':   '#2D6A9F',
    'secondary': '#4DBFA8',
    'accent':    '#F4A261',
    'danger':    '#E76F51',
    'neutral':   '#8D99AE',
}

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #F8F9FA; }
    .stMetric { background-color: white; border-radius: 10px; padding: 12px; box-shadow: 0 1px 4px rgba(0,0,0,0.08); }
    h1, h2, h3 { color: #2B2D42; }
    .highlight { background: #EDF2F7; border-left: 4px solid #2D6A9F; padding: 10px 14px; border-radius: 4px; margin: 8px 0; }
</style>
""", unsafe_allow_html=True)

# ── Load & prep ───────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv('data/gym_members_exercise_tracking.csv')
    df.columns = [c.strip() for c in df.columns]
    le = LabelEncoder()
    df['Gender_enc'] = le.fit_transform(df['Gender'])
    df['WorkoutType_enc'] = le.fit_transform(df['Workout_Type'])
    df['Exp_Label'] = df['Experience_Level'].map({1:'Beginner', 2:'Intermediate', 3:'Expert'})
    return df

@st.cache_resource
def train_models(df):
    feat_reg = ['Age','Weight (kg)','Height (m)','Max_BPM','Avg_BPM','Resting_BPM',
                'Session_Duration (hours)','Fat_Percentage','Water_Intake (liters)',
                'Workout_Frequency (days/week)','BMI','Gender_enc','WorkoutType_enc','Experience_Level']
    X_r = df[feat_reg]; y_r = df['Calories_Burned']
    X_tr, X_te, y_tr, y_te = train_test_split(X_r, y_r, test_size=0.2, random_state=42)
    rf_reg = RandomForestRegressor(n_estimators=200, random_state=42)
    rf_reg.fit(X_tr, y_tr)
    r2 = r2_score(y_te, rf_reg.predict(X_te))
    mae = mean_absolute_error(y_te, rf_reg.predict(X_te))

    feat_cls = ['Age','Weight (kg)','BMI','Fat_Percentage','Calories_Burned',
                'Session_Duration (hours)','Workout_Frequency (days/week)',
                'Max_BPM','Avg_BPM','Resting_BPM','Water_Intake (liters)','Gender_enc','WorkoutType_enc']
    X_cl = df[feat_cls]; y_cl = df['Experience_Level']
    X_tr2, X_te2, y_tr2, y_te2 = train_test_split(X_cl, y_cl, test_size=0.2, random_state=42, stratify=y_cl)
    rf_cls = RandomForestClassifier(n_estimators=200, random_state=42)
    rf_cls.fit(X_tr2, y_tr2)
    acc = (rf_cls.predict(X_te2) == y_te2).mean()

    return rf_reg, rf_cls, feat_reg, feat_cls, {'r2': r2, 'mae': mae, 'acc': acc}

df = load_data()
rf_reg, rf_cls, feat_reg, feat_cls, model_metrics = train_models(df)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### FitMetrics Intelligence")
    st.markdown("---")
    page = st.radio("Navigation", [
        "Overview",
        "EDA Explorer",
        "Calorie Predictor",
        "Experience Classifier",
        "Member Segmentation",
    ])
    st.markdown("---")
    st.markdown("**Dataset:** 973 gym members")
    st.markdown("**Models:** 3 ML models trained")
    st.markdown("**Author:** Shivam")

# ═══════════════════════════════════════════════════════════════════════════════
if page == "Overview":
    st.title("FitMetrics Intelligence")
    st.markdown("**Gym Analytics and Predictive Modeling Platform**")
    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Members", "973")
    c2.metric("Calorie Model R-Squared", f"{model_metrics['r2']:.3f}")
    c3.metric("Experience Accuracy", f"{model_metrics['acc']*100:.1f}%")
    c4.metric("Member Segments", "4")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Calories Burned by Workout Type")
        fig, ax = plt.subplots(figsize=(7, 4))
        colors = {'HIIT':'#E76F51','Cardio':'#2D6A9F','Strength':'#4DBFA8','Yoga':'#F4A261'}
        order = df.groupby('Workout_Type')['Calories_Burned'].median().sort_values(ascending=False).index
        for i, w in enumerate(order):
            vals = df[df['Workout_Type']==w]['Calories_Burned']
            ax.boxplot(vals, positions=[i], patch_artist=True,
                       boxprops=dict(facecolor=colors[w], alpha=0.7),
                       medianprops=dict(color='black', linewidth=2))
        ax.set_xticks(range(len(order))); ax.set_xticklabels(order)
        ax.set_ylabel('Calories Burned')
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        st.pyplot(fig); plt.close()

    with col2:
        st.subheader("Avg Calories by Experience Level")
        exp_avg = df.groupby('Exp_Label')['Calories_Burned'].mean().reindex(['Beginner','Intermediate','Expert'])
        fig, ax = plt.subplots(figsize=(7, 4))
        bars = ax.bar(exp_avg.index, exp_avg.values,
                      color=['#AED6F1','#2D86D9','#1A3A6B'], alpha=0.85, edgecolor='white')
        for bar in bars:
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+5,
                    f'{bar.get_height():.0f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax.set_ylabel('Avg Calories Burned')
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        st.pyplot(fig); plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
elif page == "EDA Explorer":
    st.title("Exploratory Data Analysis")
    st.markdown("Interactively explore the gym member dataset.")

    tab1, tab2, tab3 = st.tabs(["Distributions", "Relationships", "Correlation"])

    with tab1:
        col = st.selectbox("Select Feature", ['Calories_Burned','BMI','Fat_Percentage',
                                              'Session_Duration (hours)','Water_Intake (liters)','Age'])
        split = st.radio("Split by", ['None','Gender','Workout_Type','Exp_Label'], horizontal=True)
        fig, ax = plt.subplots(figsize=(10, 5))
        if split == 'None':
            ax.hist(df[col], bins=30, color=PALETTE['primary'], alpha=0.8, edgecolor='white')
        else:
            for val in df[split].unique():
                sub = df[df[split]==val][col]
                ax.hist(sub, bins=25, alpha=0.6, label=val, edgecolor='white')
            ax.legend()
        ax.set_xlabel(col); ax.set_ylabel('Count')
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        st.pyplot(fig); plt.close()

    with tab2:
        x_feat = st.selectbox("X Axis", ['Session_Duration (hours)','Water_Intake (liters)','Age','BMI','Workout_Frequency (days/week)'])
        y_feat = st.selectbox("Y Axis", ['Calories_Burned','Fat_Percentage','BMI','Avg_BPM'])
        color_by = st.radio("Color by", ['Workout_Type','Gender','Exp_Label'], horizontal=True)
        fig, ax = plt.subplots(figsize=(10, 6))
        palette = sns.color_palette("Blues_d", n_colors=df[color_by].nunique())
        for i, val in enumerate(df[color_by].unique()):
            sub = df[df[color_by]==val]
            ax.scatter(sub[x_feat], sub[y_feat], s=50, alpha=0.6, label=val,
                       color=palette[i], edgecolors='white', linewidths=0.3)
        ax.set_xlabel(x_feat); ax.set_ylabel(y_feat)
        ax.legend(title=color_by)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        st.pyplot(fig); plt.close()

    with tab3:
        num_cols = ['Age','Weight (kg)','Max_BPM','Avg_BPM','Resting_BPM',
                    'Session_Duration (hours)','Calories_Burned','Fat_Percentage',
                    'Water_Intake (liters)','Workout_Frequency (days/week)','BMI']
        fig, ax = plt.subplots(figsize=(11, 8))
        corr = df[num_cols].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
                    center=0, vmin=-1, vmax=1, ax=ax, linewidths=0.5, annot_kws={'size':8})
        ax.set_title('Feature Correlation Matrix')
        st.pyplot(fig); plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Calorie Predictor":
    st.title("Calorie Burn Predictor")
    st.markdown(f"Random Forest model trained on 80% of data. **R-squared = {model_metrics['r2']:.3f}** | **MAE = {model_metrics['mae']:.1f} kcal**")
    st.markdown("---")
    st.subheader("Enter Member Details")

    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.slider("Age", 18, 70, 30)
        weight = st.slider("Weight (kg)", 40.0, 130.0, 75.0)
        height = st.slider("Height (m)", 1.50, 2.00, 1.72)
        gender = st.selectbox("Gender", ["Male", "Female"])
    with col2:
        workout_type = st.selectbox("Workout Type", ["Cardio", "HIIT", "Strength", "Yoga"])
        session_dur = st.slider("Session Duration (hrs)", 0.5, 3.0, 1.2, 0.1)
        freq = st.slider("Workout Frequency (days/week)", 1, 7, 3)
        exp = st.selectbox("Experience Level", [1, 2, 3], format_func=lambda x: {1:'Beginner',2:'Intermediate',3:'Expert'}[x])
    with col3:
        max_bpm = st.slider("Max BPM", 120, 210, 170)
        avg_bpm = st.slider("Avg BPM", 100, 190, 145)
        rest_bpm = st.slider("Resting BPM", 40, 100, 62)
        fat_pct = st.slider("Fat Percentage (%)", 5.0, 45.0, 22.0)
        water = st.slider("Water Intake (liters)", 1.0, 4.0, 2.5, 0.1)

    bmi = weight / (height ** 2)
    le_g = LabelEncoder().fit(df['Gender'])
    le_w = LabelEncoder().fit(df['Workout_Type'])
    gender_enc = 1 if gender == 'Male' else 0
    wtype_enc = le_w.transform([workout_type])[0]

    input_data = pd.DataFrame([[age, weight, height, max_bpm, avg_bpm, rest_bpm,
                                 session_dur, fat_pct, water, freq, bmi, gender_enc, wtype_enc, exp]],
                               columns=feat_reg)
    pred = rf_reg.predict(input_data)[0]

    st.markdown("---")
    st.markdown(f"<h2 style='color:#2D6A9F; text-align:center;'>Predicted Calories Burned: <span style='color:#E76F51'>{pred:.0f} kcal</span></h2>", unsafe_allow_html=True)
    st.markdown(f"<p style='text-align:center; color:#8D99AE;'>BMI calculated: {bmi:.1f}</p>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Experience Classifier":
    st.title("Experience Level Classifier")
    st.markdown(f"Multi-class Random Forest. **Accuracy = {model_metrics['acc']*100:.1f}%**")
    st.markdown("---")
    st.subheader("Enter Member Metrics")

    col1, col2 = st.columns(2)
    with col1:
        age_c = st.slider("Age", 18, 70, 30, key='age_c')
        weight_c = st.slider("Weight (kg)", 40.0, 130.0, 75.0, key='wt_c')
        bmi_c = st.slider("BMI", 15.0, 45.0, 25.0, key='bmi_c')
        fat_c = st.slider("Fat Percentage (%)", 5.0, 45.0, 22.0, key='fat_c')
        cal_c = st.slider("Calories Burned (typical session)", 300, 2000, 900, key='cal_c')
    with col2:
        dur_c = st.slider("Session Duration (hrs)", 0.5, 3.0, 1.2, 0.1, key='dur_c')
        freq_c = st.slider("Workout Frequency (days/week)", 1, 7, 3, key='freq_c')
        maxbpm_c = st.slider("Max BPM", 120, 210, 170, key='maxbpm_c')
        avgbpm_c = st.slider("Avg BPM", 100, 190, 145, key='avgbpm_c')
        restbpm_c = st.slider("Resting BPM", 40, 100, 62, key='restbpm_c')
        water_c = st.slider("Water Intake (liters)", 1.0, 4.0, 2.5, 0.1, key='water_c')
        gender_c = st.selectbox("Gender", ["Male", "Female"], key='gen_c')
        wtype_c = st.selectbox("Workout Type", ["Cardio", "HIIT", "Strength", "Yoga"], key='wt2_c')

    le_w2 = LabelEncoder().fit(df['Workout_Type'])
    gender_enc_c = 1 if gender_c == 'Male' else 0
    wtype_enc_c = le_w2.transform([wtype_c])[0]

    input_cls = pd.DataFrame([[age_c, weight_c, bmi_c, fat_c, cal_c, dur_c, freq_c,
                                maxbpm_c, avgbpm_c, restbpm_c, water_c, gender_enc_c, wtype_enc_c]],
                              columns=feat_cls)
    proba = rf_cls.predict_proba(input_cls)[0]
    pred_cls = rf_cls.predict(input_cls)[0]
    labels = {1:'Beginner', 2:'Intermediate', 3:'Expert'}

    st.markdown("---")
    st.markdown(f"<h2 style='color:#2D6A9F; text-align:center;'>Predicted: <span style='color:#4DBFA8'>{labels[pred_cls]}</span></h2>", unsafe_allow_html=True)
    st.markdown("**Confidence per class:**")
    for cls_id, prob in zip(rf_cls.classes_, proba):
        st.progress(float(prob), text=f"{labels[cls_id]}: {prob*100:.1f}%")

# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Member Segmentation":
    st.title("Member Segmentation")
    st.markdown("K-Means clustering with 4 behavioral personas, projected via PCA.")
    st.markdown("---")

    features_cluster = ['Age','BMI','Fat_Percentage','Session_Duration (hours)',
                        'Calories_Burned','Workout_Frequency (days/week)']
    X_c = df[features_cluster].dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_c)
    from sklearn.decomposition import PCA
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    labels_k = kmeans.fit_predict(X_scaled)
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    cluster_names = {0:'High Performers', 1:'Casual Members', 2:'Intensive Trainers', 3:'Balanced Athletes'}
    cluster_colors = [PALETTE['primary'], PALETTE['accent'], PALETTE['danger'], PALETTE['secondary']]

    fig, ax = plt.subplots(figsize=(10, 7))
    for cid, (name, color) in enumerate(zip(cluster_names.values(), cluster_colors)):
        mask = labels_k == cid
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], c=color, s=60,
                   alpha=0.7, label=name, edgecolors='white', linewidths=0.3)
    centers_pca = pca.transform(kmeans.cluster_centers_)
    ax.scatter(centers_pca[:,0], centers_pca[:,1], c='#2B2D42', s=200,
               marker='X', zorder=5, label='Centroids')
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)')
    ax.legend(framealpha=0.8)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    st.pyplot(fig); plt.close()

    st.markdown("---")
    st.subheader("Cluster Profiles")
    cluster_df = X_c.copy()
    cluster_df['Segment'] = [cluster_names[c] for c in labels_k]
    profile = cluster_df.groupby('Segment')[features_cluster].mean().round(2)
    st.dataframe(profile.style.background_gradient(cmap='Blues', axis=0), use_container_width=True)
