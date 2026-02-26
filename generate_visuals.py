import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, classification_report, confusion_matrix
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')
import os

# ── Palette ────────────────────────────────────────────────────────────────────
PALETTE = {
    'primary':   '#2D6A9F',
    'secondary': '#4DBFA8',
    'accent':    '#F4A261',
    'danger':    '#E76F51',
    'neutral':   '#8D99AE',
    'bg':        '#F8F9FA',
    'dark':      '#2B2D42',
}
CMAP_SEQ  = 'Blues'
CMAP_DIV  = 'coolwarm'

sns.set_theme(style='whitegrid', font_scale=1.1)
plt.rcParams.update({
    'figure.facecolor': PALETTE['bg'],
    'axes.facecolor':   PALETTE['bg'],
    'font.family':      'DejaVu Sans',
    'axes.spines.top':  False,
    'axes.spines.right':False,
})

OUTPUT = '/home/claude/fitmetrics/visuals'
os.makedirs(OUTPUT, exist_ok=True)

# ── Load & prep ────────────────────────────────────────────────────────────────
df = pd.read_csv('/home/claude/fitmetrics/data/gym_members_exercise_tracking.csv')
df.columns = [c.strip() for c in df.columns]

le = LabelEncoder()
df['Gender_enc']       = le.fit_transform(df['Gender'])
df['WorkoutType_enc']  = le.fit_transform(df['Workout_Type'])

EXP_LABELS = {1: 'Beginner', 2: 'Intermediate', 3: 'Expert'}
df['Exp_Label'] = df['Experience_Level'].map(EXP_LABELS)
WORKOUT_COLORS = {
    'Cardio':   PALETTE['primary'],
    'HIIT':     PALETTE['danger'],
    'Strength': PALETTE['secondary'],
    'Yoga':     PALETTE['accent'],
}
EXP_COLORS = ['#AED6F1', '#2D86D9', '#1A3A6B']

print("Data loaded:", df.shape)

# ═══════════════════════════════════════════════════════════════════════════════
# VIZ 1 — Calories Burned by Workout Type (Violin + Strip)
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor(PALETTE['bg'])
order = df.groupby('Workout_Type')['Calories_Burned'].median().sort_values(ascending=False).index.tolist()
colors = [WORKOUT_COLORS[w] for w in order]
parts = ax.violinplot(
    [df[df['Workout_Type']==w]['Calories_Burned'].values for w in order],
    positions=range(len(order)), showmedians=True, showextrema=False
)
for i, (pc, c) in enumerate(zip(parts['bodies'], colors)):
    pc.set_facecolor(c); pc.set_alpha(0.6)
parts['cmedians'].set_color(PALETTE['dark']); parts['cmedians'].set_linewidth(2)
for i, w in enumerate(order):
    sub = df[df['Workout_Type']==w]['Calories_Burned']
    ax.scatter(np.random.normal(i, 0.06, len(sub)), sub, s=10,
               color=WORKOUT_COLORS[w], alpha=0.4, zorder=3)
ax.set_xticks(range(len(order))); ax.set_xticklabels(order, fontsize=12)
ax.set_ylabel('Calories Burned', fontsize=12); ax.set_xlabel('')
ax.set_title('Caloric Output by Workout Type', fontsize=15, fontweight='bold', color=PALETTE['dark'], pad=15)
patches = [mpatches.Patch(color=WORKOUT_COLORS[w], label=w) for w in order]
ax.legend(handles=patches, loc='upper right', framealpha=0.8)
plt.tight_layout()
plt.savefig(f'{OUTPUT}/01_calories_by_workout.png', dpi=150, bbox_inches='tight')
plt.close()
print("Viz 1 done")

# ═══════════════════════════════════════════════════════════════════════════════
# VIZ 2 — Session Duration vs Calories (by Experience Level)
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor(PALETTE['bg'])
for lvl, color, label in zip([1,2,3], EXP_COLORS, ['Beginner','Intermediate','Expert']):
    sub = df[df['Experience_Level']==lvl]
    ax.scatter(sub['Session_Duration (hours)'], sub['Calories_Burned'],
               c=color, s=50, alpha=0.7, label=label, edgecolors='white', linewidths=0.4)
    m, b = np.polyfit(sub['Session_Duration (hours)'], sub['Calories_Burned'], 1)
    xs = np.linspace(sub['Session_Duration (hours)'].min(), sub['Session_Duration (hours)'].max(), 100)
    ax.plot(xs, m*xs+b, color=color, linewidth=2, alpha=0.9)
ax.set_xlabel('Session Duration (hours)', fontsize=12)
ax.set_ylabel('Calories Burned', fontsize=12)
ax.set_title('Session Duration vs Calories Burned\nby Experience Level', fontsize=15, fontweight='bold', color=PALETTE['dark'])
ax.legend(title='Experience', framealpha=0.8)
plt.tight_layout()
plt.savefig(f'{OUTPUT}/02_duration_vs_calories.png', dpi=150, bbox_inches='tight')
plt.close()
print("Viz 2 done")

# ═══════════════════════════════════════════════════════════════════════════════
# VIZ 3 — Correlation Heatmap
# ═══════════════════════════════════════════════════════════════════════════════
num_cols = ['Age','Weight (kg)','Height (m)','Max_BPM','Avg_BPM','Resting_BPM',
            'Session_Duration (hours)','Calories_Burned','Fat_Percentage',
            'Water_Intake (liters)','Workout_Frequency (days/week)','BMI']
corr = df[num_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
fig, ax = plt.subplots(figsize=(12, 9))
fig.patch.set_facecolor(PALETTE['bg'])
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap=CMAP_DIV,
            center=0, vmin=-1, vmax=1, ax=ax, linewidths=0.5,
            annot_kws={'size':8}, cbar_kws={'shrink':0.8})
ax.set_title('Feature Correlation Matrix', fontsize=15, fontweight='bold', color=PALETTE['dark'], pad=15)
plt.tight_layout()
plt.savefig(f'{OUTPUT}/03_correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("Viz 3 done")

# ═══════════════════════════════════════════════════════════════════════════════
# VIZ 4 — BMI Distribution by Gender and Experience
# ═══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.patch.set_facecolor(PALETTE['bg'])
for ax2, gender, color in zip(axes, ['Male','Female'], [PALETTE['primary'], PALETTE['accent']]):
    sub = df[df['Gender']==gender]
    for lvl, ec, label in zip([1,2,3], EXP_COLORS, ['Beginner','Intermediate','Expert']):
        vals = sub[sub['Experience_Level']==lvl]['BMI']
        ax2.hist(vals, bins=15, alpha=0.65, color=ec, label=label, edgecolor='white')
    ax2.set_title(f'{gender} Members - BMI Distribution', fontsize=13, fontweight='bold', color=PALETTE['dark'])
    ax2.set_xlabel('BMI'); ax2.set_ylabel('Count')
    ax2.legend(title='Experience')
plt.suptitle('BMI Distribution by Gender and Experience Level', fontsize=15, fontweight='bold', color=PALETTE['dark'], y=1.02)
plt.tight_layout()
plt.savefig(f'{OUTPUT}/04_bmi_gender_experience.png', dpi=150, bbox_inches='tight')
plt.close()
print("Viz 4 done")

# ═══════════════════════════════════════════════════════════════════════════════
# VIZ 5 — Heart Rate Profile by Workout Type
# ═══════════════════════════════════════════════════════════════════════════════
hr_cols = ['Resting_BPM','Avg_BPM','Max_BPM']
hr_data = df.groupby('Workout_Type')[hr_cols].mean().reindex(index=['Yoga','Cardio','Strength','HIIT'])
fig, ax = plt.subplots(figsize=(11, 6))
fig.patch.set_facecolor(PALETTE['bg'])
x = np.arange(len(hr_data))
w = 0.25
bars_colors = [PALETTE['primary'], PALETTE['secondary'], PALETTE['danger']]
for i, (col, label, c) in enumerate(zip(hr_cols, ['Resting BPM','Avg BPM','Max BPM'], bars_colors)):
    bars = ax.bar(x + i*w, hr_data[col], w, label=label, color=c, alpha=0.85, edgecolor='white')
    for bar in bars:
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                f'{bar.get_height():.0f}', ha='center', va='bottom', fontsize=8)
ax.set_xticks(x + w); ax.set_xticklabels(hr_data.index, fontsize=12)
ax.set_ylabel('BPM'); ax.set_title('Heart Rate Profile by Workout Type', fontsize=15, fontweight='bold', color=PALETTE['dark'])
ax.legend(framealpha=0.8)
plt.tight_layout()
plt.savefig(f'{OUTPUT}/05_heart_rate_profile.png', dpi=150, bbox_inches='tight')
plt.close()
print("Viz 5 done")

# ═══════════════════════════════════════════════════════════════════════════════
# VIZ 6 — Fat % vs Water Intake (bubble = calories)
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 7))
fig.patch.set_facecolor(PALETTE['bg'])
sc = ax.scatter(df['Water_Intake (liters)'], df['Fat_Percentage'],
                c=df['Calories_Burned'], s=df['Calories_Burned']/8,
                cmap='Blues', alpha=0.6, edgecolors='white', linewidths=0.3)
cbar = plt.colorbar(sc, ax=ax)
cbar.set_label('Calories Burned', fontsize=11)
ax.set_xlabel('Daily Water Intake (liters)', fontsize=12)
ax.set_ylabel('Body Fat Percentage (%)', fontsize=12)
ax.set_title('Hydration vs Body Fat\n(Bubble size = Calories Burned)', fontsize=15, fontweight='bold', color=PALETTE['dark'])
plt.tight_layout()
plt.savefig(f'{OUTPUT}/06_hydration_vs_fat.png', dpi=150, bbox_inches='tight')
plt.close()
print("Viz 6 done")

# ═══════════════════════════════════════════════════════════════════════════════
# VIZ 7 — Member Segmentation (KMeans Clustering)
# ═══════════════════════════════════════════════════════════════════════════════
features_cluster = ['Age','BMI','Fat_Percentage','Session_Duration (hours)',
                    'Calories_Burned','Workout_Frequency (days/week)']
X_c = df[features_cluster].dropna()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_c)
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_scaled)
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
cluster_names = {0:'High Performers', 1:'Casual Members', 2:'Intensive Trainers', 3:'Balanced Athletes'}
cluster_colors = [PALETTE['primary'], PALETTE['accent'], PALETTE['danger'], PALETTE['secondary']]
fig, ax = plt.subplots(figsize=(10, 7))
fig.patch.set_facecolor(PALETTE['bg'])
for cid, (name, color) in enumerate(zip(cluster_names.values(), cluster_colors)):
    mask2 = labels == cid
    ax.scatter(X_pca[mask2, 0], X_pca[mask2, 1], c=color, s=60,
               alpha=0.7, label=name, edgecolors='white', linewidths=0.3)
centers_pca = pca.transform(kmeans.cluster_centers_)
ax.scatter(centers_pca[:,0], centers_pca[:,1], c=PALETTE['dark'], s=200,
           marker='X', zorder=5, label='Centroids')
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)', fontsize=11)
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)', fontsize=11)
ax.set_title('Member Segmentation via K-Means Clustering\n(PCA Projection)', fontsize=15, fontweight='bold', color=PALETTE['dark'])
ax.legend(framealpha=0.8)
plt.tight_layout()
plt.savefig(f'{OUTPUT}/07_member_segmentation.png', dpi=150, bbox_inches='tight')
plt.close()

# Save cluster profile
cluster_df = X_c.copy()
cluster_df['Cluster'] = labels
cluster_df['Cluster_Name'] = [cluster_names[c] for c in labels]
cluster_profile = cluster_df.groupby('Cluster_Name')[features_cluster].mean().round(2)
cluster_profile.to_csv('/home/claude/fitmetrics/data/cluster_profiles.csv')
print("Viz 7 done")

# ═══════════════════════════════════════════════════════════════════════════════
# VIZ 8 — Regression: Feature Importance for Calories Burned
# ═══════════════════════════════════════════════════════════════════════════════
feat_reg = ['Age','Weight (kg)','Height (m)','Max_BPM','Avg_BPM','Resting_BPM',
            'Session_Duration (hours)','Fat_Percentage','Water_Intake (liters)',
            'Workout_Frequency (days/week)','BMI','Gender_enc','WorkoutType_enc','Experience_Level']
X_r = df[feat_reg]; y_r = df['Calories_Burned']
X_tr, X_te, y_tr, y_te = train_test_split(X_r, y_r, test_size=0.2, random_state=42)
rf_reg = RandomForestRegressor(n_estimators=200, random_state=42)
rf_reg.fit(X_tr, y_tr)
y_pred = rf_reg.predict(X_te)
r2 = r2_score(y_te, y_pred)
mae = mean_absolute_error(y_te, y_pred)
print(f"Regression R2: {r2:.4f}, MAE: {mae:.2f}")

imp = pd.Series(rf_reg.feature_importances_, index=feat_reg).sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(10, 7))
fig.patch.set_facecolor(PALETTE['bg'])
colors_bar = [PALETTE['danger'] if v > imp.median() else PALETTE['primary'] for v in imp.values]
ax.barh(imp.index, imp.values, color=colors_bar, edgecolor='white', alpha=0.85)
ax.set_xlabel('Feature Importance', fontsize=12)
ax.set_title(f'Feature Importance: Calories Burned Prediction\nR² = {r2:.3f}  |  MAE = {mae:.1f} kcal', 
             fontsize=14, fontweight='bold', color=PALETTE['dark'])
high_patch = mpatches.Patch(color=PALETTE['danger'], label='High Impact')
low_patch  = mpatches.Patch(color=PALETTE['primary'], label='Lower Impact')
ax.legend(handles=[high_patch, low_patch], framealpha=0.8)
plt.tight_layout()
plt.savefig(f'{OUTPUT}/08_calorie_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("Viz 8 done")

# ═══════════════════════════════════════════════════════════════════════════════
# VIZ 9 — Classification: Experience Level Confusion Matrix
# ═══════════════════════════════════════════════════════════════════════════════
feat_cls = ['Age','Weight (kg)','BMI','Fat_Percentage','Calories_Burned',
            'Session_Duration (hours)','Workout_Frequency (days/week)',
            'Max_BPM','Avg_BPM','Resting_BPM','Water_Intake (liters)','Gender_enc','WorkoutType_enc']
X_cl = df[feat_cls]; y_cl = df['Experience_Level']
X_tr2, X_te2, y_tr2, y_te2 = train_test_split(X_cl, y_cl, test_size=0.2, random_state=42, stratify=y_cl)
rf_cls = RandomForestClassifier(n_estimators=200, random_state=42)
rf_cls.fit(X_tr2, y_tr2)
y_pred2 = rf_cls.predict(X_te2)
acc = (y_pred2 == y_te2).mean()
print(f"Classification Accuracy: {acc:.4f}")
cm = confusion_matrix(y_te2, y_pred2)
fig, ax = plt.subplots(figsize=(7, 6))
fig.patch.set_facecolor(PALETTE['bg'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['Beginner','Intermediate','Expert'],
            yticklabels=['Beginner','Intermediate','Expert'],
            linewidths=0.5)
ax.set_ylabel('Actual', fontsize=12); ax.set_xlabel('Predicted', fontsize=12)
ax.set_title(f'Experience Level Classification\nAccuracy = {acc*100:.1f}%', fontsize=14, fontweight='bold', color=PALETTE['dark'])
plt.tight_layout()
plt.savefig(f'{OUTPUT}/09_experience_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print("Viz 9 done")

# ═══════════════════════════════════════════════════════════════════════════════
# VIZ 10 — Workout Frequency vs Fat % (box plot grid)
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(11, 6))
fig.patch.set_facecolor(PALETTE['bg'])
bp = ax.boxplot(
    [df[df['Workout_Frequency (days/week)']==f]['Fat_Percentage'].values for f in sorted(df['Workout_Frequency (days/week)'].unique())],
    patch_artist=True, notch=False, widths=0.5,
    medianprops=dict(color=PALETTE['dark'], linewidth=2)
)
freqs = sorted(df['Workout_Frequency (days/week)'].unique())
cmap_vals = plt.cm.Blues(np.linspace(0.3, 0.85, len(freqs)))
for patch, color in zip(bp['boxes'], cmap_vals):
    patch.set_facecolor(color); patch.set_alpha(0.8)
ax.set_xticklabels([f'{int(f)} days/wk' for f in freqs], fontsize=11)
ax.set_ylabel('Body Fat Percentage (%)', fontsize=12)
ax.set_title('Workout Frequency Impact on Body Fat Percentage', fontsize=14, fontweight='bold', color=PALETTE['dark'])
plt.tight_layout()
plt.savefig(f'{OUTPUT}/10_frequency_vs_fat.png', dpi=150, bbox_inches='tight')
plt.close()
print("Viz 10 done")

# save model metrics
import json
metrics = {
    'regression': {'r2': round(r2, 4), 'mae': round(mae, 2)},
    'classification': {'accuracy': round(acc, 4)},
    'clustering': {'n_clusters': 4, 'explained_variance': round(sum(pca.explained_variance_ratio_)*100, 1)}
}
with open('/home/claude/fitmetrics/data/model_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

print("\nAll visuals generated!")
