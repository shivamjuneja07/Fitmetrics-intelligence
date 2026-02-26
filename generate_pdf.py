import json
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image,
                                Table, TableStyle, HRFlowable, PageBreak, KeepTogether)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
from reportlab.pdfgen import canvas
import os

# ── Colors ──────────────────────────────────────────────────────────────────────
C_PRIMARY    = colors.HexColor('#2D6A9F')
C_SECONDARY  = colors.HexColor('#4DBFA8')
C_ACCENT     = colors.HexColor('#F4A261')
C_DARK       = colors.HexColor('#2B2D42')
C_NEUTRAL    = colors.HexColor('#8D99AE')
C_BG         = colors.HexColor('#F8F9FA')
C_LIGHT      = colors.HexColor('#EDF2F7')
C_WHITE      = colors.white

VISUALS = '/home/claude/fitmetrics/visuals'
OUTPUT  = '/mnt/user-data/outputs/FitMetrics_Intelligence_Report.pdf'
os.makedirs('/mnt/user-data/outputs', exist_ok=True)

with open('/home/claude/fitmetrics/data/model_metrics.json') as f:
    metrics = json.load(f)

W, H = A4

# ── Page template with header/footer ────────────────────────────────────────────
class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_elements(num_pages)
            super().showPage()
        super().save()

    def draw_page_elements(self, page_count):
        page_num = self._saved_page_states.index(dict(self.__dict__)) if dict(self.__dict__) in self._saved_page_states else 0
        # top accent bar
        self.setFillColor(C_PRIMARY)
        self.rect(0, H - 0.6*cm, W, 0.6*cm, fill=1, stroke=0)
        # bottom bar
        self.setFillColor(C_LIGHT)
        self.rect(0, 0, W, 1.1*cm, fill=1, stroke=0)
        # footer text
        self.setFillColor(C_NEUTRAL)
        self.setFont('Helvetica', 8)
        self.drawString(2*cm, 0.38*cm, 'FitMetrics Intelligence  |  Gym Analytics and Predictive Modeling')
        self.drawRightString(W - 2*cm, 0.38*cm, f'Page {self._pageNumber} of {page_count}')

def make_doc(output_path):
    doc = SimpleDocTemplate(
        output_path, pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=1.8*cm, bottomMargin=1.8*cm,
        title='FitMetrics Intelligence Report',
        author='Shivam'
    )
    return doc

# ── Styles ───────────────────────────────────────────────────────────────────────
base = getSampleStyleSheet()

def S(name, **kw):
    s = ParagraphStyle(name, **kw)
    return s

sTitle       = S('sTitle', fontName='Helvetica-Bold', fontSize=30, textColor=C_WHITE,
                 alignment=TA_CENTER, spaceAfter=6, leading=38)
sSubtitle    = S('sSubtitle', fontName='Helvetica', fontSize=13, textColor=C_LIGHT,
                 alignment=TA_CENTER, spaceAfter=4, leading=18)
sH1          = S('sH1', fontName='Helvetica-Bold', fontSize=17, textColor=C_PRIMARY,
                 spaceBefore=14, spaceAfter=6, leading=22)
sH2          = S('sH2', fontName='Helvetica-Bold', fontSize=13, textColor=C_DARK,
                 spaceBefore=10, spaceAfter=4, leading=18)
sBody        = S('sBody', fontName='Helvetica', fontSize=10, textColor=C_DARK,
                 spaceAfter=6, leading=16, alignment=TA_JUSTIFY)
sCaption     = S('sCaption', fontName='Helvetica-Oblique', fontSize=9, textColor=C_NEUTRAL,
                 alignment=TA_CENTER, spaceAfter=10, leading=13)
sSignoff     = S('sSignoff', fontName='Helvetica-Bold', fontSize=11, textColor=C_DARK,
                 alignment=TA_LEFT, spaceAfter=2)
sBullet      = S('sBullet', fontName='Helvetica', fontSize=10, textColor=C_DARK,
                 spaceAfter=4, leading=15, leftIndent=14, bulletIndent=0)
sLabel       = S('sLabel', fontName='Helvetica-Bold', fontSize=9, textColor=C_WHITE,
                 alignment=TA_CENTER, leading=12)
sMetricVal   = S('sMetricVal', fontName='Helvetica-Bold', fontSize=22, textColor=C_PRIMARY,
                 alignment=TA_CENTER, leading=26)
sMetricLbl   = S('sMetricLbl', fontName='Helvetica', fontSize=9, textColor=C_NEUTRAL,
                 alignment=TA_CENTER, leading=13)

# ── Helper functions ─────────────────────────────────────────────────────────────
def hr(color=C_LIGHT, thickness=1):
    return HRFlowable(width='100%', thickness=thickness, color=color, spaceAfter=8, spaceBefore=4)

def section_header(text):
    return [Paragraph(text, sH1), hr(C_PRIMARY, 1.5)]

def img(path, w=16*cm):
    try:
        from PIL import Image as PILImage
        pil = PILImage.open(path)
        pw, ph = pil.size
        ratio = ph / pw
        return Image(path, width=w, height=w * ratio)
    except:
        im = Image(path, width=w)
        return im

def bullet(text):
    return Paragraph(f'<bullet>&bull;</bullet> {text}', sBullet)

def kpi_table(data):
    """data = list of (value, label) tuples"""
    col_w = (W - 4*cm) / len(data)
    tdata = [
        [Paragraph(str(v), sMetricVal) for v, _ in data],
        [Paragraph(str(l), sMetricLbl) for _, l in data],
    ]
    t = Table(tdata, colWidths=[col_w]*len(data))
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), C_LIGHT),
        ('BACKGROUND', (0,1), (-1,1), C_BG),
        ('BOX', (0,0), (-1,-1), 0.5, C_NEUTRAL),
        ('INNERGRID', (0,0), (-1,-1), 0.3, C_NEUTRAL),
        ('TOPPADDING', (0,0), (-1,-1), 10),
        ('BOTTOMPADDING', (0,0), (-1,-1), 10),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('RIGHTPADDING', (0,0), (-1,-1), 6),
    ]))
    return t

# ═══════════════════════════════════════════════════════════════════════════════
# BUILD STORY
# ═══════════════════════════════════════════════════════════════════════════════
story = []

# ── COVER PAGE ──────────────────────────────────────────────────────────────────
cover_data = [[
    Paragraph('FitMetrics Intelligence', sTitle),
    Spacer(1, 0.3*cm),
    Paragraph('Gym Analytics and Predictive Modeling Report', sSubtitle),
    Spacer(1, 0.6*cm),
    Paragraph('A comprehensive analysis of gym member behavior, fitness patterns,<br/>and machine learning driven insights for gym operators and fitness professionals.', sSubtitle),
]]
cover_table = Table(cover_data, colWidths=[W - 4*cm])
cover_table.setStyle(TableStyle([
    ('BACKGROUND', (0,0), (-1,-1), C_PRIMARY),
    ('TOPPADDING', (0,0), (-1,-1), 40),
    ('BOTTOMPADDING', (0,0), (-1,-1), 40),
    ('LEFTPADDING', (0,0), (-1,-1), 30),
    ('RIGHTPADDING', (0,0), (-1,-1), 30),
    ('ROUNDEDCORNERS', [8]),
]))
story.append(Spacer(1, 1.5*cm))
story.append(cover_table)
story.append(Spacer(1, 1*cm))

# KPI strip
story.append(kpi_table([
    ('973', 'Total Members'),
    ('4', 'Workout Types'),
    ('3', 'Experience Tiers'),
    ('15', 'Features Analyzed'),
]))
story.append(Spacer(1, 0.6*cm))

# model metrics strip
story.append(kpi_table([
    (f"{metrics['regression']['r2']*100:.1f}%", 'Regression R-Squared'),
    (f"{metrics['regression']['mae']:.0f} kcal", 'Mean Abs. Error'),
    (f"{metrics['classification']['accuracy']*100:.1f}%", 'Classification Accuracy'),
    (f"{metrics['clustering']['n_clusters']}", 'Member Segments'),
]))

story.append(Spacer(1, 1*cm))
story.append(hr(C_PRIMARY, 1))
story.append(Paragraph('Prepared by <b>Shivam</b>', sSignoff))
story.append(Paragraph('Business Analyst and Data Scientist', S('role', fontName='Helvetica', fontSize=10, textColor=C_NEUTRAL, spaceAfter=2, leading=14)))
story.append(PageBreak())

# ── 1. EXECUTIVE SUMMARY ────────────────────────────────────────────────────────
story += section_header('1. Executive Summary')
story.append(Paragraph(
    'This report presents a full-stack analytical study of 973 gym members drawn from a structured fitness tracking dataset. '
    'The analysis spans exploratory data profiling, behavioral pattern discovery, and three independent machine learning models: '
    'a regression model for calorie burn prediction, a multi-class classifier for experience level identification, '
    'and an unsupervised clustering model for member segmentation.',
    sBody))
story.append(Paragraph(
    'The findings reveal strong correlations between session duration, workout frequency, and caloric output. '
    'HIIT and Strength training consistently produce the highest caloric expenditure. '
    'Body fat percentage declines meaningfully as workout frequency increases. '
    'Predictive models achieved exceptional performance, with the calorie regression model '
    f'reaching an R-squared of {metrics["regression"]["r2"]*100:.1f}% and the experience classifier '
    f'achieving {metrics["classification"]["accuracy"]*100:.1f}% accuracy. '
    'Member clustering revealed four distinct behavioral personas that can guide targeted retention and engagement strategies.',
    sBody))
story.append(Spacer(1, 0.3*cm))

story.append(Paragraph('Key Recommendations', sH2))
for rec in [
    'Target high-frequency HIIT and Strength members with premium membership tiers and performance tracking features.',
    'Introduce hydration and nutrition programs for members showing high fat percentage relative to their workout frequency.',
    'Use the calorie prediction model to power a real-time progress dashboard for members.',
    'Apply the segmentation model to personalize onboarding journeys and class recommendations.',
    'Design retention campaigns specifically for Casual Members (Cluster 1), who show the lowest engagement metrics.',
]:
    story.append(bullet(rec))
story.append(PageBreak())

# ── 2. DATA OVERVIEW ────────────────────────────────────────────────────────────
story += section_header('2. Dataset Overview')
story.append(Paragraph(
    'The dataset contains 973 anonymized gym member records with 15 variables covering demographics, '
    'biometrics, workout behavior, and physiological performance indicators. '
    'There are no missing values, making the dataset suitable for direct modeling without imputation. '
    'The target variables for modeling are Calories Burned (continuous), Experience Level (ordinal categorical), '
    'and member segment (latent cluster).',
    sBody))

schema = [
    ['Feature', 'Type', 'Description'],
    ['Age', 'Numeric', 'Member age in years'],
    ['Gender', 'Categorical', 'Male or Female'],
    ['Weight (kg) / Height (m)', 'Numeric', 'Physical dimensions'],
    ['Max / Avg / Resting BPM', 'Numeric', 'Heart rate measurements'],
    ['Session Duration (hrs)', 'Numeric', 'Length of each workout session'],
    ['Calories Burned', 'Numeric (Target)', 'Total calories per session'],
    ['Workout Type', 'Categorical', 'Cardio, HIIT, Strength, Yoga'],
    ['Fat Percentage', 'Numeric', 'Body fat as percentage of total weight'],
    ['Water Intake (liters)', 'Numeric', 'Daily hydration volume'],
    ['Workout Frequency', 'Numeric', 'Sessions per week'],
    ['Experience Level', 'Ordinal (Target)', 'Beginner (1) to Expert (3)'],
    ['BMI', 'Numeric', 'Derived: Weight / Height squared'],
]
t = Table(schema, colWidths=[5*cm, 4*cm, 8.5*cm])
t.setStyle(TableStyle([
    ('BACKGROUND', (0,0), (-1,0), C_PRIMARY),
    ('TEXTCOLOR', (0,0), (-1,0), C_WHITE),
    ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
    ('FONTSIZE', (0,0), (-1,-1), 9),
    ('ROWBACKGROUNDS', (0,1), (-1,-1), [C_BG, C_LIGHT]),
    ('GRID', (0,0), (-1,-1), 0.3, C_NEUTRAL),
    ('TOPPADDING', (0,0), (-1,-1), 5),
    ('BOTTOMPADDING', (0,0), (-1,-1), 5),
    ('LEFTPADDING', (0,0), (-1,-1), 7),
    ('RIGHTPADDING', (0,0), (-1,-1), 7),
    ('FONTNAME', (0,1), (-1,-1), 'Helvetica'),
    ('TEXTCOLOR', (0,1), (-1,-1), C_DARK),
]))
story.append(t)
story.append(PageBreak())

# ── 3. EXPLORATORY ANALYSIS ──────────────────────────────────────────────────────
story += section_header('3. Exploratory Data Analysis')

story.append(Paragraph('3.1 Caloric Output by Workout Type', sH2))
story.append(Paragraph(
    'HIIT and Cardio sessions generate the highest median calorie burn, '
    'reflecting their aerobic intensity. Yoga, while lower in caloric output, '
    'shows a tighter distribution suggesting more consistent performance across members. '
    'Strength training shows the widest spread, indicating high variability driven by weight, '
    'experience, and session structure.',
    sBody))
story.append(img(f'{VISUALS}/01_calories_by_workout.png'))
story.append(Paragraph('Figure 1: Caloric output distribution across the four workout types, with individual data points overlaid.', sCaption))

story.append(Paragraph('3.2 Session Duration and Caloric Output by Experience', sH2))
story.append(Paragraph(
    'A strong positive linear relationship exists between session duration and calories burned across all experience levels. '
    'Expert members not only train longer on average but burn disproportionately more calories per hour, '
    'likely due to higher intensity and optimized form. Beginners cluster at shorter durations and lower calorie ranges.',
    sBody))
story.append(img(f'{VISUALS}/02_duration_vs_calories.png'))
story.append(Paragraph('Figure 2: Scatter plot with linear regression lines per experience tier.', sCaption))
story.append(PageBreak())

story.append(Paragraph('3.3 Feature Correlations', sH2))
story.append(Paragraph(
    'The correlation matrix reveals several actionable relationships. '
    'Session Duration and Calories Burned share the strongest positive correlation, followed by Workout Frequency. '
    'Fat Percentage correlates negatively with Experience Level, confirming that more experienced members have optimized their body composition. '
    'BMI and Weight are highly collinear, a natural consequence of the BMI formula.',
    sBody))
story.append(img(f'{VISUALS}/03_correlation_heatmap.png', w=14*cm))
story.append(Paragraph('Figure 3: Lower-triangle correlation heatmap across all numeric features.', sCaption))

story.append(Paragraph('3.4 BMI Distribution by Gender and Experience', sH2))
story.append(Paragraph(
    'Both male and female members show declining BMI spread as experience level increases, '
    'suggesting that consistent training normalizes body composition over time. '
    'Beginner members of both genders exhibit the widest BMI ranges, which is expected given the heterogeneous population entering gyms.',
    sBody))
story.append(img(f'{VISUALS}/04_bmi_gender_experience.png'))
story.append(Paragraph('Figure 4: BMI histogram by gender panel, segmented by experience level.', sCaption))
story.append(PageBreak())

story.append(Paragraph('3.5 Heart Rate Profile by Workout Type', sH2))
story.append(Paragraph(
    'HIIT generates the highest average and maximum heart rates, consistent with its high-intensity interval structure. '
    'Yoga shows the lowest BPM across all three measures. '
    'Resting BPM remains relatively stable across workout types, which is expected as it reflects baseline cardiovascular fitness '
    'rather than session intensity.',
    sBody))
story.append(img(f'{VISUALS}/05_heart_rate_profile.png'))
story.append(Paragraph('Figure 5: Grouped bar chart of resting, average, and maximum BPM by workout type.', sCaption))

story.append(Paragraph('3.6 Hydration vs Body Fat', sH2))
story.append(Paragraph(
    'Members with higher daily water intake tend to show lower body fat percentages, '
    'particularly those burning more calories per session (larger bubbles). '
    'This suggests that hydration is a supporting factor in body composition outcomes, '
    'though causality cannot be established from this data alone.',
    sBody))
story.append(img(f'{VISUALS}/06_hydration_vs_fat.png'))
story.append(Paragraph('Figure 6: Bubble chart where bubble size encodes calories burned per session.', sCaption))
story.append(PageBreak())

story.append(Paragraph('3.7 Workout Frequency vs Body Fat Percentage', sH2))
story.append(Paragraph(
    'Members who train 5 or more days per week consistently show lower median body fat percentages. '
    'The interquartile range narrows at higher frequencies, indicating that frequent training produces more predictable body composition outcomes. '
    'Members training only 2 days per week show the highest median fat percentage.',
    sBody))
story.append(img(f'{VISUALS}/10_frequency_vs_fat.png'))
story.append(Paragraph('Figure 7: Box plot of body fat percentage across workout frequency categories.', sCaption))
story.append(PageBreak())

# ── 4. MACHINE LEARNING ──────────────────────────────────────────────────────────
story += section_header('4. Predictive Modeling')

story.append(Paragraph('4.1 Calorie Burn Regression (Random Forest)', sH2))
story.append(Paragraph(
    'A Random Forest Regressor was trained on 80% of the data using 200 estimators to predict calories burned per session. '
    f'The model achieved an R-squared of {metrics["regression"]["r2"]:.4f} on the holdout test set, '
    f'with a Mean Absolute Error of just {metrics["regression"]["mae"]:.1f} kcal. '
    'Session Duration, Average BPM, and Workout Frequency emerged as the top three predictors, '
    'together accounting for the majority of explained variance.',
    sBody))
story.append(img(f'{VISUALS}/08_calorie_feature_importance.png'))
story.append(Paragraph(f'Figure 8: Feature importance ranking for calories burned prediction. R-squared = {metrics["regression"]["r2"]:.3f}.', sCaption))

story.append(Paragraph('4.2 Experience Level Classification (Random Forest)', sH2))
story.append(Paragraph(
    'A multi-class Random Forest Classifier was trained to predict whether a member is a Beginner, Intermediate, or Expert '
    'based on their behavioral and physiological metrics. '
    f'The model achieved {metrics["classification"]["accuracy"]*100:.1f}% accuracy on stratified test data. '
    'This classifier can be embedded into onboarding workflows to immediately profile new members '
    'and recommend appropriate programs without requiring a manual assessment.',
    sBody))
story.append(img(f'{VISUALS}/09_experience_confusion_matrix.png', w=10*cm))
story.append(Paragraph(f'Figure 9: Confusion matrix for experience level classification. Accuracy = {metrics["classification"]["accuracy"]*100:.1f}%.', sCaption))
story.append(PageBreak())

story.append(Paragraph('4.3 Member Segmentation (K-Means Clustering)', sH2))
story.append(Paragraph(
    'K-Means clustering with k=4 was applied to six normalized behavioral and physical features: '
    'Age, BMI, Fat Percentage, Session Duration, Calories Burned, and Workout Frequency. '
    'Principal Component Analysis was used to project clusters into two dimensions for visualization. '
    f'The first two principal components explain {metrics["clustering"]["explained_variance"]:.1f}% of total variance.',
    sBody))
story.append(Paragraph('Four distinct member personas emerged:', sBody))
for persona, desc in [
    ('High Performers', 'High calorie burn, long sessions, low fat percentage. These are the gym champions.'),
    ('Casual Members', 'Low frequency, shorter sessions, higher fat percentage. At-risk for churn.'),
    ('Intensive Trainers', 'Very high BPM, short to medium sessions, focused on intensity over volume.'),
    ('Balanced Athletes', 'Moderate across all metrics, consistent frequency. The stable membership core.'),
]:
    story.append(bullet(f'<b>{persona}:</b> {desc}'))
story.append(Spacer(1, 0.3*cm))
story.append(img(f'{VISUALS}/07_member_segmentation.png'))
story.append(Paragraph('Figure 10: K-Means member segments projected via PCA. Centroids marked with X.', sCaption))
story.append(PageBreak())

# ── 5. BUSINESS RECOMMENDATIONS ──────────────────────────────────────────────────
story += section_header('5. Business Recommendations')

recs = [
    ('Retention Strategy', [
        'Casual Members (Cluster 1) are the highest churn risk. Introduce check-in nudges, progress milestone rewards, and weekly goal-setting prompts.',
        'Pair Casual Members with expert-level mentors or group sessions to increase social commitment.',
    ]),
    ('Revenue Optimization', [
        'High Performers and Intensive Trainers are the most engaged segments. Offer premium tiers with advanced performance analytics, nutrition integration, and personal coaching.',
        'HIIT and Strength classes show the highest caloric output. Increase class frequency and introduce premium slots during peak demand windows.',
    ]),
    ('Health and Wellness Programs', [
        'Members with high fat percentage and low workout frequency should be enrolled in structured beginner pathways with progressive difficulty.',
        'Introduce a hydration tracking challenge linked to the water intake data, given its correlation with fat percentage outcomes.',
    ]),
    ('Technology Integration', [
        'Deploy the calorie regression model as a live session estimator within a mobile app, updating in real time based on heart rate and duration.',
        'Use the experience classifier during sign-up to auto-assign members to the right program track without manual assessment.',
        'Run the segmentation model quarterly to detect cluster migration, which signals whether members are progressing or at risk.',
    ]),
]
for title, points in recs:
    story.append(Paragraph(title, sH2))
    for p in points:
        story.append(bullet(p))
    story.append(Spacer(1, 0.2*cm))
story.append(PageBreak())

# ── 6. CONCLUSIONS ───────────────────────────────────────────────────────────────
story += section_header('6. Conclusions')
story.append(Paragraph(
    'This analysis demonstrates that gym member data, even at relatively modest scale, '
    'can yield high-fidelity predictive models and rich behavioral insights. '
    'The three-model framework covering regression, classification, and clustering '
    'offers both operational value (calorie estimation, member profiling) '
    'and strategic value (segmentation, retention targeting).',
    sBody))
story.append(Paragraph(
    'The most impactful variables across all analyses are Session Duration, Workout Frequency, '
    'Average BPM, and Fat Percentage. Gym operators should prioritize collecting and maintaining '
    'these fields at the highest accuracy to ensure model performance in production.',
    sBody))
story.append(Paragraph(
    'The next phase of this work should focus on longitudinal tracking to observe cluster migration over time, '
    'integration with wearable device data for real-time BPM capture, '
    'and A/B testing of the retention and upsell strategies identified through segmentation.',
    sBody))
story.append(Spacer(1, 1.5*cm))
story.append(hr(C_PRIMARY, 1))
story.append(Spacer(1, 0.4*cm))
story.append(Paragraph('Prepared and Authored by', S('prep', fontName='Helvetica', fontSize=10, textColor=C_NEUTRAL, spaceAfter=2, leading=13)))
story.append(Paragraph('Shivam', sSignoff))
story.append(Paragraph('Business Analyst and Data Scientist', S('role2', fontName='Helvetica', fontSize=10, textColor=C_NEUTRAL, spaceAfter=2, leading=14)))
story.append(Spacer(1, 0.4*cm))
story.append(Paragraph(
    'All models trained on anonymized gym member data (n=973). '
    'Random Forest models use 200 estimators with an 80/20 train-test split. '
    'Clustering uses K-Means with k=4 on standardized features.',
    S('footnote', fontName='Helvetica-Oblique', fontSize=8, textColor=C_NEUTRAL, leading=12, spaceAfter=4)))

# ── BUILD ────────────────────────────────────────────────────────────────────────
doc = make_doc(OUTPUT)
doc.build(story, canvasmaker=NumberedCanvas)
print(f"PDF saved to {OUTPUT}")
