# ============================================================
# CLASS BUNK PREDICTOR — Machine Learning Approach
# Data Science Project | Linear & Logistic Regression
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (mean_absolute_error, r2_score,
                              accuracy_score, classification_report,
                              confusion_matrix)
from sklearn.preprocessing import StandardScaler

# ── 1. GENERATE SYNTHETIC DATASET ───────────────────────────
np.random.seed(42)
n = 300

total_classes      = np.random.randint(80, 150, n)
attended           = np.array([np.random.randint(int(t*0.5), t+1) for t in total_classes])
remaining          = np.array([np.random.randint(0, int(t*0.4)+1) for t in total_classes])
threshold          = np.random.choice([75, 80, 85], n)
current_pct        = (attended / total_classes * 100).round(2)
study_hours        = np.random.randint(1, 8, n)
subject_difficulty = np.random.randint(1, 6, n)   # 1 = easy … 5 = hard

# Ground truth labels
effective_total = attended + remaining
min_required    = np.ceil(effective_total * threshold / 100).astype(int)
can_bunk        = np.maximum(0, remaining - np.maximum(0, min_required - attended))
safe_to_bunk    = (can_bunk > 0).astype(int)

df = pd.DataFrame({
    'total_classes':      total_classes,
    'attended':           attended,
    'remaining':          remaining,
    'threshold':          threshold,
    'current_pct':        current_pct,
    'study_hours':        study_hours,
    'subject_difficulty': subject_difficulty,
    'can_bunk':           can_bunk,       # regression target
    'safe_to_bunk':       safe_to_bunk,   # classification target
})

print("Dataset shape:", df.shape)
print(df.head())
print("\ncan_bunk stats:\n", df['can_bunk'].describe())

# ── 2. FEATURE / TARGET SPLIT ───────────────────────────────
features = ['total_classes', 'attended', 'remaining', 'threshold',
            'current_pct', 'study_hours', 'subject_difficulty']

X   = df[features]
y_r = df['can_bunk']          # regression target
y_c = df['safe_to_bunk']      # classification target

X_train, X_test, yr_train, yr_test, yc_train, yc_test = train_test_split(
    X, y_r, y_c, test_size=0.2, random_state=42)

scaler      = StandardScaler()
X_train_sc  = scaler.fit_transform(X_train)
X_test_sc   = scaler.transform(X_test)

# ── 3. LINEAR REGRESSION — predict exact bunk count ─────────
lr    = LinearRegression()
lr.fit(X_train_sc, yr_train)
yr_pred = lr.predict(X_test_sc)

mae = mean_absolute_error(yr_test, yr_pred)
r2  = r2_score(yr_test, yr_pred)
print(f"\n[Linear Regression]  MAE = {mae:.2f}  |  R² = {r2:.4f}")

coef_df = pd.DataFrame({'feature': features, 'coefficient': lr.coef_}
                        ).sort_values('coefficient', key=abs, ascending=False)
print(coef_df.to_string(index=False))

# ── 4. LOGISTIC REGRESSION — classify safe/unsafe ───────────
log  = LogisticRegression(max_iter=1000)
log.fit(X_train_sc, yc_train)
yc_pred = log.predict(X_test_sc)

acc = accuracy_score(yc_test, yc_pred)
print(f"\n[Logistic Regression] Accuracy = {acc*100:.2f}%")
print(classification_report(yc_test, yc_pred, target_names=['Not Safe', 'Safe']))

# ── 5. VISUALISATIONS ───────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Class Bunk Predictor — ML Analysis', fontsize=14, fontweight='bold')

# 5a. Actual vs Predicted
ax = axes[0, 0]
ax.scatter(yr_test, yr_pred, alpha=0.6, color='steelblue', edgecolors='white', linewidths=0.4)
lim = max(yr_test.max(), yr_pred.max()) + 2
ax.plot([0, lim], [0, lim], 'r--', linewidth=1.5, label='Perfect Fit')
ax.set_xlabel('Actual Bunks'); ax.set_ylabel('Predicted Bunks')
ax.set_title(f'Actual vs Predicted  (R²={r2:.2f}, MAE={mae:.2f})')
ax.legend()

# 5b. Feature Coefficients
ax = axes[0, 1]
colors = ['#2d9e6b' if c > 0 else '#e63946' for c in coef_df['coefficient']]
ax.barh(coef_df['feature'], coef_df['coefficient'], color=colors)
ax.axvline(0, color='black', linewidth=0.8)
ax.set_title('Linear Regression Coefficients')
ax.set_xlabel('Coefficient Value')

# 5c. Distribution of can_bunk
ax = axes[1, 0]
ax.hist(df['can_bunk'], bins=20, color='steelblue', edgecolor='white', linewidth=0.5)
ax.set_xlabel('can_bunk'); ax.set_ylabel('Count')
ax.set_title('Distribution of Bunkable Classes')

# 5d. Confusion Matrix
ax = axes[1, 1]
cm = confusion_matrix(yc_test, yc_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['Not Safe', 'Safe'],
            yticklabels=['Not Safe', 'Safe'])
ax.set_title(f'Logistic Regression Confusion Matrix\nAccuracy = {acc*100:.1f}%')

plt.tight_layout()
plt.savefig('bunk_ml_analysis.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nPlot saved as bunk_ml_analysis.png")

# ── 6. PREDICTION FUNCTION ──────────────────────────────────
def predict_bunks(total, attended, remaining, threshold=75,
                  study_hours=3, difficulty=3):
    """
    Predict how many classes a student can safely bunk.
    Uses the trained Linear + Logistic Regression models.
    """
    current_pct = round(attended / total * 100, 2)
    X_new = scaler.transform([[total, attended, remaining, threshold,
                                current_pct, study_hours, difficulty]])
    bunk_count  = max(0, round(lr.predict(X_new)[0]))
    is_safe     = bool(log.predict(X_new)[0])
    return {
        'predicted_bunks':  bunk_count,
        'safe_to_bunk':     is_safe,
        'current_pct':      current_pct,
        'confidence':       f"± {mae:.1f} classes"
    }

# Example
print("\n── EXAMPLE PREDICTION ──")
result = predict_bunks(total=120, attended=75, remaining=30, threshold=75)
for k, v in result.items():
    print(f"  {k:20s}: {v}")
