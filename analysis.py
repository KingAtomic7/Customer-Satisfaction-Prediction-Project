"""
Customer Satisfaction Prediction - Full Analysis & ML Pipeline
Unified Mentor Internship Project
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, classification_report, 
                              confusion_matrix, f1_score)

# ── Paths ──────────────────────────────────────────────────────────────
BASE = "/home/claude/customer_satisfaction_prediction"
DATA_PATH = f"{BASE}/data/customer_support_tickets.csv"
FIG_DIR = f"{BASE}/outputs/figures"
os.makedirs(FIG_DIR, exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted")
COLORS = ["#4C72B0","#DD8452","#55A868","#C44E52","#8172B2"]

# ══════════════════════════════════════════════════════════════════════
# 1. LOAD & BASIC EDA
# ══════════════════════════════════════════════════════════════════════
print("=" * 60)
print("  CUSTOMER SATISFACTION PREDICTION - ML PIPELINE")
print("=" * 60)

data = pd.read_csv(DATA_PATH)
print(f"\n[1] Dataset Shape: {data.shape}")
print(f"    Columns: {list(data.columns)}")
print(f"\n[2] Missing Values:\n{data.isnull().sum()}")
print(f"\n[3] Basic Stats:\n{data[['Customer Age','Customer Satisfaction Rating']].describe()}")

# ══════════════════════════════════════════════════════════════════════
# 2. EDA CHARTS
# ══════════════════════════════════════════════════════════════════════

# Chart 1 – Satisfaction Distribution
fig, ax = plt.subplots(figsize=(9, 5))
closed = data.dropna(subset=['Customer Satisfaction Rating'])
sns.histplot(closed['Customer Satisfaction Rating'], bins=5, kde=True, color=COLORS[0], ax=ax)
ax.set_title('Customer Satisfaction Rating Distribution', fontsize=14, fontweight='bold')
ax.set_xlabel('Rating (1–5)')
ax.set_ylabel('Frequency')
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/1_satisfaction_distribution.png", dpi=150)
plt.close()
print("\n[Chart 1] Satisfaction Distribution saved.")

# Chart 2 – Ticket Status Pie
status_counts = data['Ticket Status'].value_counts()
fig, ax = plt.subplots(figsize=(7, 7))
ax.pie(status_counts, labels=status_counts.index, autopct='%1.1f%%',
       colors=COLORS[:3], startangle=140)
ax.set_title('Ticket Status Distribution', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/2_ticket_status.png", dpi=150)
plt.close()
print("[Chart 2] Ticket Status saved.")

# Chart 3 – Ticket Channel Bar
channel_counts = data['Ticket Channel'].value_counts()
fig, ax = plt.subplots(figsize=(9, 5))
channel_counts.plot(kind='bar', color=COLORS, ax=ax)
ax.set_title('Ticket Channel Distribution', fontsize=14, fontweight='bold')
ax.set_xlabel('Channel'); ax.set_ylabel('Count')
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/3_ticket_channel.png", dpi=150)
plt.close()
print("[Chart 3] Ticket Channel saved.")

# Chart 4 – Avg Satisfaction by Gender
avg_sat = data.groupby('Customer Gender')['Customer Satisfaction Rating'].mean().reset_index()
fig, ax = plt.subplots(figsize=(8, 5))
sns.barplot(x='Customer Gender', y='Customer Satisfaction Rating',
            data=avg_sat, palette='muted', ax=ax,
            order=['Male','Female','Other'])
ax.set_ylim(1, 5)
ax.set_title('Avg Customer Satisfaction by Gender', fontsize=14, fontweight='bold')
ax.set_xlabel('Gender'); ax.set_ylabel('Avg Rating')
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/4_satisfaction_by_gender.png", dpi=150)
plt.close()
print("[Chart 4] Satisfaction by Gender saved.")

# Chart 5 – Top 10 Products
product_counts = data['Product Purchased'].value_counts().head(10)
fig, ax = plt.subplots(figsize=(10, 6))
product_counts.sort_values().plot(kind='barh', color=COLORS[0], ax=ax)
ax.set_title('Top 10 Products Purchased', fontsize=14, fontweight='bold')
ax.set_xlabel('Count')
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/5_top_products.png", dpi=150)
plt.close()
print("[Chart 5] Top Products saved.")

# Chart 6 – Ticket Type Pie
type_counts = data['Ticket Type'].value_counts()
fig, ax = plt.subplots(figsize=(8, 8))
ax.pie(type_counts, labels=type_counts.index, autopct='%1.1f%%',
       colors=COLORS, startangle=90)
ax.set_title('Ticket Type Distribution', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/6_ticket_type.png", dpi=150)
plt.close()
print("[Chart 6] Ticket Type saved.")

# Chart 7 – Priority Distribution
priority_counts = data['Ticket Priority'].value_counts()
fig, ax = plt.subplots(figsize=(8, 8))
ax.pie(priority_counts, labels=priority_counts.index, autopct='%1.1f%%',
       colors=COLORS[:4], startangle=140)
ax.set_title('Ticket Priority Distribution', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/7_priority_distribution.png", dpi=150)
plt.close()
print("[Chart 7] Priority Distribution saved.")

# Chart 8 – Tickets by Age Group
bins = [0, 20, 30, 40, 50, 60, 70, 80]
labels = ['<20','21-30','31-40','41-50','51-60','61-70','71+']
data['Age Group'] = pd.cut(data['Customer Age'], bins=bins, labels=labels, right=False)
age_tickets = data.groupby('Age Group', observed=False).size()
fig, ax = plt.subplots(figsize=(10, 5))
age_tickets.plot(kind='bar', color=COLORS[0], ax=ax)
ax.set_title('Tickets Raised by Age Group', fontsize=14, fontweight='bold')
ax.set_xlabel('Age Group'); ax.set_ylabel('Number of Tickets')
plt.xticks(rotation=30)
plt.grid(axis='y')
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/8_tickets_by_age.png", dpi=150)
plt.close()
print("[Chart 8] Tickets by Age saved.")

# Chart 9 – Ticket Trends Over Time
data['Date of Purchase'] = pd.to_datetime(data['Date of Purchase'])
data['YearMonth'] = data['Date of Purchase'].dt.to_period('M')
trends = data.groupby('YearMonth').size()
fig, ax = plt.subplots(figsize=(12, 5))
trends.plot(kind='line', marker='o', color=COLORS[0], ax=ax)
ax.set_title('Ticket Trends Over Time', fontsize=14, fontweight='bold')
ax.set_xlabel('Year-Month'); ax.set_ylabel('Number of Tickets')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/9_ticket_trends.png", dpi=150)
plt.close()
print("[Chart 9] Ticket Trends saved.")

# ══════════════════════════════════════════════════════════════════════
# 3. MACHINE LEARNING – CUSTOMER SATISFACTION PREDICTION
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  MACHINE LEARNING PIPELINE")
print("=" * 60)

# Use only closed tickets (have satisfaction rating)
ml_data = data.dropna(subset=['Customer Satisfaction Rating']).copy()
print(f"\nML Dataset size (closed tickets): {ml_data.shape[0]}")

# Feature engineering
ml_data['Satisfaction_Category'] = ml_data['Customer Satisfaction Rating'].apply(
    lambda x: 'High' if x >= 4 else ('Low' if x <= 2 else 'Medium')
)

# Encode categorical features
cat_cols = ['Customer Gender', 'Product Purchased', 'Ticket Type',
            'Ticket Subject', 'Ticket Priority', 'Ticket Channel']

le_map = {}
for col in cat_cols:
    le = LabelEncoder()
    ml_data[col + '_enc'] = le.fit_transform(ml_data[col])
    le_map[col] = le

target_le = LabelEncoder()
ml_data['target'] = target_le.fit_transform(ml_data['Satisfaction_Category'])

feature_cols = ['Customer Age'] + [c + '_enc' for c in cat_cols]
X = ml_data[feature_cols]
y = ml_data['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

print(f"\nTraining set: {X_train.shape[0]} | Test set: {X_test.shape[0]}")
print(f"Classes: {list(target_le.classes_)}")

# ─── Train Models ───────────────────────────────────────────────────
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=8, random_state=42),
}

results = {}
print("\n[Model Training & Evaluation]")
for name, model in models.items():
    if name == "Logistic Regression":
        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s)
        cv_scores = cross_val_score(model, X_train_s, y_train, cv=5)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    
    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average='weighted')
    results[name] = {
        "accuracy": round(acc, 4),
        "f1": round(f1, 4),
        "cv_mean": round(cv_scores.mean(), 4),
        "cv_std": round(cv_scores.std(), 4),
        "model": model,
        "y_pred": y_pred
    }
    print(f"  {name:25s} | Acc={acc:.4f} | F1={f1:.4f} | CV={cv_scores.mean():.4f}±{cv_scores.std():.4f}")

# Best model
best_name = max(results, key=lambda k: results[k]['accuracy'])
best = results[best_name]
print(f"\n[Best Model] {best_name} (Accuracy: {best['accuracy']})")
print("\n[Classification Report – Best Model]")
print(classification_report(y_test, best['y_pred'], target_names=target_le.classes_))

# ─── Chart 10: Model Comparison ────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
names_list = list(results.keys())
accs = [results[n]['accuracy'] for n in names_list]
f1s  = [results[n]['f1'] for n in names_list]
x = np.arange(len(names_list))
w = 0.35
ax.bar(x - w/2, accs, w, label='Accuracy', color=COLORS[0])
ax.bar(x + w/2, f1s,  w, label='F1 Score', color=COLORS[1])
ax.set_xticks(x); ax.set_xticklabels(names_list, rotation=15, ha='right')
ax.set_ylim(0, 1.1)
ax.set_title('Model Comparison – Accuracy & F1 Score', fontsize=14, fontweight='bold')
ax.legend()
for i, (a, f) in enumerate(zip(accs, f1s)):
    ax.text(i-w/2, a+0.01, f"{a:.3f}", ha='center', fontsize=9)
    ax.text(i+w/2, f+0.01, f"{f:.3f}", ha='center', fontsize=9)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/10_model_comparison.png", dpi=150)
plt.close()
print("[Chart 10] Model Comparison saved.")

# ─── Chart 11: Confusion Matrix (Best Model) ─────────────────────
cm = confusion_matrix(y_test, best['y_pred'])
fig, ax = plt.subplots(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=target_le.classes_,
            yticklabels=target_le.classes_, ax=ax)
ax.set_title(f'Confusion Matrix – {best_name}', fontsize=13, fontweight='bold')
ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/11_confusion_matrix.png", dpi=150)
plt.close()
print("[Chart 11] Confusion Matrix saved.")

# ─── Chart 12: Feature Importance (RF) ──────────────────────────
rf_model = results['Random Forest']['model']
feat_imp = pd.Series(rf_model.feature_importances_, index=feature_cols).nlargest(10)
fig, ax = plt.subplots(figsize=(10, 5))
feat_imp.sort_values().plot(kind='barh', color=COLORS[2], ax=ax)
ax.set_title('Top 10 Feature Importances (Random Forest)', fontsize=14, fontweight='bold')
ax.set_xlabel('Importance')
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/12_feature_importance.png", dpi=150)
plt.close()
print("[Chart 12] Feature Importance saved.")

# ─── Save results summary ──────────────────────────────────────────
results_df = pd.DataFrame([
    {"Model": n, "Accuracy": r['accuracy'], "F1 Score": r['f1'],
     "CV Mean": r['cv_mean'], "CV Std": r['cv_std']}
    for n, r in results.items()
]).sort_values("Accuracy", ascending=False)
results_df.to_csv(f"{BASE}/outputs/model_results.csv", index=False)

# Common issues
top_issues = data['Ticket Subject'].value_counts().head(10)
top_issues.to_csv(f"{BASE}/outputs/top_issues.csv")

print(f"\n[✓] All charts saved to {FIG_DIR}")
print(f"[✓] Model results saved.")
print("\n" + "=" * 60)
print("  PIPELINE COMPLETE")
print("=" * 60)

# Return results for report generation
print("\n[SUMMARY FOR REPORT]")
for name, r in results.items():
    print(f"  {name}: Accuracy={r['accuracy']}, F1={r['f1']}")
print(f"  Best Model: {best_name}")
print(f"  Best Accuracy: {best['accuracy']}")
