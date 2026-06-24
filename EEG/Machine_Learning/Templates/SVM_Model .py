# ──────────────────────────────────────────────────────────────────────────────
# SVM Template with State-of-the-Art Grid Search
# ──────────────────────────────────────────────────────────────────────────────

# Core Python libraries
import pandas as pd
import numpy as np
import os
import warnings
from pathlib import Path
from datetime import datetime
import logging
import matplotlib.pyplot as plt
import seaborn as sns

# Core ML libraries
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance

# ──────────────────────────────────────────────────────────────────────────────
# 1. CONFIGURATION & SETUP
# ──────────────────────────────────────────────────────────────────────────────

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

OUTPUT_DIR = Path("./svm_sota_results")
FIGURES_DIR = OUTPUT_DIR / "figures"
TABLES_DIR = OUTPUT_DIR / "tables"

for dir_path in [OUTPUT_DIR, FIGURES_DIR, TABLES_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

RANDOM_STATE = 42
N_RUNS_STABILITY = 50
TOP_K_FEATURES = 15

# ──────────────────────────────────────────────────────────────────────────────
# 2. DATA PREPARATION (USER INPUT REQUIRED)
# ──────────────────────────────────────────────────────────────────────────────

# NOTE: In a real scenario, replace this block with your data loading logic.
# Example:
# df = pd.read_csv('your_data.csv')
# X = df.drop('target', axis=1)
# y = df['target']
# feature_names = X.columns.tolist()
# RUN_NAME = "My_SVM_Experiment"

try:
    X
    y
    feature_names
    RUN_NAME
except NameError:
    logger.warning("Data variables not found. Generating synthetic data for demonstration.")
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, random_state=RANDOM_STATE)
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    RUN_NAME = "Synthetic_Demo_Run"

if isinstance(X, pd.DataFrame):
    feature_names = X.columns.tolist()
    X = X.values

# ──────────────────────────────────────────────────────────────────────────────
# 3. UTILITY: GPU CHECK
# ──────────────────────────────────────────────────────────────────────────────

logger.info("Checking environment...")
device = 'cpu' # Sklearn SVC is CPU-bound

# ──────────────────────────────────────────────────────────────────────────────
# 4. PREPROCESSING & HYPERPARAMETER TUNING (STATE-OF-THE-ART GRID)
# ──────────────────────────────────────────────────────────────────────────────

logger.info(f"Starting SOTA analysis for: {RUN_NAME}")

le = LabelEncoder()
y_encoded = le.fit_transform(y)

# State-of-the-Art Parameter Grid
# Optimized for balance between coverage and computational cost
param_grid = {
    'svc__C': [0.01, 0.1, 1, 10, 100],
    'svc__kernel': ['rbf', 'linear', 'poly'],
    'svc__gamma': ['scale', 'auto', 0.01, 0.1],
    'svc__degree': [2, 3],  # Reduced to most common degrees
    'svc__coef0': [0.0, 1.0],
    'svc__class_weight': [None, 'balanced']
}

# Pipeline with StandardScaler for robust feature normalization
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('svc', SVC(probability=True, random_state=RANDOM_STATE))
])

cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

logger.info("Starting Comprehensive Grid Search...")
grid_search = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    cv=cv_strategy,
    scoring='accuracy', 
    n_jobs=-1,
    verbose=1,
    return_train_score=False,
    pre_dispatch='2*n_jobs'
)

grid_search.fit(X, y_encoded)

logger.info(f"Best Parameters Found: {grid_search.best_params_}")
logger.info(f"Best CV Score: {grid_search.best_score_:.3f}")

best_params = grid_search.best_params_

# ──────────────────────────────────────────────────────────────────────────────
# 5. FINAL TRAIN/TEST SPLIT
# ──────────────────────────────────────────────────────────────────────────────

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, stratify=y_encoded,
    random_state=RANDOM_STATE
)

logger.info(f"Training set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")

# ──────────────────────────────────────────────────────────────────────────────
# 6. MODEL TRAINING (MAIN RUN WITH BEST PARAMS)
# ──────────────────────────────────────────────────────────────────────────────

logger.info(f"Training Final SVM model with tuned parameters...")

# Reconstruct pipeline with best params
final_params = {k.replace('svc__', ''): v for k, v in best_params.items()}
clf = Pipeline([
    ('scaler', StandardScaler()),
    ('svc', SVC(probability=True, random_state=RANDOM_STATE, **final_params))
])

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:, 1] if len(np.unique(y_encoded)) == 2 else None

# ──────────────────────────────────────────────────────────────────────────────
# 7. EVALUATION METRICS
# ──────────────────────────────────────────────────────────────────────────────

metrics = {
    'accuracy': accuracy_score(y_test, y_pred),
    'precision': precision_score(y_test, y_pred, zero_division=0, average='weighted'),
    'recall': recall_score(y_test, y_pred, zero_division=0, average='weighted'),
    'f1': f1_score(y_test, y_pred, zero_division=0, average='weighted'),
}

if len(np.unique(y_encoded)) == 2:
    metrics['auc'] = roc_auc_score(y_test, y_proba)
else:
    metrics['auc'] = roc_auc_score(y_test, clf.predict_proba(X_test), multi_class='ovr')

logger.info(f"--- Results for {RUN_NAME} ---")
for k, v in metrics.items():
    logger.info(f"{k.capitalize()}: {v:.3f}")

# Permutation Importance for Model-Agnostic Feature Analysis
logger.info("Calculating Permutation Importance...")
perm_imp = permutation_importance(clf, X_test, y_test, n_repeats=30, random_state=RANDOM_STATE, scoring='accuracy')
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': perm_imp.importances_mean
}).sort_values('importance', ascending=False)

# ──────────────────────────────────────────────────────────────────────────────
# 8. FEATURE STABILITY ANALYSIS
# ──────────────────────────────────────────────────────────────────────────────

logger.info(f"Computing feature stability ({N_RUNS_STABILITY} runs)...")

feature_counts = {name: 0 for name in feature_names}
all_importances_list = []

for i in range(N_RUNS_STABILITY):
    X_train_stab, X_test_stab, y_train_stab, y_test_stab = train_test_split(
        X, y_encoded, test_size=0.2, stratify=y_encoded,
        random_state=RANDOM_STATE + i
    )
    
    clf_stab = Pipeline([
        ('scaler', StandardScaler()),
        ('svc', SVC(probability=True, random_state=RANDOM_STATE + i, **final_params))
    ])
    
    clf_stab.fit(X_train_stab, y_train_stab)
    
    perm_imp_run = permutation_importance(clf_stab, X_test_stab, y_test_stab, n_repeats=10, random_state=RANDOM_STATE + i, scoring='accuracy')
    
    imp_df_run = pd.DataFrame({
        'feature': feature_names,
        'importance': perm_imp_run.importances_mean,
        'run': i
    }).sort_values('importance', ascending=False).head(TOP_K_FEATURES)
    
    all_importances_list.append(imp_df_run)
    
    for feature in imp_df_run['feature']:
        if feature in feature_counts:
            feature_counts[feature] += 1

    if (i + 1) % 10 == 0:
        logger.info(f"Stability run {i + 1}/{N_RUNS_STABILITY} complete.")

all_imp_concat = pd.concat(all_importances_list, ignore_index=True)

stability_data = []
for feat, count in feature_counts.items():
    stability_data.append({
        'feature': feat,
        'stability_score': count / N_RUNS_STABILITY,
        'selection_frequency': count
    })
    
stability_df = pd.DataFrame(stability_data).sort_values('stability_score', ascending=False)
mean_imp = all_imp_concat.groupby('feature')['importance'].mean()
stability_df['mean_importance'] = stability_df['feature'].map(mean_imp).fillna(0)

# ──────────────────────────────────────────────────────────────────────────────
# 9. VISUALIZATION
# ──────────────────────────────────────────────────────────────────────────────

logger.info("Generating plots...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle(f'{RUN_NAME} - SOTA SVM Analysis', fontsize=16, fontweight='bold')

# 1. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
           xticklabels=le.classes_,
           yticklabels=le.classes_, ax=axes[0,0])
axes[0,0].set_title('Confusion Matrix')

# 2. Top 10 Feature Importances
top_features = importance_df.head(10)
axes[0,1].barh(range(len(top_features)), top_features['importance'])
axes[0,1].set_yticks(range(len(top_features)))
labels = [f[:30] + '...' if len(f) > 30 else f for f in top_features['feature']]
axes[0,1].set_yticklabels(labels, fontsize=8)
axes[0,1].set_title('Top 10 Permutation Importance')
axes[0,1].invert_yaxis()

# 3. Feature Stability
stable_features = stability_df.head(10)
bars = axes[1,0].bar(range(len(stable_features)), stable_features['stability_score'])
axes[1,0].set_title('Top 10 Most Stable Features')
axes[1,0].set_xticks(range(len(stable_features)))
axes[1,0].set_xticklabels([f"F{i+1}" for i in range(len(stable_features))])

# 4. Performance Metrics
metric_names = list(metrics.keys())
metric_values = list(metrics.values())
bars = axes[1,1].bar(metric_names, metric_values, color='skyblue', alpha=0.8)
axes[1,1].set_title('Classification Metrics')
axes[1,1].set_ylim(0, 1.1) 

for bar, value in zip(bars, metric_values):
    axes[1,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                  f'{value:.3f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
save_path = FIGURES_DIR / f'{RUN_NAME}_analysis.png'
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.show()

# ──────────────────────────────────────────────────────────────────────────────
# 10. SAVE TABLES & RESULTS
# ──────────────────────────────────────────────────────────────────────────────

stability_df.head(20).to_csv(TABLES_DIR / f'{RUN_NAME}_stability.csv', index=False)
importance_df.head(20).to_csv(TABLES_DIR / f'{RUN_NAME}_importance.csv', index=False)
pd.DataFrame(list(best_params.items()), columns=['Parameter', 'Value']).to_csv(TABLES_DIR / f'{RUN_NAME}_best_params.csv', index=False)

results = {
    'run_name': RUN_NAME,
    'model': clf,
    'best_params': best_params,
    'metrics': metrics,
    'importance': importance_df,
    'stability': stability_df
}

print("\n✅ SOTA Analysis Finished Successfully.")
print(f" Best Parameters: {best_params}")