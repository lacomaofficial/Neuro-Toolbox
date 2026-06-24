# ──────────────────────────────────────────────────────────────────────────────
# XGBoost Template with SOTA Hyperparameter Tuning (Randomized Search)
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
from scipy.stats import uniform, randint

# Core ML libraries
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
from sklearn.preprocessing import LabelEncoder

# ──────────────────────────────────────────────────────────────────────────────
# 1. CONFIGURATION & SETUP
# ──────────────────────────────────────────────────────────────────────────────

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

OUTPUT_DIR = Path("./xgb_analysis_results_sota")
FIGURES_DIR = OUTPUT_DIR / "figures"
TABLES_DIR = OUTPUT_DIR / "tables"

for dir_path in [OUTPUT_DIR, FIGURES_DIR, TABLES_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

RANDOM_STATE = 42
N_RUNS_STABILITY = 50
TOP_K_FEATURES = 15
N_ITER_SEARCH = 100  # Number of parameter settings sampled for RandomizedSearch

# ⚠️ USER MUST SET DEVICE MANUALLY ('cuda' or 'cpu')
device = 'cpu' 
tree_method = 'hist' if device == 'cpu' else 'gpu_hist'

# ──────────────────────────────────────────────────────────────────────────────
# 2. DATA PREPARATION (USER INPUT REQUIRED)
# ──────────────────────────────────────────────────────────────────────────────

# ⚠️ USER MUST DEFINE THESE VARIABLES BEFORE RUNNING
# RUN_NAME = "Alpha_Band_SOTA_Tuned"
# X = ... 
# y = ... 
# feature_names = ... 

# ──────────────────────────────────────────────────────────────────────────────
# 3. PREPROCESSING & HYPERPARAMETER TUNING (RANDOMIZED SEARCH)
# ──────────────────────────────────────────────────────────────────────────────

logger.info(f"Starting SOTA analysis for: {RUN_NAME}")

le = LabelEncoder()
y_encoded = le.fit_transform(y)

# SOTA Parameter Distributions
param_distributions = {
    'learning_rate': uniform(0.01, 0.2),      # Continuous uniform between 0.01 and 0.21
    'max_depth': randint(3, 11),               # Integers between 3 and 10
    'n_estimators': randint(100, 1001),        # Integers between 100 and 1000
    'subsample': uniform(0.6, 0.4),            # Continuous uniform between 0.6 and 1.0
    'colsample_bytree': uniform(0.6, 0.4),     # Continuous uniform between 0.6 and 1.0
    'colsample_bylevel': uniform(0.6, 0.4),    # Column sampling per level
    'reg_alpha': [0, 0.01, 0.1, 1, 10],        # L1 regularization
    'reg_lambda': [0.1, 1, 10, 100],           # L2 regularization
    'min_child_weight': randint(1, 11),        # Minimum sum of instance weight in child
    'gamma': [0, 0.1, 0.2, 0.5, 1]             # Minimum loss reduction required
}

base_clf = XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    random_state=RANDOM_STATE,
    n_jobs=-1,
    tree_method=tree_method,
    device=device,
    verbosity=0
)

cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

logger.info(f"Starting Randomized Search ({N_ITER_SEARCH} iterations) for Hyperparameter Tuning...")
random_search = RandomizedSearchCV(
    estimator=base_clf,
    param_distributions=param_distributions,
    n_iter=N_ITER_SEARCH,
    cv=cv_strategy,
    scoring='roc_auc',  # AUC is more robust than accuracy for tuning
    n_jobs=-1,
    verbose=1,
    random_state=RANDOM_STATE,
    return_train_score=False
)

random_search.fit(X, y_encoded)

logger.info(f"Best Parameters Found: {random_search.best_params_}")
logger.info(f"Best CV Score during Tuning: {random_search.best_score_:.3f}")

best_params = random_search.best_params_

# ──────────────────────────────────────────────────────────────────────────────
# 4. FINAL TRAIN/TEST SPLIT
# ──────────────────────────────────────────────────────────────────────────────

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, stratify=y_encoded,
    random_state=RANDOM_STATE
)

logger.info(f"Training set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")

# ──────────────────────────────────────────────────────────────────────────────
# 5. MODEL TRAINING (MAIN RUN WITH BEST PARAMS)
# ──────────────────────────────────────────────────────────────────────────────

logger.info(f"Training Final XGBoost model on {device.upper()} with tuned parameters...")

clf = XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    random_state=RANDOM_STATE,
    n_jobs=-1,
    tree_method=tree_method,
    device=device,
    verbosity=0,
    **best_params
)

# Note: For SOTA performance, consider using early_stopping_rounds here if you have a validation set
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:, 1]

# ──────────────────────────────────────────────────────────────────────────────
# 6. EVALUATION METRICS (TEST SET ONLY)
# ──────────────────────────────────────────────────────────────────────────────

metrics = {
    'accuracy': accuracy_score(y_test, y_pred),
    'precision': precision_score(y_test, y_pred, zero_division=0),
    'recall': recall_score(y_test, y_pred, zero_division=0),
    'f1': f1_score(y_test, y_pred, zero_division=0),
    'auc': roc_auc_score(y_test, y_proba)
}

logger.info(f"--- Results for {RUN_NAME} ---")
for k, v in metrics.items():
    logger.info(f"{k.capitalize()}: {v:.3f}")

importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': clf.feature_importances_
}).sort_values('importance', ascending=False)

# ──────────────────────────────────────────────────────────────────────────────
# 7. FEATURE STABILITY ANALYSIS (LOOP)
# ──────────────────────────────────────────────────────────────────────────────

logger.info(f"Computing feature stability ({N_RUNS_STABILITY} runs)...")

feature_counts = {name: 0 for name in feature_names}
all_importances_list = []

for i in range(N_RUNS_STABILITY):
    X_train_stab, X_test_stab, y_train_stab, y_test_stab = train_test_split(
        X, y_encoded, test_size=0.2, stratify=y_encoded,
        random_state=RANDOM_STATE + i
    )
    
    clf_stab = XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        random_state=RANDOM_STATE + i,
        n_jobs=1,
        tree_method=tree_method,
        device=device,
        verbosity=0,
        **best_params
    )
    
    clf_stab.fit(X_train_stab, y_train_stab)
    
    imp_df_run = pd.DataFrame({
        'feature': feature_names,
        'importance': clf_stab.feature_importances_,
        'run': i
    }).sort_values('importance', ascending=False).head(TOP_K_FEATURES)
    
    all_importances_list.append(imp_df_run)
    
    for feature in imp_df_run['feature']:
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

logger.info("Stability analysis complete.")

# ──────────────────────────────────────────────────────────────────────────────
# 8. VISUALIZATION
# ──────────────────────────────────────────────────────────────────────────────

logger.info("Generating plots...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle(f'{RUN_NAME} - SOTA Analysis Results', fontsize=16, fontweight='bold')

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
           xticklabels=le.classes_,
           yticklabels=le.classes_, ax=axes[0,0])
axes[0,0].set_title('Confusion Matrix')
axes[0,0].set_ylabel('True Label')
axes[0,0].set_xlabel('Predicted Label')

top_features = importance_df.head(10)
axes[0,1].barh(range(len(top_features)), top_features['importance'])
axes[0,1].set_yticks(range(len(top_features)))
labels = [f[:30] + '...' if len(f) > 30 else f for f in top_features['feature']]
axes[0,1].set_yticklabels(labels, fontsize=8)
axes[0,1].set_xlabel('XGBoost Importance')
axes[0,1].set_title('Top 10 Most Important Features')
axes[0,1].invert_yaxis()

stable_features = stability_df.head(10)
bars = axes[1,0].bar(range(len(stable_features)), stable_features['stability_score'])
axes[1,0].set_xlabel('Feature Rank')
axes[1,0].set_ylabel('Stability Score')
axes[1,0].set_title('Top 10 Most Stable Features')
axes[1,0].set_xticks(range(len(stable_features)))
axes[1,0].set_xticklabels([f"F{i+1}" for i in range(len(stable_features))])

for bar, score in zip(bars, stable_features['stability_score']):
    bar.set_color(plt.cm.viridis(score))

metric_names = list(metrics.keys())
metric_values = list(metrics.values())
bars = axes[1,1].bar(metric_names, metric_values, color='skyblue', alpha=0.8)
axes[1,1].set_ylabel('Score')
axes[1,1].set_title('Classification Metrics')
axes[1,1].set_ylim(0, 1.1) 

for bar, value in zip(bars, metric_values):
    axes[1,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                  f'{value:.3f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()

save_path = FIGURES_DIR / f'{RUN_NAME}_analysis.png'
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.show()
logger.info(f"Plot saved to {save_path}")

# ──────────────────────────────────────────────────────────────────────────────
# 9. SAVE TABLES
# ──────────────────────────────────────────────────────────────────────────────

stability_df.head(20).to_csv(TABLES_DIR / f'{RUN_NAME}_stability.csv', index=False)
importance_df.head(20).to_csv(TABLES_DIR / f'{RUN_NAME}_importance.csv', index=False)

best_params_df = pd.DataFrame(list(best_params.items()), columns=['Parameter', 'Value'])
best_params_df.to_csv(TABLES_DIR / f'{RUN_NAME}_best_params.csv', index=False)

logger.info("Analysis complete. Tables saved.")

# ──────────────────────────────────────────────────────────────────────────────
# 10. FINAL RESULT OBJECT
# ──────────────────────────────────────────────────────────────────────────────

results = {
    'run_name': RUN_NAME,
    'model': clf,
    'best_params': best_params,
    'search_results': random_search.cv_results_,
    'metrics': metrics,
    'importance': importance_df,
    'stability': stability_df,
    'label_encoder': le,
    'test_data': (X_test, y_test, y_pred)
}

print("\n✅ SOTA Analysis Finished Successfully.")
print(f" Best Parameters: {best_params}")
print(f" Metrics: {metrics}")