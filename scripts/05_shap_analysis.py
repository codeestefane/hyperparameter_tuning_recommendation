import pandas as pd
import numpy as np
import random
from tqdm import tqdm

from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from scripts.modules.pipeline import create_preprocessor
from scripts.modules.preprocessing import RemoveMissingValues

import shap

seeds = [3, 5, 7, 13, 27, 35, 42, 66, 72, 111]

metadataset = 'ela_features_flacco'
approach_missing_values = {"remove_missing_values": RemoveMissingValues()}
approach = "remove_missing_values"
threshold = 0.8
algorithms = {"SVM_RBF": SVC(kernel="rbf", probability=True)}

data = pd.read_csv("../datasets/" + metadataset + ".csv")

X = data.iloc[:, 1:-1].copy()
y = data.iloc[:, -1].copy()
y = y.map({'Defaults': 0, 'Tuning': 1}).astype(int)

bool_missing_value = X.isna().any().any()

preprocessor = create_preprocessor(
    bool_missing_value, threshold, approach, approach_missing_values
)

X_prep_df = preprocessor.fit_transform(X, y)

modelo = SVC(kernel="rbf", probability=True, random_state=42)
modelo.fit(X_prep_df, y)

explainer = shap.KernelExplainer(modelo.predict_proba, X_prep_df)
shap_values = explainer(X_prep_df)

shap_fold = shap_values[:, :, 1].values

features_shap_analysis = [i for i in X.columns.tolist() if not i in preprocessor.named_steps['constant_values'].constant_features and not i in preprocessor.named_steps['missing_values'].features_missing_values and not i in preprocessor.named_steps['correlated_features'].removed_corr_features]

import pandas as pd
import matplotlib.pyplot as plt

teste = pd.DataFrame(shap_fold, columns=features_shap_analysis)
teste.to_csv("../resultados/default/shap_values.csv", index=False)

shap.summary_plot(
    shap_fold,
    X_prep_df,
    feature_names = features_shap_analysis,
    show=False
)

plt.savefig("../resultados/default/plots/shap_summary.png", dpi=300, bbox_inches='tight')
plt.close()