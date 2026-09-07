# %%
# bibliotecas básicas para manipulação de dados e valores numéricos
import pandas as pd
import numpy as np
import random

# algoritmos de Machine Learning utilizados
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

# bibliotecas de pipeline do scikit-learn, pré-processamento de dados e métricas de desempenho
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.model_selection import StratifiedKFold

from modules.preprocessing import RemoveMissingValues
from modules.report_results import write_csv_result, write_csv_analyzed_features, identify_analized_features, calculate_metrics
from modules.pipeline import create_preprocessor, create_pipeline

# bibliotecas para manipulação de arquivos e controle de experimento
import os
from tqdm import tqdm

# sementes aleatórias utilizadas
seeds = [3, 5, 7, 13, 27, 35, 42, 66, 72, 111]

# %%
# meta-datasets
files = ['ela_features_flacco', 'classif_svm_169d_95_average', 'classif_svm_ela_features_flacco']
# %%
# abordagens de tratamento de valores faltantes aplicadas aos dados
approach_missing_values = {"remove_missing_values": RemoveMissingValues(), "imputer_mean": SimpleImputer(strategy = 'mean'), "knn_imputer": KNNImputer(), "set_ela_error": RemoveMissingValues()}

# thresholds do coeficiente de correlação
corr_threshold = [0.8, 0.85, 0.9, 0.95]

# algoritmos de Machine Learning utilizados no experimento
algorithms = {"NB": GaussianNB(), "DT": DecisionTreeClassifier(), "KNN": KNeighborsClassifier(), "RF": RandomForestClassifier(), "SVM_RBF": SVC(kernel = "rbf", probability = True), "SVM_LIN": SVC(kernel = 'linear', probability = True), "LogisticRegression": LogisticRegression(), "XGBoost": XGBClassifier()}

# %%
# para cada dataset
for f in files:
    data = pd.read_csv("../datasets/" + f + ".csv")

    # separa features de interesse (exclui coluna de identificadores)
    X = data.iloc[:, 1:-1].copy()

    # separa target
    y = data.iloc[:, -1].copy()

    # mapeia a classe negativa e positiva para facilitar o treinamento (XGBoost só aceita target com valores numéricos)
    y = y.map({'Defaults': 0, 'Tuning': 1}).astype(int)

    # verifica se existe valores faltantes no dataset
    bool_missing_value = X.isna().any().any()

    # para cada algoritmo de Machine Learning
    for algorithm in algorithms:
        # para cada threshold
        for threshold in corr_threshold:
            # para cada abordagem de tratamento de valores faltantes
            for approach in approach_missing_values: 
                # para cada seed em um conjunto de 10
                for seed in tqdm(seeds, desc = f"{f} | {algorithm} | thr = {threshold} | {approach}"):
                    random.seed(seed)
                    np.random.seed(seed)

                    preprocessor = create_preprocessor(bool_missing_value, threshold, approach, approach_missing_values)

                    pipeline = create_pipeline(preprocessor, algorithm, seed, algorithms)

                    cv_stratified = StratifiedKFold(n_splits = 10, shuffle = True, random_state = seed)

                    # percorre cada um dos conjuntos de treinamento e teste criados pelo StratifiedKFold
                    for counter, idx in enumerate(cv_stratified.split(X, y)):
                        X_train, X_test = X.iloc[list(idx[0])], X.iloc[list(idx[1])]
                        y_train, y_test = y.iloc[list(idx[0])], y.iloc[list(idx[1])]

                        pipeline.fit(X_train, y_train)
                        
                        y_predict = pipeline.predict(X_test)

                        y_proba = pipeline.predict_proba(X_test)
                        
                        positive_proba = y_proba[:, 1]

                        features_missing_values, not_corr_features = identify_analized_features(pipeline, bool_missing_value, approach, algorithm, algorithms)

                        f1, balanced_acc, auc_score = calculate_metrics(y_test, y_predict, positive_proba)

                        # mapeia os valores previstos para as labels originais da target
                        predict = pd.Series(y_predict).map({0: 'Defaults', 1: 'Tuning'}).astype(object)

                        # escreve resultados finais
                        write_csv_result("default", f, bool_missing_value, approach, algorithm, threshold, list(pd.Series(idx[1]).astype(int)), list(predict), f1, balanced_acc, auc_score, seed, counter, seeds)
                        write_csv_analyzed_features("default", f, threshold, not_corr_features, features_missing_values, seed, counter, seeds)

                # se o dataset não tiver valores faltantes
                if not bool_missing_value:
                    # não percorre todas as abordagens de tratamento de valores faltantes
                    break
                    
