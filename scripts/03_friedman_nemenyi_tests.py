import numpy as np
import pandas as pd

from scipy import stats
import scikit_posthocs as sp

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.colors as mcolors

resultados = pd.read_csv("../resultados/default/resumo_resultado_default.csv", sep = ';')

# TODO: criar módulos para o teste de friedman e nemenyi para otimizar o processamento do código, visto que o mesmo é repetido para cada análise realizada.

# ============================================================================
# ANÁLISE DE TÉCNICAS DE IMPUTAÇÃO
#
#   - filtra arquivos que não possuem valores ausentes (None)
#   - salvo valores da métrica de interesse (f1 score, auc ou balanced accuracy) para cada técnica de imputação
#   - aplica teste de Friedman para verificar se existe diferença significativa entre as técnicas de imputação
#   - se houver diferença significativa, aplica teste de Nemenyi para identificar quais técnicas possuem diferença significativa entre si
#
# =============================================================================

imputation_approaches = {"imputer_mean": [], "set_ela_error": [], "remove_missing_values": [], "knn_imputer": []}

# substituir "media_acc_balanceada" por "media_f1_score" para análise de f1 score ou "media_auc" para análise de auc
lista = [(resultados.loc[i]["arquivo"], resultados.loc[i]["media_acc_balanceada"]) for i, f in enumerate(resultados.values) if not "None" in resultados.loc[i]["arquivo"]]

for l in lista:
    for f in imputation_approaches:
        if f in l[0]:
            imputation_approaches[f].append(l[1])

teste_friedman = stats.friedmanchisquare(imputation_approaches["imputer_mean"], imputation_approaches["knn_imputer"], imputation_approaches["remove_missing_values"], imputation_approaches["set_ela_error"])

print("Teste de Friedman: ", teste_friedman)

data = np.array([imputation_approaches["imputer_mean"], imputation_approaches["knn_imputer"], imputation_approaches["remove_missing_values"], imputation_approaches["set_ela_error"]])

nemenyi = sp.posthoc_nemenyi_friedman(data.T)

buffer_imputation = []
for imputation in imputation_approaches:
    buffer_imputation.append(np.mean(imputation_approaches[imputation]))

teste = pd.DataFrame(imputation_approaches)

ranks = teste.rank(axis = 1, ascending = False)
vg_rank = ranks.mean()

nemenyi.columns = ["imputer_mean", "knn_imputer", "remove_missing_values", "set_ela_error"]
nemenyi.index = ["imputer_mean", "knn_imputer", "remove_missing_values", "set_ela_error"]

plt.figure(figsize=(10, 2), dpi=300)

sp.critical_difference_diagram(vg_rank, nemenyi, label_fmt_left='{label} [{rank:.2f}]  ',
    label_fmt_right='  [{rank:.2f}] {label}', color_palette = sns.dark_palette("#000"))

# substituir "acc_balanced" por "f1_score" para análise de f1 score ou "auc" para análise de auc
plt.savefig('../resultados/default/plots/acc_balanced_critical_difference_diagram_imputation_approach.png', dpi=300, bbox_inches='tight')
plt.tight_layout()
plt.show()

# ============================================================================
# ANÁLISE DE THRESHOLD DE CORRELAÇÃO
#
#   - filtra arquivos que possuem a melhor técnica de imputação encontrada na análise anterior
#   - salvo valores da métrica de interesse (f1 score, auc ou balanced accuracy) para cada limiar de correlação aplicado
#   - aplica teste de Friedman para verificar se existe diferença significativa entre os limiares de correlação
#   - se houver diferença significativa, aplica teste de Nemenyi para identificar quais thresholds possuem diferença significativa entre si
#
# =============================================================================

lista_arquivos_flacco_melhor_tecnica_imputacao = [resultados.loc[i]["arquivo"] for i, f in enumerate(resultados.values) if "ela_features_flacco" in resultados.loc[i]["arquivo"] and not "set_ela_error" in resultados.loc[i]["arquivo"]]

lista = [(resultados.loc[i]["arquivo"], resultados.loc[i]["media_acc_balanceada"]) for i, f in enumerate(resultados.values) if not resultados.loc[i]["arquivo"] in lista_arquivos_flacco_melhor_tecnica_imputacao]

corr_thresholds = {"0_8.": [], "0_85": [], "0_9.": [], "0_95": []}

for l in lista:
    for f in corr_thresholds:
        if f in l[0]:
            corr_thresholds[f].append(l[1])

teste_friedman = stats.friedmanchisquare(corr_thresholds["0_8."], corr_thresholds["0_85"], corr_thresholds["0_9."], corr_thresholds["0_95"])

print("Teste de Friedman: ", teste_friedman)

buffer_corr = []
for corr in corr_thresholds:
    buffer_corr.append(np.mean(corr_thresholds[corr]))

data = np.array([corr_thresholds["0_8."], corr_thresholds["0_85"], corr_thresholds["0_9."], corr_thresholds["0_95"]])

nemenyi = sp.posthoc_nemenyi_friedman(data.T, y_col='values', group_col='groups')

nemenyi.columns = ["0.8", "0.85", "0.9", "0.95"]
nemenyi.index = ["0.8", "0.85", "0.9", "0.95"]

teste = pd.DataFrame(corr_thresholds)
teste.columns = ["0.8", "0.85", "0.9", "0.95"]

ranks = teste.rank(axis = 1, ascending = False)
vg_rank = ranks.mean()

plt.figure(figsize=(10, 2), dpi=300)
sp.critical_difference_diagram(vg_rank, nemenyi, label_fmt_left='{label} [{rank:.2f}]  ',
    label_fmt_right='  [{rank:.2f}] {label}', color_palette = sns.dark_palette("#000"))

# substituir "acc_balanced" por "f1_score" para análise de f1 score ou "auc" para análise de auc
plt.savefig('../resultados/default/plots/acc_balanced_critical_difference_diagram_corr_thresholds.png', dpi=300, bbox_inches='tight')
plt.tight_layout()
plt.show()

cmap = ['1', '#fb6a4a',  '#08306b',  '#4292c6', '#c6dbef']
heatmap_args = {'cmap': cmap, 'linewidths': 0.25, 'linecolor': '0.5', 'clip_on': False, 'square': True, 'cbar_ax_bbox': [0.80, 0.35, 0.04, 0.3]}
sp.sign_plot(nemenyi, **heatmap_args)

# ============================================================================
# ANÁLISE DE ALGORITMOS DE MACHINE LEARNING
#
#   - filtra arquivos que possuem a melhor técnica de imputação e o melhor threshold encontrados nas análises anteriores
#   - salvo valores da métrica de interesse (f1 score, auc ou balanced accuracy) para cada algoritmo de ML aplicado
#   - aplica teste de Friedman para verificar se existe diferença significativa entre os algoritmos
#   - se houver diferença significativa, aplica teste de Nemenyi para identificar quais algoritmos possuem diferença significativa entre si
#
# =============================================================================

lista = [(resultados.loc[i]["arquivo"], resultados.loc[i]["media_acc_balanceada"]) for i, f in enumerate(resultados.values) if not resultados.loc[i]["arquivo"] in lista_arquivos_flacco_melhor_tecnica_imputacao and "0_8." in resultados.loc[i]["arquivo"]]
algorithms = {"NB": [], "DT":[], "KNN": [], "RF": [], "SVM_RBF": [], "SVM_LIN": [], "LogisticRegression": [], "XGBoost": []}

for l in lista:
    for f in algorithms:
        if f in l[0]:
            algorithms[f].append(l[1])

teste_friedman = stats.friedmanchisquare(algorithms["NB"], algorithms["DT"], algorithms["KNN"], algorithms["RF"], algorithms["SVM_RBF"], algorithms["SVM_LIN"], algorithms["LogisticRegression"], algorithms["XGBoost"])

print("Teste de Friedman: ", teste_friedman)

data = np.array([algorithms["NB"], algorithms["DT"], algorithms["KNN"], algorithms["RF"], algorithms["SVM_RBF"], algorithms["SVM_LIN"], algorithms["LogisticRegression"], algorithms["XGBoost"]])

nemenyi = sp.posthoc_nemenyi_friedman(data.T)

nemenyi.columns = ["NB", "DT", "KNN", "RF", "SVM_RBF", "SVM_LIN", "RL", "XGB"]
nemenyi.index   = ["NB", "DT", "KNN", "RF", "SVM_RBF", "SVM_LIN", "RL", "XGB"]

buffer_al = []
for al in algorithms:
    buffer_al.append(np.mean(algorithms[al]))

teste = pd.DataFrame(algorithms)
teste.columns = ["NB", "DT", "KNN", "RF", "SVM_RBF", "SVM_LIN", "RL", "XGB"]
ranks = teste.rank(axis = 1, ascending = False)
vg_rank = ranks.mean()

plt.figure(figsize=(10, 2), dpi=300)
sp.critical_difference_diagram(vg_rank, nemenyi, label_fmt_left='{label} [{rank:.2f}]', label_fmt_right='  [{rank:.2f}] {label}', color_palette = ["#000000"] * len(vg_rank))
plt.savefig('../resultados/default/plots/acc_balanced_critical_difference_diagram_algorithms.png', dpi=300, bbox_inches='tight')
plt.tight_layout()
plt.show()

# ============================================================================
# ANÁLISE DE META-DATASETS
#
#   - filtra arquivos que possuem a melhor técnica de imputação, threshold e algoritmo de ML encontrados nas análises anteriores
#   - salvo valores da métrica de interesse (f1 score, auc ou balanced accuracy) para cada meta-dataset aplicado
#   - aplica teste de Friedman para verificar se existe diferença significativa entre os meta-datasets
#   - se houver diferença significativa, aplica teste de Nemenyi para identificar quais meta-datasets possuem diferença significativa entre si
#
# =============================================================================

lista = [(resultados.loc[i]["arquivo"], resultados.loc[i]["media_acc_balanceada"]) for i, f in enumerate(resultados.values) if not resultados.loc[i]["arquivo"] in lista_arquivos_flacco_melhor_tecnica_imputacao and "0_8." in resultados.loc[i]["arquivo"] and "XGB" in resultados.loc[i]["arquivo"]]

teste_friedman = stats.friedmanchisquare(lista[0][1], lista[1][1], lista[2][1])

print("Teste de Friedman: ", teste_friedman)

metadatasets = {'ela_features_flacco': [], 'classif_svm_169d_95_average': [], 'classif_svm_ela_features_flacco': []}

for r in range(len(resultados)):
    for md in metadatasets:
        if md in resultados.loc[r]["arquivo"] and ("None" in resultados.loc[r]["arquivo"] or "remove_missing_values" in resultados.loc[r]["arquivo"]):
            if md == 'ela_features_flacco' and 'classif_svm' in resultados.loc[r]["arquivo"]:
                pass
            else:
                metadatasets[md].append(resultados.loc[r]["media_acc_balanceada"])
    
teste_friedman = stats.friedmanchisquare(metadatasets['ela_features_flacco'], metadatasets['classif_svm_169d_95_average'], metadatasets['classif_svm_ela_features_flacco'])

data = np.array([metadatasets['ela_features_flacco'], metadatasets['classif_svm_169d_95_average'], metadatasets['classif_svm_ela_features_flacco']])

nemenyi = sp.posthoc_nemenyi_friedman(data.T)

nemenyi.columns = ["ela_features_flacco", "classif_svm_169d_95_average", "classif_svm_ela_features_flacco"]
nemenyi.index   = ["ela_features_flacco", "classif_svm_169d_95_average", "classif_svm_ela_features_flacco"]

teste = pd.DataFrame(metadatasets)
ranks = teste.rank(axis = 1, ascending = False)
vg_rank = ranks.mean()

plt.figure(figsize=(10, 2), dpi=300)
sp.critical_difference_diagram(vg_rank, nemenyi, label_fmt_left='{label} [{rank:.2f}]', label_fmt_right='  [{rank:.2f}] {label}', color_palette = ["#000000"] * len(vg_rank))
plt.savefig('../resultados/default/plots/acc_balanced_critical_difference_diagram_metadatasets.png', dpi=300, bbox_inches='tight')
plt.tight_layout()
plt.show()


