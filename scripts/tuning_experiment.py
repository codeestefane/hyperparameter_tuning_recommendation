# %%
# bibliotecas básicas para manipulação de dados e valores numéricos
import pandas as pd
import numpy as np
import random

# algoritmos de Machine Learning
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

# bibliotecas de pipeline do scikit-learn, pré-processamento de dados e métricas de desempenho
from sklearn.preprocessing import MinMaxScaler, FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn import metrics

# bibliotecas de otimização
from skopt.space import Integer
from skopt.space import Real
from skopt.space import Categorical
from skopt.utils import use_named_args
from skopt import gp_minimize

# bibliotecas para manipulação de arquivos e controle de experimento
import os

from tqdm import tqdm

# sementes aleatórias utilizadas
seeds = [3, 5, 7, 13, 27, 35, 42, 66, 72, 111]

search_space = []

# %%
# datasets
files = ['ela_features_flacco', 'classif_svm_169d_95_average', 'classif_svm_ela_features_flacco']

# %%
# pré-processamento: remoção de features com valores constantes que não agregam informações significativas para o modelo
class RemoveConstantValues(BaseEstimator, TransformerMixin):
    def __init__(self):
        super().__init__()
        
        self.constant_features = []
    
    def fit(self, X, y = None):
        # identifica as features que possuem valores constantes
        self.constant_features = [i for i in X.columns if len(np.unique(list(X[i].values), return_counts = True)[0]) == 1]

        return self

    def transform(self, X, y = None):
        # remove as features que tem valores constantes e foram identificadas pelo estimador
        if self.constant_features:
            X = X.drop(self.constant_features, axis = 1) 

        return X

# %%
# pré-processamento: remoção de features correlacionadas, positivamente ou negativamente, que possuem menor correlação com a target
class RemoveCorrelatedFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, corr_threshold = 0.8):
        super().__init__()

        self.data = None

        self.corr_matrix = None
        self.corr_threshold = corr_threshold

        self.tuple_corr_features = []
        self.removed_corr_features = []
        
    def find_corr_features(self):
        self.tuple_corr_features = []

        # percorre a matriz de correlação e identifica os pares que possuem o coeficiente de correlação maior do que o valor de threshold
        for i in range(len(self.corr_matrix.values)):
            for j in range(len(self.corr_matrix.values[i])):
                if j > i:
                    if np.abs(self.corr_matrix.values[i][j]) >= self.corr_threshold:
                        # adiciona os pares identificados na lista de features correlacionadas 
                        self.tuple_corr_features.append((i, j))

    def remove_corr_features(self):
        for pair in self.tuple_corr_features:
            list_corr = [abs(self.corr_matrix.values[i][len(self.data.columns) - 1]) for i in pair]

            # identifica a feature da tupla correlacionada que possui menor correlação com a target e ainda não foi removida
            if list_corr[0] <= list_corr[1] and not self.data.columns[pair[0]] in self.removed_corr_features:
                self.removed_corr_features.append(self.data.columns[pair[0]])
            elif list_corr[1] < list_corr[0] and not self.data.columns[pair[1]] in self.removed_corr_features:
                self.removed_corr_features.append(self.data.columns[pair[1]])

        if self.removed_corr_features:
            # remove features correlacionadas 
            self.data.drop(self.removed_corr_features, axis = 1, inplace = True) 
    
    def fit(self, X, y):
        self.data = pd.concat([pd.DataFrame(X), pd.DataFrame(y)], axis = 1)

        # calcula a matriz de correlação usando o coeficiente de Pearson
        self.corr_matrix = self.data.corr()

        self.find_corr_features()

        self.remove_corr_features()
        
        return self
                
    def transform(self, X, y = None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        # percorre a lista de features correlacionadas a serem removidas
        if self.removed_corr_features:
            cols_to_drop = [c for c in self.removed_corr_features if c in X.columns]
            # remove features correlacionadas 
            X = X.drop(cols_to_drop, axis = 1) 

        return X   

# %%
# pré-processamento: tratamento de valores ausentes - remoção de features que apresentam valores faltantes (imputação)
class RemoveMissingValues(BaseEstimator, TransformerMixin):
    def __init__(self):
        super().__init__()

        self.features_missing_values = []
        
    def fit(self, X, y = None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
  
        self.features_missing_values = []

        # para cada feature de X
        for i in X.columns:
            # verifica se existe algum valor faltante
            if (X[i].isnull().sum() != 0):
                # se existir, salva o nome da feature na lista de controle
                self.features_missing_values.append(i)

        return self

    def transform(self, X, y = None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        # remove todas as features com missing values que foram identificadas
        X = X.drop(self.features_missing_values, axis = 1)

        # se existir alguma feature com valor faltante no teste que não foi identificada no treinamento, substitui os valores faltantes por -3
        X = X.fillna(-3)

        return X

# %%
# pré-processamento: tratamento de valores ausentes - criação de features que indicam a presença de valores faltantes por categoria de ELA
def identifyGroupMissingValues(X, y=None):
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    # identifica as colunas de features dos conjuntos "ela_level" e "ela_distr", respectivamente
    columns_ela_level = [c for c in X.columns if "ela_level" in str(c)]
    columns_ela_distr = [c for c in X.columns if "ela_distr" in str(c)]

    # verifica se existe algum valor faltante nas colunas que correspondem a cada conjunto de ELA 
    error_ela_level = X[columns_ela_level].isna().any(axis = 1).astype(int) if columns_ela_level else 0
    error_ela_distr = X[columns_ela_distr].isna().any(axis = 1).astype(int) if columns_ela_distr else 0

    # cria um dataFrame com as novas colunas que indicam se há valor faltante ou não nos conjuntos correspondentes; utiliza índices do dataFrame X original
    new_features = pd.DataFrame({
        "error_ela_level": error_ela_level,
        "error_ela_distr": error_ela_distr
    }, index = X.index)

    return new_features

# %%
# cria arquivos com os resultados obtidos por cada combinação (setup) do experimento
def write_csv_result(tipo_experimento, file, bool_missing_value, abordagem_missing_value, algoritmo, threshold, test_idx, resultados, f1_score, acc_balanceada, auc_score, seed, counter):
    # se não tinha valores faltantes no dataset
    if not bool_missing_value:
        # não foi utilizada nenhuma abordagem para tratar valores faltantes
        abordagem_missing_value = None

    os.makedirs('resultados/' + tipo_experimento, exist_ok = True)

    with open('resultados/' + tipo_experimento + '/' + file + '_' + str(abordagem_missing_value) + '_' + algoritmo + '_' + str(threshold).replace(".", "_") + '.csv', 'a') as f:
        # se estiver na primeira seed e na primeira iteração da validação cruzada
        if seed == seeds[0] and counter == 0:
            # escreve o cabeçalho do arquivo
            f.write("seed;iteracao_validacao_cruzada;indices;previsoes;f1_score;acuracia_balanceada;auc\n")

        # resultados obtidos por seed e iteração
        f.write(";".join([str(seed), str(counter), str(test_idx), str(resultados), str(f1_score), str(acc_balanceada), str(auc_score)]) + "\n")

# %%
def identify_analized_features(pipeline, bool_missing_value, approach, algorithm):
    # se a abordagem de tratamento de valores faltantes for a primeira e o pipeline estiver no primeiro algoritmo 
    # OBS: condição criada só para analisar as features com valores faltantes e não correlacionadas uma vez por seed e iteração da validação cruzada (pré-processamento não tem comportamento estocástico)
    if approach == "remove_missing_values" and algorithm == list(algorithms.keys()).pop(0):
        features_missing_values = []

        # identifica features que permaneceram após a remoção das features correlacionadas
        not_corr_features = list(pipeline.named_steps['preprocessing'].named_steps['correlated_features'].data.columns[0:-1])

        # se tiver valores faltantes no dataset
        if bool_missing_value:
            # identifica as features com valores faltantes
            features_missing_values = list(pipeline.named_steps['preprocessing'].named_steps['missing_values'].features_missing_values)

        # senão
        else:
            # atribui None
            features_missing_values = None

        return features_missing_values, not_corr_features

    return None, None
    

# %%
# cria arquivos com o resumo das features que possuem valores faltantes; também salva features não correlacionadas para identificar as features consideradas no treinamento de cada modelo 
def write_csv_analyzed_features(tipo_experimento, file, threshold, features_nao_correlacionadas, features_missing_values, bool_missing_value, seed, counter):
    # se tiver features não correlacionadas para analisar
    if features_nao_correlacionadas != None:
        os.makedirs('resultados/' + tipo_experimento + '/features/', exist_ok = True)

        with open('resultados/' + tipo_experimento + '/features/' + file + str(threshold).replace(".", "_") + '.csv', 'a') as f:
             # se estiver na primeira seed e na primeira iteração da validação cruzada
            if seed == seeds[0] and counter == 0:
                # escreve cabeçalho do arquivo
                f.write("seed;iteracao_validacao_cruzada;features_nao_correlacionadas;features_missing_values\n")

            # escreve resultado obtido
            f.write(";".join([str(seed), str(counter), str(features_nao_correlacionadas), str(features_missing_values)]) + "\n")

# %%
# abordagens de tratamento de valores faltantes aplicadas aos dados
approach_missing_values = {"remove_missing_values": RemoveMissingValues(), "imputer_mean": SimpleImputer(strategy = 'mean'), "knn_imputer": KNNImputer(), "set_ela_error": RemoveMissingValues()}

# thresholds do coeficiente de correlação
corr_threshold = [0.8, 0.85, 0.9, 0.95]

# algoritmos de Machine Learning utilizados no experimento
algorithms = {"DT": DecisionTreeClassifier(), "KNN": KNeighborsClassifier(), "RF": RandomForestClassifier(), "SVM_RBF": SVC(kernel = "rbf", probability = True), "SVM_LIN": SVC(kernel = 'linear', probability = True), "LogisticRegression": LogisticRegression(), "XGBoost": XGBClassifier()}

# %%
# métricas calculadas para análise de resultados
def calculate_metrics(y, predict, positive_proba):
    f1 = metrics.f1_score(y, predict)

    balanced_acc = metrics.balanced_accuracy_score(y, predict)
    
    fpr, tpr, thresholds = metrics.roc_curve(np.array(y), positive_proba)
    auc_score = metrics.auc(fpr, tpr)

    return f1, balanced_acc, auc_score

# %%
def create_preprocessor(bool_missing_value, threshold_corr, approach_missing_value = None):
    # se tiver valores faltantes
    if bool_missing_value:
        # cria um pipeline de preprocessamento considerando a etapa de tratamento de valores faltantes
        preprocessor = Pipeline([('constant_values', RemoveConstantValues()), ('missing_values', clone(approach_missing_values[approach_missing_value])), ('correlated_features', RemoveCorrelatedFeatures(threshold_corr)), ('scaler', MinMaxScaler())])

        # se a abordagem de valores faltantes for criar features que identifiquem erros em categorias específicas de ELA
        if approach_missing_value == "set_ela_error":
            # crie um pipeline com function transformer (permite adicionar colunas ao dataset)
            set_ela_error = Pipeline([
                ('create_ela_error_set', FunctionTransformer(identifyGroupMissingValues))
            ])

            # cria o pipeline de preprocessamento com feature union
            preprocessor = FeatureUnion([
                ('original', preprocessor),
                ('calculated', set_ela_error)
            ])
    # senão
    else:
        # cria pipeline de processamento sem etapa de tratamento de valores faltantes
        preprocessor = Pipeline([('constant_values', RemoveConstantValues()), ('correlated_features', RemoveCorrelatedFeatures(threshold_corr)), ('scaler', MinMaxScaler())])

    return preprocessor

# %%
def create_pipeline(preprocessor, algorithm, seed):
    estimator = clone(algorithms[algorithm])

    try: 
        # define a seed do algoritmo de machine learning se ele for estocástico
        estimator.set_params(random_state = seed)
    except:
        pass

    # cria pipeline com preprocessamento e estimador
    return Pipeline([('preprocessing', preprocessor), ('model', estimator)])

# %%
# espaços de busca de cada algoritmo
algorithms_search_space = {"SVM_RBF": [Real(2e-10, 2e10, 'log-uniform', name = 'C'), Real(2e-10, 2e10, 'log-uniform', name = 'gamma')], 
                           "SVM_LIN": [Real(2e-10, 2e10, 'log-uniform', name = 'C'), Real(2e-10, 2e10, 'log-uniform', name = 'gamma')],
                           "KNN": [Integer(1, 50, name = 'n_neighbors'), Categorical(['uniform', 'distance'], name = 'weights'), Categorical(['ball_tree', 'kd_tree', 'brute', 'auto'], name = 'algorithm')],
                           "RF": [Integer(1, 2000, name = 'n_estimators'), Integer(1, 20, name = 'max_depth'), Categorical(['gini', 'entropy', 'log_loss'], name = 'criterion')],
                           "LogisticRegression": [Real(2e-10, 2e10, 'log-uniform', name = 'C'), Integer(500, 2000, name = 'max_iter')],
                           "XGBoost": [Real(2e-10, 2e10, 'log-uniform', name = 'alpha'), Integer(1, 15, name = 'max_depth')],
                           "DT": [Categorical(['gini', 'entropy', 'log_loss'], name = 'criterion'), Integer(1, 20, name = 'max_depth')]
                           }

# %%
def set_params_algorithm(pipeline, params_result, algorithm):
    # define os parâmetros do modelo do pipeline de acordo com o algoritmo em execução
    if algorithm == "SVM_RBF" or algorithm == "SVM_LIN":
        pipeline.named_steps['model'].set_params(C = params_result[0], gamma = params_result[1])

    elif algorithm == "DT":
        pipeline.named_steps['model'].set_params(criterion = params_result[0], max_depth = params_result[1])

    elif algorithm == "RF":
        pipeline.named_steps['model'].set_params(n_estimators = params_result[0], max_depth = params_result[1], criterion = params_result[2])

    elif algorithm == "KNN":
        pipeline.named_steps['model'].set_params(n_neighbors = params_result[0], weights = params_result[1], algorithm = params_result[2])

    elif algorithm == "LogisticRegression":
        pipeline.named_steps['model'].set_params(C = params_result[0], max_iter = params_result[1])

    elif algorithm == "XGBoost":
        pipeline.named_steps['model'].set_params(alpha = params_result[0], max_depth = params_result[1])

    return pipeline

# %%
# factory function
def create_objective_fn(pipeline, search_space, X, y, cv):

    # define dimensão do espaço de busca do modelo a ser avaliado
    @use_named_args(search_space)
    def evaluate_model(**params):
        # define os parâmetros selecionados pelo SMBO no modelo a ser testado
        pipeline.named_steps['model'].set_params(**params)

        # faz validação cruzada e retorna array com o f1 score obtido em cada iteração
        scores = cross_val_score(pipeline, X, y, cv = cv, scoring = "f1")

        # retorna 1 - média dos scores (problema de minimização)
        return 1.0 - scores.mean()

    # retorna a função de avaliação do modelo
    return evaluate_model

# %%
def tuning_process(pipeline, search_space, X, y, cv_stratified, seed):
    objective_fn = create_objective_fn(pipeline, search_space, X, y, cv_stratified)

    # inicia o processo de otimização em cima da função que avalia o modelo e o espaço de busca
    result = gp_minimize(objective_fn, search_space, random_state = seed)

    return result

# %%
# para cada dataset
for f in files:
    data = pd.read_csv("datasets/" + f + ".csv")

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
                for seed in seeds:
                    random.seed(seed)
                    np.random.seed(seed)

                    preprocessor = create_preprocessor(bool_missing_value, threshold, approach)

                    pipeline = create_pipeline(preprocessor, algorithm, seed)

                    cv_stratified = StratifiedKFold(n_splits = 10, shuffle = True, random_state = seed)

                    # define espaço de busca atual
                    search_space = algorithms_search_space[algorithm]
                    
                    # percorre cada um dos conjuntos de treinamento e teste criados pelo StratifiedKFold
                    for counter, idx in tqdm(enumerate(cv_stratified.split(X, y)), desc = f"{f} | {algorithm} | thr = {threshold} | imputation = {approach} | seed = {seed}"):
                        X_train, X_test = X.iloc[list(idx[0])], X.iloc[list(idx[1])]
                        y_train, y_test = y.iloc[list(idx[0])], y.iloc[list(idx[1])]

                        cv_stratified_optimization = StratifiedKFold(n_splits = 10, shuffle = True, random_state = seed)

                        # aplica otimização de hiperparâmetros
                        optimize_result = tuning_process(pipeline, search_space, X_train, y_train, cv_stratified_optimization, seed)

                        # configura pipeline com hiperparâmetros do modelo otimizados
                        pipeline = set_params_algorithm(pipeline, list(optimize_result.x), algorithm)

                        pipeline.fit(X_train, y_train)
                        
                        y_predict = pipeline.predict(X_test)

                        y_proba = pipeline.predict_proba(X_test)
                                                
                        positive_proba = y_proba[:, 1]

                        features_missing_values, not_corr_features = identify_analized_features(pipeline, bool_missing_value, approach, algorithm)

                        f1, balanced_acc, auc_score = calculate_metrics(y_test, y_predict, positive_proba)

                        # mapeia os valores previstos para as labels originais da target
                        predict = pd.Series(y_predict).map({0: 'Defaults', 1: 'Tuning'}).astype(object)

                        # escreve resultados finais
                        write_csv_result("tuning", f, bool_missing_value, approach, algorithm, threshold, list(idx[1]), list(predict), f1, balanced_acc, auc_score, seed, counter)
                        write_csv_analyzed_features("tuning", f, threshold, not_corr_features, features_missing_values, bool_missing_value, seed, counter)

                # se o dataset não tiver valores faltantes
                if not bool_missing_value:
                    # não percorre todas as abordagens de tratamento de valores faltantes
                    break
                    


