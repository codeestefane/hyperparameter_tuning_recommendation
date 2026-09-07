# bibliotecas básicas para manipulação de dados e valores numéricos
import pandas as pd
import numpy as np
import random

from sklearn.base import BaseEstimator, TransformerMixin

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

        # se existir alguma feature com valor faltante no teste que não foi identificada no treinamento, substitui os valores faltantes por -3 (valor selecionada após análise dos datasets a serem processados)
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