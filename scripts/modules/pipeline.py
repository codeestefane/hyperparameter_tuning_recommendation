from sklearn.preprocessing import MinMaxScaler, FunctionTransformer
from sklearn.base import clone
from sklearn.pipeline import Pipeline, FeatureUnion

from .preprocessing import RemoveConstantValues, RemoveCorrelatedFeatures, RemoveMissingValues, identifyGroupMissingValues

# %%
def create_preprocessor(bool_missing_value, threshold_corr, approach_missing_value = None, approach_missing_values = None):
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
def create_pipeline(preprocessor, algorithm, seed, algorithms):
    estimator = clone(algorithms[algorithm])

    try: 
        # define a seed do algoritmo de machine learning se ele for estocástico
        estimator.set_params(random_state = seed)
    except:
        pass

    # cria pipeline com preprocessamento e estimador
    return Pipeline([('preprocessing', preprocessor), ('model', estimator)])