import pandas as pd

dataframe = pd.read_csv("datasets/classif_svm_169d_95_average.csv")

dataframe_2 = pd.read_csv("datasets/ela_features_flacco.csv")

buffer = pd.DataFrame(columns = dataframe_2.columns)

for l, i in enumerate(dataframe["avail.datasets"].values):
    for k, j in enumerate(dataframe_2["File"].values):
        if i == j:
            buffer.loc[l] = dataframe_2.loc[k].values

dataframe_2.drop("File", axis = 1, inplace = True)

dataframe.drop("Class", axis = 1, inplace = True)

dataset_combinado = pd.concat([dataframe, dataframe_2], axis = 1)

dataset_combinado.to_csv("datasets/classif_svm_ela_features_flacco.csv", index = False)