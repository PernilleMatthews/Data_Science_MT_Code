# %%
import pandas as pd
import matplotlib.pyplot as plt

# Own Modules
import auxiliary as aux
import data_auxiliary as data_aux

# Matplotlib Global Settings
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

# %%
""" Provide column names for the dataframe """
columns = ["variance", "skewness", "kurtosis", "entropy", "y"]
features = ["Variance",  "Skewness", "Kurtosis", "Entropy"]
classes = ["Real", "Forged"]

""" Load the data """
df = pd.read_csv('datasets/data_banknote_authentication.txt', header=None, names=columns)
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = data_aux.train_test_split(data_aux.standardise(X), y, test_size=0.3)


#%%
""" Create model and evaluate"""
model = aux.random_forest(X_train, y_train, 100, 2, 42)
y_pred = aux.predict_evaluate_model(model, X_test, y_test)

""" Create SHAP values and separate into df for each class """
shap_values = aux.calc_shap_values(model, X_test)

# SHAP values for X_test
sv = shap_values.copy()

# Forged
sv_forged = pd.DataFrame(sv[1], columns=['petal_length_shap', 'petal_width_shap', 'sepal_length_shap', 'sepal_width_shap'])

#%%
""" Create PCP for data and explanation space """
pcp_ds = aux.create_pcp(X_test, y_pred, features, classes, title="Banknote Authentication Data Space", outpath="images/pcp_banknote_data_space.png")
pcp_setosa = aux.create_pcp(sv_forged, y_pred, features, classes, title="Banknote Authentication Explanation Space Towards Class Forged", outpath="images/pcp_banknote_expl_setosa.png")

#%%
""" HDBSCAN Testing """
# HDBSCAN_val = aux.hdbscan_validation(X_test)      # Best 11, 4
clusterer =  aux.hdbscan_clustering(X_test, 11, 4)
tsne_viz = aux.tsne_visualization(X_test, clusterer, "t-SNE over HDBSCAN clustering for Data Space")


#%%
""" Build the K-Means classifier and print metrics """
K, inertias, silhouettes = aux.k_means_gridsearch(X_test)
plot_silhouettes = aux.plot_silhouettes(K, silhouettes, title="Silhouettes for K-Means Clustering in Data Space")

