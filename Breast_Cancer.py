# %%
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer

# Own Modules
import auxiliary as aux
import data_auxiliary as data_aux

# Matplotlib Global Settings
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

""" Load the data """
bc_data = load_breast_cancer(as_frame=True)
features = bc_data.data.columns
classes = ["Malignant", "Benign"]

# columns = ["variance", "skewness", "kurtosis", "entropy", "y"]
df = pd.DataFrame(bc_data.data, columns=bc_data.feature_names)
df['y'] = bc_data.target
X = df.drop('y', axis=1)
y = df['y']
X_train, X_test, y_train, y_test = data_aux.train_test_split(data_aux.standardise(X), y, test_size=0.33)

#%%
""" Create model and evaluate"""
model = aux.random_forest(X_train, y_train, 100, 2, 42)
y_pred = aux.predict_evaluate_model(model, X_test, y_test)

""" Create SHAP values and separate into df for each class """
shap_values = aux.calc_shap_values(model, X_test, kernel_explainer=False)

# SHAP values for X_test
sv = shap_values.copy()

# Benign
sv_benign = pd.DataFrame(sv[1], columns=features)

#%%
""" Create PCP for data and explanation space """
pcp_ds = aux.create_pcp(X_test, y_pred, features, classes, title="Breast Cancer Data Space",
                        outpath="images/pcp_bc_data_space.png", label_90=True)
pcp_setosa = aux.create_pcp(sv_benign, y_pred, features, classes,
                            title="Breast Cancer Explanation Space Towards Class Forged", label_90=True,
                            outpath="images/pcp_bc_expl_setosa.png")

#%%
""" HDBSCAN Testing """
# HDBSCAN_val = aux.hdbscan_validation(X_test)      # Best 11, 4
clusterer =  aux.hdbscan_clustering(X_test, 11, 4)
tsne_viz = aux.tsne_visualization(X_test, clusterer, "t-SNE over HDBSCAN clustering for Data Space")
tsne_viz = aux.tsne_visualization(sv_benign, clusterer, "t-SNE over HDBSCAN clustering for Explanation Space")

#%%
""" Build the K-Means classifier and print metrics """
K, inertias, silhouettes = aux.k_means_gridsearch(X_test)
plot_silhouettes = aux.plot_silhouettes(K, silhouettes, title="Silhouettes for K-Means Clustering in Data Space")

