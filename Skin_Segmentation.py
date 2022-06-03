import pandas as pd
import matplotlib.pyplot as plt

# Own Modules
import auxiliary as aux
import data_auxiliary as data_aux

# Matplotlib Global Settings
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

""" Provide column names for the dataframe """
columns = ["B",  "G", "R", "y"]
features = ["B", "G", "R"]
classes = ["Skin", "Non-Skin"]

""" Load the data """
df = pd.read_csv("datasets/Skin_NonSkin.txt", delimiter = "\t", names=columns)
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
y.columns = ['y']
X_train, X_test, y_train, y_test = data_aux.train_test_split(data_aux.standardise(X), y, test_size=0.33)

""" Create model and evaluate"""
model = aux.random_forest(X_train, y_train, 100, 2, 42)
y_pred = aux.predict_evaluate_model(model, X_test, y_test)

""" Create SHAP values and separate into df for each class """
shap_values = aux.calc_shap_values(model, X_test, kernel_explainer=False)

sv = shap_values.copy()
sv_no_skin = pd.DataFrame(sv[1], columns=features)

""" Create PCP for data and explanation space """
pcp_ds = aux.create_pcp(X_test, y_pred, features, classes, title="Skin Segmentation Data Space", outpath="images/pcp .png")
pcp_setosa = aux.create_pcp(sv_no_skin, y_pred, features, classes, title="Skin Segmentation Explanation Space Towards Class No Skin", outpath="images/pcp_iris_expl_setosa.png")

""" HDBSCAN Testing """
# HDBSCAN_val = aux.hdbscan_validation(X_test)      # Best 11, 4
clusterer =  aux.hdbscan_clustering(X_test, 11, 4)
tsne_viz = aux.tsne_visualization(X_test, clusterer, "t-SNE over HDBSCAN clustering for Data Space")

""" Build the K-Means classifier and print metrics """
K, inertias, silhouettes = aux.k_means_gridsearch(X_test)
plot_silhouettes = aux.plot_silhouettes(K, silhouettes, title="Silhouettes for K-Means Clustering in Data Space")
