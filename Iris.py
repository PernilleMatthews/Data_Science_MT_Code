#%%
import pandas as pd
from sklearn import datasets
import matplotlib.pyplot as plt

# Own Modules
import auxiliary as aux
import data_auxiliary as data_aux

# Matplotlib Global Settings
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False


""" Provide column names for the dataframe """
classes = ['Iris-Setosa', 'Iris-Versicolor', 'Iris-Virginica'] # Classes: [0, 1, 2]
columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

""" Load the data """
iris = datasets.load_iris(as_frame=True)
X = pd.DataFrame(iris.data)
y = pd.DataFrame(iris.target)
y.columns = ['y']
X_train, X_test, y_train, y_test = data_aux.train_test_split(data_aux.standardise(X), y, test_size=0.33)

""" Create model and evaluate"""
model = aux.random_forest(X_train, y_train, 100, 2, 42)
y_pred = aux.predict_evaluate_model(model, X_test, y_test)

""" Create SHAP values and separate into df for each class """
shap_values = aux.calc_shap_values(model, X_test)

# SHAP values for X_test
sv = shap_values.copy()

# Setosa
sv_0 = pd.DataFrame(sv[0], columns=['petal_length_shap', 'petal_width_shap', 'sepal_length_shap', 'sepal_width_shap', ])

# Veriscolor
sv_1 = pd.DataFrame(sv[1],  columns=['sepal_length_shap', 'sepal_width_shap', 'petal_length_shap', 'petal_width_shap'])

# Virginica
sv_2 = pd.DataFrame(sv[2], columns=['sepal_length_shap', 'sepal_width_shap', 'petal_length_shap', 'petal_width_shap'])

#%%
""" Create PCP for data and explanation space """
pcp_ds = aux.create_pcp(X_test, y_pred, columns, classes, title="Iris Data Space", outpath="images/pcp_iris_data_space.png")
pcp_setosa = aux.create_pcp(sv_0, y_pred, columns, classes, title="Iris Explanation Space Towards Class Setosa", outpath="images/pcp_iris_expl_setosa.png")
pcp_veriscolor = aux.create_pcp(sv_1, y_pred, columns, classes, title="Iris Explanation Space Towards Class Veriscolor", outpath="images/pcp_iris_expl_veriscolor.png")
pcp_virginica = aux.create_pcp(sv_2, y_pred, columns, classes, title="Iris Explanation Space Towards Class Virginica", outpath="images/pcp_iris_expl_virginica.png")


#%%
""" HDBSCAN Testing """
# HDBSCAN_val = aux.hdbscan_validation(X_test)      # Best 5, 1
clusterer =  aux.hdbscan_clustering(X_test, 5, 1)
tsne_viz = aux.tsne_visualization(X_test, clusterer, "t-SNE over HDBSCAN clustering for Data Space")
tsne_viz_es = aux.tsne_visualization(sv_1, clusterer, "t-SNE over HDBSCAN clustering for Explanation Space")


#%%
""" Build the K-Means classifier and print metrics """
K, inertias, silhouettes = aux.k_means_gridsearch(X_test)
plot_silhouettes = aux.plot_silhouettes(K, silhouettes, title="Silhouettes for K-Means Clustering in Data Space")






