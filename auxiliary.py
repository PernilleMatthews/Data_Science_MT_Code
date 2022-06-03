import shap
import hdbscan
import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from yellowbrick.features import ParallelCoordinates
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, silhouette_score as sc



def random_forest(X_train, y_train, n_estimators=100, max_depth=2,random_state=42):
    rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
    rf.fit(X_train, y_train)
    return rf


def predict_evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print('Accuracy: ', accuracy_score(y_test, y_pred))
    print('Confusion Matrix: \n', confusion_matrix(y_test, y_pred))
    print('Classification Report: \n', classification_report(y_test, y_pred))
    return y_pred


def calc_shap_values(model, X_test, kernel_explainer=True):
    if kernel_explainer:
        explainer = shap.KernelExplainer(model.predict_proba, X_test)
    else:
        explainer = shap.TreeExplainer(model)

    print(explainer.expected_value)
    shap_values = explainer.shap_values(X_test)
    return shap_values


def create_pcp(X, y, features, classes, title="default", show=True, label_90=False, outpath="default"):
    ylimits = (X.min().min(), X.max().max())

    visualizer = ParallelCoordinates(
        classes=classes, features=features, fast=False,
        alpha=.40, title=title, size=(1600, 700)
    )

    # Fit the visualizer and display it
    visualizer.fit_transform(X, y)
    visualizer.finalize()
    visualizer.ax.set_ylim(ylimits[0], ylimits[1])
    visualizer.ax.tick_params(labelsize=16)                 # change size of tick labels
    visualizer.ax.title.set_fontsize(25)                    # change size of title

    for text in visualizer.ax.legend_.texts:                # change size of legend texts
        text.set_fontsize(18)
    if label_90:
        visualizer.ax.tick_params(axis='x', labelrotation=90)
    visualizer.fig.tight_layout()                           # fit all texts nicely into the surrounding figure
    if show:
        visualizer.show()
    else:
        visualizer.fig.savefig(outpath)


def hdbscan_validation(df):
    rel_val_list = {}

    # Loop to test parameters for HDBSCAN
    for min_cluster_size in range(2, 15):
        for min_samples in range(1, 15):
            clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples,
                                        metric="euclidean", gen_min_span_tree=True)
            clusterer.fit(df)
            c_labels = pd.DataFrame(clusterer.labels_)

            rel_val = clusterer.relative_validity_
            rel_val_list[rel_val] = [min_cluster_size, min_samples]

        # Print best relva_val and its parameters
        print("\n")
        print(rel_val_list)
        print(max(rel_val_list))
        print(rel_val_list[max(rel_val_list)])


def hdbscan_clustering(df, min_cluster_size, min_samples):
    return hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, gen_min_span_tree=True,
                           prediction_data=True).fit(df)


def tsne_visualization(X_test, clusterer, title="default", outpath="default", save=False):
    color_palette = sns.color_palette('Paired', 12)
    cluster_colors = [color_palette[x] if x >= 0
                      else (0.5, 0.5, 0.5)
                      for x in clusterer.labels_]
    cluster_member_colors = [sns.desaturate(x, p) for x, p in
                             zip(cluster_colors, clusterer.probabilities_)]

    projection = TSNE(random_state=42).fit_transform(X_test)
    plt.scatter(*projection.T, s=50, linewidth=0, c=cluster_member_colors, alpha=0.8)

    plt.title(title, fontsize=16)
    plt.grid(False)

    print(pd.DataFrame(clusterer.labels_).value_counts())
    print(len(pd.DataFrame(clusterer.labels_).value_counts()))
    print(round(clusterer.relative_validity_, 3))

    if save:
        plt.savefig(outpath)
    else:
        plt.show()



def cluster_purity(combined_df):
    """ Compute Purity for a cluster - not whole clustering"""
    print(len(combined_df))
    print(combined_df.value_counts())
    purity = ((max(combined_df.y.value_counts())/len(combined_df)))
    print(f"Purity: {round(purity, 3)}")
    return round(purity, 3)


def k_means_gridsearch(df):
    """
    This function performs a grid search for the optimal number of clusters for K-Means clustering.
    :param df:  Dataframe with the data to be clustered.
    :return:    The optimal number of cluster, the inertia and the silhouette score.
    """
    K = range(1, 10)
    inertias = []
    # distortions = []
    silhouettes = []
    for k in K:
        kmeans = KMeans(k, random_state=42).fit(df)
        inertias.append(kmeans.inertia_)
        if k > 1:
            silhouettes.append(sc(df, kmeans.labels_, metric='euclidean'))

    return K, inertias, silhouettes


def plot_silhouettes(K, silhouettes, title="default"):
    """
    This function plots the silhouette plot for the K-Means clustering.
    :return:    The silhouette plot.
    """
    plt.plot(K[1:], silhouettes)
    plt.title(title)
    plt.show()

def kmeans_clustering(X_train, X_test, y_train, y_test, n_clusters, title="default", outpath="default", save=False):
    pass