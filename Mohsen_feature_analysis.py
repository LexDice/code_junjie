import numpy as np
from classification import Classification
from sklearn.decomposition import PCA
from dwd.socp_dwd import DWD
from sklearn import metrics

N_POS_SAMPLES = 34
N_NEG_SAMPLES = 143

neg_features_data = np.load('features_resources/neg_features.npy')
pos_features_data = np.load('features_resources/pos_features.npy')

features_data = np.concatenate((pos_features_data, neg_features_data), axis=0)
labels = np.array([1] * N_POS_SAMPLES + [0] * N_NEG_SAMPLES)

# perform PCA
pca_comps = N_POS_SAMPLES + N_NEG_SAMPLES - 1
pca_model = PCA(n_components=pca_comps)
pca_features = pca_model.fit_transform(features_data)

aucs = []
for i in range(1000):
    X_train, X_test, y_train, y_test = Classification.partition(pca_features, labels, 0.2, i)

    dwd = DWD().fit(X_train, y_train)
    # euclid_X_test_trans = pca_model.transform(X_test)
    y_pred = dwd.decision_function(X_test)
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred, pos_label=1)

    roc_auc = metrics.auc(fpr, tpr)
    aucs.append(roc_auc)

print(np.mean(aucs))