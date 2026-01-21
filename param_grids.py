PARAM_GRIDS = {
    "LR_PCA": {
        "pca__n_components": [2, 5, 10],
        "lr__C": [0.01, 0.1, 1]
    },
    "SVM_PCA": {
        "pca__n_components": [2, 5],
        "svm__C": [0.1, 1],
        "svm__kernel": ["rbf"]
    },
    "KNN_PCA": {
        "pca__n_components": [2, 5],
        "knn__n_neighbors": [3, 5, 7]
    },

    "LR_LDA": {
        "lr__C": [0.01, 0.1, 1]
    },
    "SVM_LDA": {
        "svm__C": [0.1, 1]
    },
    "KNN_LDA": {
        "knn__n_neighbors": [3, 5, 7]
    },

    "LR_SVD": {
        "svd__n_components": [2, 5, 10],
        "lr__C": [0.01, 0.1, 1]
    },
    "SVM_SVD": {
        "svd__n_components": [2,5,10],
        "svm__C": [0.1, 1]
    },

    "DTREE": {
        "dtree__max_depth": [None, 10, 20]
    },
    "RF": {
        "rf__n_estimators": [100, 200],
        "rf__max_depth": [None, 10, 20]
    }
}
