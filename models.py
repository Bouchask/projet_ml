from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


class Models:
    def __init__(self):
        # -------- Models --------
        self.lr = LogisticRegression(
            penalty="l2",
            max_iter=1000,
            solver="lbfgs",
            C=0.1,
            random_state=42
        )

        self.svm = SVC(
            C=0.1,
            kernel="rbf",
            gamma="scale",
            probability=True,
            random_state=42
        )

        self.knn = KNeighborsClassifier(n_neighbors=5)

        self.dtree = DecisionTreeClassifier(random_state=42)

        self.rf = RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )

    # =====================================================
    # PCA PIPELINES (Dense data)
    # =====================================================
    def pca_pipelines(self):
        return {
            "LR_PCA": Pipeline([
                ("pca", PCA(n_components=2, random_state=42)),
                ("lr", self.lr)
            ]),
            "SVM_PCA": Pipeline([
                ("pca", PCA(n_components=2, random_state=42)),
                ("svm", self.svm)
            ]),
            "KNN_PCA": Pipeline([
                ("pca", PCA(n_components=2, random_state=42)),
                ("knn", self.knn)
            ])
        }

    # =====================================================
    # LDA PIPELINES (Supervised – classification only)
    # =====================================================
    def lda_pipelines(self):
        lda = LinearDiscriminantAnalysis(n_components=2)

        return {
            "LR_LDA": Pipeline([
                ("lda", lda),
                ("lr", self.lr)
            ]),
            "SVM_LDA": Pipeline([
                ("lda", lda),
                ("svm", self.svm)
            ]),
            "KNN_LDA": Pipeline([
                ("lda", lda),
                ("knn", self.knn)
            ])
        }

    # =====================================================
    # SVD PIPELINES (Sparse / OneHot)
    # =====================================================
    def svd_pipelines(self):
        return {
            "LR_SVD": Pipeline([
                ("svd", TruncatedSVD(n_components=2, random_state=42)),
                ("lr", self.lr)
            ]),
            "SVM_SVD": Pipeline([
                ("svd", TruncatedSVD(n_components=2, random_state=42)),
                ("svm", self.svm)
            ])
        }

    # =====================================================
    # TREE-BASED (No reduction)
    # =====================================================
    def tree_pipelines(self):
        return {
            "DTREE": Pipeline([
                ("dtree", self.dtree)
            ]),
            "RF": Pipeline([
                ("rf", self.rf)
            ])
        }

    # =====================================================
    # ALL PIPELINES
    # =====================================================
    def create_all_pipelines(self):
        pipelines = {}
        pipelines.update(self.pca_pipelines())
        pipelines.update(self.lda_pipelines())
        pipelines.update(self.svd_pipelines())
        pipelines.update(self.tree_pipelines())
        return pipelines
