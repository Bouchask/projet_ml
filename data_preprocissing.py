from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import pandas as pd
import numpy as np 

class Data:
    def __init__(self, url: str, target_column: str):
        self.url = url
        self.target_column = target_column
        self.data = pd.read_csv(url)

        
        if 'Time' in self.data.columns:
            self.data['Time'] = pd.to_datetime(self.data['Time'], format='%I:%M:%S %p', errors='coerce')
            self.data['Time_seconds'] = (
                self.data['Time'].dt.hour * 3600 +
                self.data['Time'].dt.minute * 60 +
                self.data['Time'].dt.second
            ).fillna(0).astype("int") 
            self.data.drop(columns=['Time'], inplace=True)

    def preprocessing(self, X):
        
        numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
        categorical_cols = X.select_dtypes(include="object").columns.tolist()

        pipe_numeric = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])
        
        pipe_class = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ])

        preprocessor = ColumnTransformer(transformers=[
            ("numeric", pipe_numeric, numeric_cols),
            ("class", pipe_class, categorical_cols)
        ])
        
        return preprocessor

    def split_data(self):
        Y = self.data[self.target_column]
        X = self.data.drop(columns=[self.target_column])

        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, train_size=0.7, stratify=Y, random_state=42
        )

        
        preprocessor = self.preprocessing(X_train)

        
        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)

        return X_train_processed, X_test_processed, Y_train, Y_test

    def get_data_pca(self, X_train, X_test):
        pca = PCA(n_components=2)
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)
        return X_train_pca, X_test_pca

    def get_data_lda(self, X_train, X_test, Y_train):
        # LDA requires Y_train!
        lda = LinearDiscriminantAnalysis(n_components=2)
        X_train_lda = lda.fit_transform(X_train, Y_train)
        X_test_lda = lda.transform(X_test)
        return X_train_lda, X_test_lda

    def get_data_svd(self, X_train, X_test):
        svd = TruncatedSVD(n_components=2)
        X_train_svd = svd.fit_transform(X_train)
        X_test_svd = svd.transform(X_test)
        return X_train_svd, X_test_svd