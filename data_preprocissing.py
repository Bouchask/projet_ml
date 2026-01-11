import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import os

class Data:
    def __init__(self, url_data: str):
        self.url = url_data
        self.column_classes = [] 

    # ==============================
    # Load data safely
    # ==============================
    def upload_data(self):
        try:
            data = pd.read_csv(self.url)
            self.column_classes = self.feature_classes(data)
            return data
        except FileNotFoundError:
            raise FileNotFoundError("❌ File not found. Check the path.")
        except Exception as e:
            raise Exception(f"❌ Error while loading data: {e}")

    # ==============================
    # Data info
    # ==============================
    def info_data(self, data):
        if not isinstance(data, pd.DataFrame):
            raise TypeError("❌ Input must be a pandas DataFrame")
        return data.info()

    # ==============================
    # Identify Categorical Features
    # ==============================
    def feature_classes(self, data):
        if not isinstance(data, pd.DataFrame):
            raise TypeError("❌ Input must be a pandas DataFrame")

        # UPDATED LOGIC:
        # We only want to LabelEncode TEXT columns (Day of the week, Traffic Situation)
        # We exclude Time and Date explicitly.
        return [
            col for col in data.columns
            if data[col].dtype == 'object' and col not in ["Date", "Time"]
        ]

   
    # ==============================
    # Encode binary/categorical features
    # ==============================
    def transform_classes_features(self, data):
        data = data.copy()
        columns = self.feature_classes(data)

        if not columns:
            return data

        le = LabelEncoder()
        for col in columns:
            # Convert to string to ensure safe encoding
            data[col] = le.fit_transform(data[col].astype(str))

        return data


    # ==============================
    # Train / Val / Test split
    # ==============================
    def split_data(self, X, y):
        if len(X) != len(y):
            raise ValueError("❌ X and y must have the same length")

        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y,
            train_size=0.7,
            random_state=42,
            stratify=y
        )

        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            train_size=0.5,
            random_state=42,
            stratify=y_temp
        )

        return X_train, X_val, X_test, y_train, y_val, y_test

    # ==============================
    # Save Preprocessed Data
    # ==============================
    def sauvgarde_data(self, data, folder_name="data", file_name="data_model.csv"):
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)
            
        os.makedirs(folder_name, exist_ok=True)
        save_path = os.path.join(folder_name, file_name)
        data.to_csv(save_path, index=False)
        print(f"✅ Dataframe saved successfully to: {save_path}")

    # ==============================
    # Standard Data (StandardScaler)
    # ==============================
    def standard_data(self, data):
        data = data.copy()
        sd = StandardScaler()
        
        # 1. Get categorical columns to exclude
        classes_to_exclude = self.feature_classes(data)
        
        # 2. Select ONLY numeric columns, EXCLUDING Date/Time
        list_feature_numeric = [
            col for col in data.columns 
            if (col not in classes_to_exclude 
                and col not in ["Date", "Time"]  # Explicit exclusion
                and pd.api.types.is_numeric_dtype(data[col]))
        ]
        
        # 3. Apply Scaling
        for col in list_feature_numeric:
            # Use [[col]] to pass a 2D DataFrame (N,1) instead of Series (N,)
            data[col] = sd.fit_transform(data[[col]])
            
        return data

    # ==============================
    # MinMax Scaler Data
    # ==============================
    def MinMax_data(self, data):
        data = data.copy()
        sl = MinMaxScaler()
        
        classes_to_exclude = self.feature_classes(data)
        
        list_feature_numeric = [
            col for col in data.columns 
            if (col not in classes_to_exclude 
                and col not in ["Date", "Time"] 
                and pd.api.types.is_numeric_dtype(data[col]))
        ]
        
        for col in list_feature_numeric:
            data[col] = sl.fit_transform(data[[col]])
            
        return data