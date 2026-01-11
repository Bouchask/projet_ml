from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np 
class Data:
    def __init__(self,url :str,target_column :str):
        self.url = url
        self.status_std = False
        self.status_coder = False
        self.target_column = target_column
        self.scaler = StandardScaler()
        self.le= LabelEncoder()
        data = pd.read_csv(url)
        self.data = pd.DataFrame(data)

        self.data['Time'] = pd.to_datetime(
            self.data['Time'],
            format='%I:%M:%S %p',
            errors='coerce')

        self.data['Time_seconds'] = (
            self.data['Time'].dt.hour * 3600 +
            self.data['Time'].dt.minute * 60 +
            self.data['Time'].dt.second).astype("int")

        self.data.drop(columns=['Time'], inplace=True)
    def info(self):
        self.data.info()
    def feature_classes(self):
        list_feature_classes = []
        list_feature_classes = [ name_column for name_column in self.data.columns if (self.data[name_column].nunique()>=2) and (self.data[name_column].nunique()<=7) ]
        return list_feature_classes
    def standardScaler(self):
        self.transfer_classes()
        for  i  in self.data.columns :
            if i != self.target_column :
                self.data[i] = self.scaler.fit_transform(self.data[[i]])
        self.status_std = True
    def transfer_classes(self):
        list_feature = self.feature_classes()
        for i  in list_feature :
            self.data[i] = self.le.fit_transform(self.data[i])
        self.status_coder = True
    def split_data(self):
        if self.status_std == False :
            self.standardScaler()
        if self.status_coder == False :
            self.transfer_classes()
        Y = self.data[self.target_column]
        X = self.data.drop(self.target_column , axis=1)
        X_tarin,X_test,Y_train,Y_test = train_test_split(X,Y,train_size=0.7,stratify=Y)
        return X_tarin,X_test,Y_train,Y_test
    
        
    