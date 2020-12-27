import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier

class RFClassifier(object):
    def __init__(self, model):
        self.model = model
        self.rf = RandomForestClassifier(n_estimators = 100, random_state = 42)
        self.scaler = StandardScaler()

    def train(self, train_loader):
        X, Y = self.convert_tensor_to_np(train_loader)
        self.scaler.fit(X)
        X = self.scaler.transform(X)
        self.rf.fit(X, Y)

    def test(self, validation_loader):
        X, Y = self.convert_tensor_to_np(validation_loader)
        X = self.scaler.transform(X)
        return self.rf.score(X, Y)

    def convert_tensor_to_np(self, data_loader):
        train_feature_vector = []
        train_labels_vector = []
        for batch_x, batch_y in data_loader:
          batch_x = batch_x.to(self.device)
          train_labels_vector.extend(batch_y)
          features, _ = self.model(batch_x)
          train_feature_vector.extend(features.cpu().detach().numpy())

        train_feature_vector = np.array(train_feature_vector)
        train_labels_vector = np.array(train_labels_vector)

        return train_feature_vector, train_labels_vector
