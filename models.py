from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

from helper import coclust


class Classifier1NN:
    def __init__(self, train_dataset):
        self.train_dataset = train_dataset
        self.model = KNeighborsClassifier(n_neighbors=1)

        if len(self.train_dataset.queries) > 0:
            self.update()

    def update(self):
        '''
        Refits the classifier using new queries
        '''
        assert (len(self.train_dataset.queries) > 0)
        self.model.fit(self.train_dataset.x[self.train_dataset.queries, :],
                       self.train_dataset.y[self.train_dataset.queries])
        self.predicted_labels = self.model.predict(self.train_dataset.x)
        self.accuracy = coclust(self.predicted_labels, self.train_dataset.y)

    def predict(self, new_data=None):
        if new_data is None:
            return self.model.predict(self.train_dataset.x)
        else:
            return self.model.predict(new_data.x)

    def get_accuracy(self, new_data=None):
        if new_data is None:
            return accuracy_score(self.train_dataset.y, self.predict(self.train_dataset))
        else:
            return accuracy_score(new_data.y, self.predict(new_data))
