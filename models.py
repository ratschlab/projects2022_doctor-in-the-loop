from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


class Classifier1NN:
    def __init__(self, dataset):
        self.dataset= dataset
        self.model= KNeighborsClassifier(n_neighbors=1)

        if len(self.dataset.queries)>0:
            self.update()

    def update(self):
        '''
        Refits the classifier using new queries
        '''
        assert(len(self.dataset.queries)>0)
        self.model.fit(self.dataset.x[self.dataset.queries, :], self.dataset.y[self.dataset.queries])
        self.pseudo_labels = self.model.predict(self.dataset.x)
        self.accuracy = accuracy_score(self.pseudo_labels, self.dataset.y)

    def predict(self, new_data=None):
        if new_data is None:
            return self.model.predict(self.dataset.x)
        else:
            return self.model.predict(new_data.x)

    def get_accuracy(self, new_data=None):
        if new_data is None:
            return accuracy_score(self.dataset.y, self.predict(self.dataset))
        else:
            return accuracy_score(new_data.y, self.predict(new_data))



