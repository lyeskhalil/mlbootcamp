from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import neighbors
from sklearn.datasets import load_wine

def prepared_data_lab2():
    wine_data = load_wine()
    feature_data = wine_data.data
    target_data = wine_data.target
    target_names = wine_data.target_names
    return train_test_split(feature_data, target_data, test_size=0.3, random_state=0)

def report_model_score(data, model, report=True):
    scores = cross_val_score(model, data[0], data[1], cv=5)
    if report:
        print("CV Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    return scores.mean()
    
def train_test_split_1d(feature_data, target_data, test_size, random_state):
    X_train, X_test, y_train, y_test = train_test_split(feature_data, target_data, test_size=test_size, random_state=random_state)
    return X_train.reshape(-1,1), X_test.reshape(-1,1), y_train, y_test