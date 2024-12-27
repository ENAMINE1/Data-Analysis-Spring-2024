# Importing the required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score

# Load the dataset
url = 'https://docs.google.com/spreadsheets/d/1k4LkeZctJlmkzwnePxfaG3hfD-pgsJ8pFH5clW9FlqA/gviz/tq?tqx=out:csv'
url2 = './Synthetic_SalaryData_Test 13.csv'
data = pd.read_csv(url2)

# Handling missing values
data.loc[(data['age'] < 18) & (data['maritalstatus'].isnull()), 'maritalstatus'] = 'Never-married'
marital_mode = data['maritalstatus'].mode()[0]
data['maritalstatus'].fillna(marital_mode, inplace=True)

data.loc[data['workclass'] == 'Without-pay', 'hoursperweek'] = 0
median_hours = data[data['workclass'] != 'Without-pay']['hoursperweek'].median()
data['hoursperweek'].fillna(median_hours, inplace=True)

data['race'].fillna(data['race'].mode()[0], inplace=True)
data['sex'].fillna(data['sex'].mode()[0], inplace=True)

# Convert categorical columns to numerical codes
for column in ['workclass', 'education', 'maritalstatus', 'occupation', 'relationship', 'race', 'sex', 'native']:
    data[column] = data[column].astype('category').cat.codes

# Target variable 'Possibility' processing
data['Possibility'] = (data['Possibility'] == '>0.5').astype(int)

# Splitting the dataset into training and testing sets
X = data.drop('Possibility', axis=1)
y = data['Possibility']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Naive Bayes from scratch
def separate_by_class(dataset):
    classes = np.unique(dataset['Possibility'])
    separated_classes = {}
    for cls in classes:
        separated_classes[cls] = dataset[dataset['Possibility'] == cls]
    return separated_classes

def summarize_data(dataset):
    summaries = {}
    for column in dataset.columns[:-1]:
        summaries[column] = (dataset[column].mean(), dataset[column].std(), len(dataset[column]))
    return summaries

def summarize_by_class(dataset):
    separated = separate_by_class(dataset)
    summaries = {}
    for class_value, instances in separated.items():
        summaries[class_value] = summarize_data(instances)
    return summaries

def calculate_probability(x, mean, std):
    if std == 0:
        std = 0.0001
    exponent = np.exp(-((x-mean)**2 / (2 * std**2)))
    return (1 / (np.sqrt(2 * np.pi) * std)) * exponent

def calculate_class_probabilities(summaries, input_vector):
    total_rows = sum([summaries[label][next(iter(summaries[label]))][2] for label in summaries])
    probabilities = {}
    for class_value, class_summaries in summaries.items():
        probabilities[class_value] = summaries[class_value][next(iter(summaries[class_value]))][2] / float(total_rows)
        for feature, feature_sum in class_summaries.items():
            mean, std, count = feature_sum
            probabilities[class_value] *= calculate_probability(input_vector[feature], mean, std)
    return probabilities

def predict(summaries, input_vector):
    probabilities = calculate_class_probabilities(summaries, input_vector)
    best_label, best_prob = None, -1
    for class_value, probability in probabilities.items():
        if best_label is None or probability > best_prob:
            best_prob = probability
            best_label = class_value
    return best_label

def compute_confusion_matrix(true, pred):
    K = len(np.unique(true))
    result = np.zeros((K, K))
    for i in range(len(true)):
        result[true[i]][pred[i]] += 1
    return result

def compute_precision_recall_f1(conf_matrix):
    true_positive = conf_matrix[1, 1]
    false_positive = conf_matrix[0, 1]
    false_negative = conf_matrix[1, 0]
    true_negative = conf_matrix[0, 0]

    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) != 0 else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) != 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

    return precision, recall, f1

model_summaries = summarize_by_class(X_train.join(y_train))
predictions_naive = np.array([predict(model_summaries, row) for index, row in X_test.iterrows()])
actuals = np.array(y_test)

conf_matrix = compute_confusion_matrix(actuals, predictions_naive)
precision, recall, f1 = compute_precision_recall_f1(conf_matrix)

print(f"Custom Naive Bayes - Precision: {precision:.2f}")
print(f"Custom Naive Bayes - Recall: {recall:.2f}")
print(f"Custom Naive Bayes - F1 Score: {f1:.6f}")
accuracy_naive = accuracy_score(y_test, predictions_naive)
print(f"Custom Naive Bayes - Accuracy: {accuracy_naive:.6f}")

# Scikit-Learn Models
models = {
    "Naive Bayes": GaussianNB(),
    "SVM": SVC(),
    "Decision Tree": DecisionTreeClassifier(),
    "KNN": KNeighborsClassifier()
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    f1_result = f1_score(y_test, predictions, average='binary')
    results[name] = (accuracy, f1_result)
    print(f"{name} - Accuracy: {accuracy:.6f}, F1 Score: {f1_result:.6f}")

# Comparing with custom Naive Bayes
new_predictions_naive = np.array([predict(model_summaries, row) for index, row in X_test.iterrows()])
accuracy_naive = accuracy_score(y_test, new_predictions_naive)
f1_naive = f1_score(y_test, new_predictions_naive, average='binary')
print(f"Custom Naive Bayes - Accuracy: {accuracy_naive:.6f}, F1 Score: {f1_naive:.6f}")

# Ensemble Model
class VotingEnsemble:
    def __init__(self):
        self.svm = SVC(probability=True)
        self.dt = DecisionTreeClassifier()
        self.knn = KNeighborsClassifier()

    def fit(self, X, y):
        self.svm.fit(X, y)
        self.dt.fit(X, y)
        self.knn.fit(X, y)

    def predict(self, X):
        nb_preds = new_predictions_naive
        svm_preds = self.svm.predict(X)
        dt_preds = self.dt.predict(X)
        knn_preds = self.knn.predict(X)

        final_preds = []
        for i in range(len(X)):
            votes = [nb_preds[i], svm_preds[i], dt_preds[i], knn_preds[i]]
            final_preds.append(np.bincount(votes).argmax())
        return np.array(final_preds)

    def plot_accuracies(self, X_test, y_test):
        nb_preds = new_predictions_naive
        svm_preds = self.svm.predict(X_test)
        dt_preds = self.dt.predict(X_test)
        knn_preds = self.knn.predict(X_test)

        nb_accuracy = accuracy_score(y_test, nb_preds)
        svm_accuracy = accuracy_score(y_test, svm_preds)
        dt_accuracy = accuracy_score(y_test, dt_preds)
        knn_accuracy = accuracy_score(y_test, knn_preds)

        ensemble_preds = self.predict(X_test)
        ensemble_accuracy = accuracy_score(y_test, ensemble_preds)

        models = ['Naive Bayes', 'SVM', 'Decision Tree', 'KNN', 'Ensemble']
        accuracies = [nb_accuracy, svm_accuracy, dt_accuracy, knn_accuracy, ensemble_accuracy]

        plt.figure(figsize=(10, 6))
        bars = plt.bar(models, accuracies, color=['skyblue', 'lightgreen', 'salmon', 'plum', 'gold'])
        plt.ylim([0, 1])
        plt.ylabel('Accuracy')
        plt.title('Individual Model Accuracies vs Ensemble Accuracy')

        for bar, acc in zip(bars, accuracies):
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.02, f"{acc:.2f}", ha='center', va='bottom')
        plt.savefig('model_accuracies.png')

ensemble = VotingEnsemble()
ensemble.fit(X_train, y_train)

# Make predictions on the test set
y_pred = ensemble.predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Ensemble Model Accuracy: {accuracy:.4f}")
ensemble.plot_accuracies(X_test, y_test)

#Evaluste F1 score
f1_result = f1_score(y_test, y_pred, average='binary')
print(f"Ensemble Model F1 Score: {f1_result:.4f}")


