import pandas as pd  # pandas is a dataframe library
import matplotlib.pyplot as plt  # matplotlib.pyplot plots data
from sklearn.impute import SimpleImputer  # Hidden Missing Values
from sklearn.model_selection import train_test_split
from sklearn import metrics  # import the performance metrics library
from sklearn.naive_bayes import GaussianNB  # Importing algorithm - Naive Bayes
from sklearn import metrics
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("./pima-data.csv")

print(df.shape)  # describes data in terms of number of variables

print("First N rows")
print(df.head(5))

print(df.isnull().values.any())  # Check for null values

print(df.corr())  # correlation table

del df['skin']  # The skin and thickness columns are correlated 1 to 1. Dropping the skin column

df.head(5)

diabetes_map = {True: 1, False: 0}
df['diabetes'] = df['diabetes'].map(diabetes_map)  # Change diabetes from boolean to integer, True=1, False=0

df.head(5)

df.to_pickle("./pima-data-processed.p")  # Save pre-processed dataframe for later use

feature_col_names = ['num_preg', 'glucose_conc', 'diastolic_bp', 'thickness', 'insulin', 'bmi', 'diab_pred', 'age']
predicted_class_names = ['diabetes']

X = df[feature_col_names].values  # predictor feature columns (8 X m)
y = df[predicted_class_names].values  # predicted class (1=true, 0=false) column (1 X m)

split_test_size = 0.30

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_test_size, random_state=42)
# test_size = 0.3 is 30%, 42 is the answer to everything

print(df.head())

fill_0 = SimpleImputer(missing_values=0, strategy="mean")  # Impute with mean all 0 readings

X_train = fill_0.fit_transform(X_train)
X_test = fill_0.fit_transform(X_test)

nb_model = GaussianNB()

nb_model.fit(X_train, y_train.ravel())

nb_predict_train = nb_model.predict(X_train)  # predict values using the training data

metrics.accuracy_score(y_train, nb_predict_train)  # Accuracy

nb_predict_test = nb_model.predict(X_test)  # predict values using the testing data

metrics.accuracy_score(y_test, nb_predict_test)  # training metrics

print("Confusion Matrix:\n")
print(metrics.confusion_matrix(y_test, nb_predict_test))
print("\nClassification Report:\n")
print(metrics.classification_report(y_test, nb_predict_test))

lr_model = LogisticRegression(C=0.7, solver='liblinear', random_state=42)
lr_model.fit(X_train, y_train.ravel())
lr_predict_test = lr_model.predict(X_test)

print("Accuracy: {0:.4f}".format(metrics.accuracy_score(y_test, lr_predict_test)))
print(metrics.confusion_matrix(y_test, lr_predict_test))
print("")
print("Classification Report")
print(metrics.classification_report(y_test, lr_predict_test))
