from struct import error

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.feature_selection import SelectKBest, chi2, SelectPercentile
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import RandomOverSampler, SMOTEN
import re

def filter_location(location):
    result = re.findall("\,\s[A-Z]{2}", location)
    if len(result) > 0:
        return result[0][2:]
    else:
        return location

data = pd.read_excel("final_project.ods", engine="odf", dtype= str)

# Apply function filter_function to location column in the dataset
data["location"] = data["location"].apply(filter_location)
print(len(data["industry"].unique()))

# Drop 1 row containing missing values
#print(data.info())
data = data.dropna(axis=0)
target = "career_level"

x = data.drop(target, axis=1)
y = data[target]

#print(data.info())
#print(y.value_counts())




x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=1009, stratify=y)

ros = SMOTEN(random_state=42, k_neighbors=2, sampling_strategy={
    "bereichsleiter" : 1000,
    "director_business_unit_leader": 500,
    "specialist" : 500,
    "managing_director_small_medium_company": 100
})
print(y_train.value_counts())
print("-------------------")
x_train, y_train = ros.fit_resample(x_train, y_train)
print(y_train.value_counts())

#print(y_train.value_counts())

# TFIDF to preprocessing title column , ngram_range to indicate using unigram or bigrams, unigram and bigram
# vectorizer = TfidfVectorizer(ngram_range=(1, 2))
# output = vectorizer.fit_transform(x_train["title"])

# One Hot Encoder to preprocessing location column
# encoder = OneHotEncoder()
# output = encoder.fit_transform(x_train[["location"]])
# print(output.shape)


# output = vectorizer.fit_transform(x_train["description"])
# print(output.shape)
# unigram = (6459, 67181)
# unigram + bigrams = (6459, 753358)
# only bigrams = (6459, 686177)


# preprocessor = ColumnTransformer(transformers=[
#     ("title", TfidfVectorizer(stop_words="english"), "title"),
#     ("location", OneHotEncoder(handle_unknown='ignore'), ["location"]),
#     ("description", TfidfVectorizer(stop_words="english", ngram_range=(1,2),
#                                     min_df=0.01, max_df = 0.95), "description"),
#     ("function", OneHotEncoder(), ["function"]),
#     ("industry", TfidfVectorizer(stop_words="english"), "industry")
# ])
#
# # output = preprocessor.fit_transform(x_train)
# # print(x_train.shape)
# # print(output.shape)
#
#
# classifier = Pipeline(steps= [
#     ("preprocessor", preprocessor),
#     #("feature_selector", SelectKBest(chi2, k=300)),
#     ("feature_selector", SelectPercentile(chi2, percentile=10)),
#     ("classifier", RandomForestClassifier(random_state=42))
# ])
#
# params = {
#     "feature_selector__percentile": [10, 5, 2],
#     "preprocessor__description__min_df": [0.01, 0.05],
#     "preprocessor__description__max_df": [0.95, 0.99]
# }
#
# model = GridSearchCV(
#     estimator=classifier,
#     param_grid= params,
#     scoring="recall_weighted",
#     cv=6,
#     verbose=1
# )
# model.fit(x_train, y_train)
# print(model.best_score_)
# print(model.best_params_)
#
# y_predicted = model.predict(x_test)
#
# print(classification_report(y_test, y_predicted))


# Default Random Forest ~= 850,000 features
#                                         precision    recall  f1-score   support
#
#                         bereichsleiter       0.60      0.03      0.06       192
#          director_business_unit_leader       1.00      0.07      0.13        14
#                    manager_team_leader       0.63      0.54      0.58       534
# managing_director_small_medium_company       0.00      0.00      0.00         1
#   senior_specialist_or_project_manager       0.73      0.96      0.83       868
#                             specialist       0.00      0.00      0.00         6
#
#                               accuracy                           0.70      1615
#                              macro avg       0.49      0.27      0.27      1615
#                           weighted avg       0.68      0.70      0.65      1615

# Default Random Forest (apply min_df and max_df) ~= 8,000 features
#                                         precision    recall  f1-score   support
#
#                         bereichsleiter       0.50      0.04      0.07       192
#          director_business_unit_leader       1.00      0.07      0.13        14
#                    manager_team_leader       0.64      0.69      0.67       534
# managing_director_small_medium_company       0.00      0.00      0.00         1
#   senior_specialist_or_project_manager       0.80      0.94      0.86       868
#                             specialist       0.00      0.00      0.00         6
#
#                               accuracy                           0.74      1615
#                              macro avg       0.49      0.29      0.29      1615
#                           weighted avg       0.71      0.74      0.69      1615

# Default Random Forest (apply min_df and max_df +
# apply selectKbest using chi2 with 800 feature selected) ~= 800 features
#                                         precision    recall  f1-score   support
#
#                         bereichsleiter       0.64      0.15      0.24       192
#          director_business_unit_leader       1.00      0.07      0.13        14
#                    manager_team_leader       0.65      0.73      0.69       534
# managing_director_small_medium_company       0.00      0.00      0.00         1
#   senior_specialist_or_project_manager       0.83      0.93      0.88       868
#                             specialist       0.00      0.00      0.00         6
#
#                               accuracy                           0.76      1615
#                              macro avg       0.52      0.31      0.32      1615
#                           weighted avg       0.75      0.76      0.73      1615

# Default Random Forest (apply min_df and max_df +
# apply selectKbest using chi2 with 500 feature selected) ~= 500 features
#                                         precision    recall  f1-score   support
#
#                         bereichsleiter       0.57      0.16      0.24       192
#          director_business_unit_leader       1.00      0.07      0.13        14
#                    manager_team_leader       0.66      0.75      0.70       534
# managing_director_small_medium_company       0.00      0.00      0.00         1
#   senior_specialist_or_project_manager       0.84      0.93      0.88       868
#                             specialist       0.00      0.00      0.00         6
#
#                               accuracy                           0.76      1615
#                              macro avg       0.51      0.32      0.33      1615
#                           weighted avg       0.75      0.76      0.74      1615


# Default Random Forest (apply min_df and max_df +
# apply selectKbest using chi2 with 300 feature selected) ~= 300 features
#                                         precision    recall  f1-score   support
#
#                         bereichsleiter       0.54      0.23      0.32       192
#          director_business_unit_leader       1.00      0.07      0.13        14
#                    manager_team_leader       0.67      0.74      0.70       534
# managing_director_small_medium_company       0.00      0.00      0.00         1
#   senior_specialist_or_project_manager       0.85      0.93      0.89       868
#                             specialist       0.00      0.00      0.00         6
#
#                               accuracy                           0.77      1615
#                              macro avg       0.51      0.33      0.34      1615
#                           weighted avg       0.75      0.77      0.75      1615