import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
import re

def filter_location(location):
    result = re.findall("\,\s[A-Z]{2}", location)

    if len(result) > 0:
        return result[0][2:]
    else:
        return location


data = pd.read_excel("final_project.ods", dtype=str)

# Apply function filter_function to location column in the dataset
data["location"] = data["location"].apply(filter_location)

target = "career_level"

x = data.drop(target, axis=1)
y = data[target]

#print(data.info())
#print(y.value_counts())

# Drop 1 row containing missing values
data = data.dropna(axis=0)
#print(data.info())


x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=1009, stratify=y)

#print(y_train.value_counts())

# TFIDF to preprocessing title column , ngram_range to indicate using unigram or bigrams, unigram and bigram
vectorizer = TfidfVectorizer(ngram_range=(1, 2))
# output = vectorizer.fit_transform(x_train["title"])

# One Hot Encoder to preprocessing location column
# encoder = OneHotEncoder()
# output = encoder.fit_transform(x_train[["location"]])
# print(output.shape)


output = vectorizer.fit_transform(x_train["description"].values.astype('U'))
print(output.shape)


# unigram = (6459, 67181)
# unigram + bigrams = (6459, 753358)
# only bigrams = (6459, 686177)

