import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

data = pd.read_excel("final_project.ods", dtype=str)
target = "career_level"

x = data.drop(target, axis=1)
y = data[target]
#print(data.info())
#print(y.value_counts())

# Drop 1 row containing missing values
data = data.dropna(axis=0)
#print(data.info())

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y)

#print(y_train.value_counts())

vectorizer = TfidfVectorizer()
output = vectorizer.fit_transform(x_train["title"])

