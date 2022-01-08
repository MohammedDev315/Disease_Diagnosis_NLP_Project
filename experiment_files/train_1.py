#%%
import nltk
import pandas as pd
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
#%%
file_path = "/Volumes/Lexar/DataScience/Disease_Diagnosis_NLP_Project/experiment_files/Data/ch13_label_sub_m1.csv"
df = pd.read_csv(file_path , names=["Label", "Disease"])
print(df.head())

##########################
###   Pre-Processing   ###
##########################

#%%
punc_lower = lambda x: re.sub('[%s]' % re.escape(string.punctuation), ' ', x.lower())
df['Disease'] = df.Disease.map(punc_lower)
df.head()

#%%
X = df.Disease
y = df.Label
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3 , random_state=73)

#%%
cv1 = CountVectorizer(stop_words='english')
X_train_cv1 = cv1.fit_transform(X_train)
X_test_cv1  = cv1.transform(X_test)
pd_cv1 = pd.DataFrame(X_train_cv1.toarray(), columns=cv1.get_feature_names())
#%%
lr = LogisticRegression()
lr.fit(X_train_cv1, y_train)
y_pred_cv1 = lr.predict(X_test_cv1)
score = lr.score(X_test_cv1, y_test)
print(score)
#%%
X, y = make_classification(n_samples=1000, n_features=4,                            n_informative=2, n_redundant=0,
                            random_state=0, shuffle=False)
clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(X_train_cv1, y_train)
y_pred_cv1 = clf.predict(X_test_cv1)
score = clf.score(X_test_cv1, y_test)
print(score)

#%%
print(X_test)
# print(X_test_cv1)
print("-------")
print(y_pred_cv1)

#%%
print(X_test)
#%%
X_test = ["Itching in your ear canal. Slight redness inside your ear pinna or auricle odorless fluid"]
X_test_cv3  = cv1.transform(X_test)
y_pred_cv3 = lr.predict(X_test_cv3)
print(y_pred_cv3)

