#%%
import pandas as pd
import numpy as np
import scipy.sparse as ss
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import datasets
from corextopic import corextopic as ct
from corextopic import vis_topic as vt
import matplotlib.pyplot as plt
import seaborn as sns
#%%
file_path = "/Volumes/Lexar/DataScience/Disease_Diagnosis_NLP_Project/experiment_files/Data/df_signs_and_diagnosis.csv"
df = pd.read_csv(file_path , names=["Title" , "Des"])
#%%
vectorizer = CountVectorizer(max_features=20000,
                             stop_words='english', token_pattern="\\b[a-z][a-z]+\\b",
                             binary=True)

doc_word = vectorizer.fit_transform(df["Des"])
words = list(np.asarray(vectorizer.get_feature_names()))
#%%
num_of_topic = 20
#%%
# I recommend adding docs=df.data to make it easier to check which sentences are in each resulting topic
topic_model = ct.Corex(n_hidden=num_of_topic, words=words, seed=1)
topic_model.fit(doc_word, words=words, docs=df["Des"])

#%%
# Print all topics from the CorEx topic model
topics = topic_model.get_topics()
print(topics)
#%%
for n ,topic in enumerate(topics):
    topic_words ,_,_ = zip(*topic)
    print('{}: '.format(n) + ','.join(topic_words))

#%%
predictions = pd.DataFrame(topic_model.predict(doc_word), columns=['topic'+str(i) for i in range(num_of_topic)])
predictions.head(3)