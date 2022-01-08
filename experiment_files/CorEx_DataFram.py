#%%
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
from nltk.tokenize import word_tokenize
import re
import spacy
from corextopic import corextopic as ct
nlp = spacy.load('en_core_web_sm')

#%%
num_of_topic = 3
file_path = "/Volumes/Lexar/DataScience/Disease_Diagnosis_NLP_Project/experiment_files/Data/df_signs_and_diagnosis.csv"
df = pd.read_csv(file_path , names=["Title" , "Des"])


#%%
domain_stop_word = [
"and", "symptom" ,  'patient' , 'cause', 'include', 'surgery', 'packing', 'allergic', 'usually', 'relieve', 'treatment', 'use', 'place',
'medium', 'loss', 'hearing', 'pain', 'include', 'produce','middle', 'sever', 'effect',
'child' , 'travel', 'insert', 'insertion', 'blood', 'adult', 'commonly',
'symptom', 'ear', 'attack' , 'sever' , 'occure', 'fever', 'fall', 'direction', 'unaffected',
'tip', 'depend', 'pediatric',  'parent', 'associate'
]

def clean_text(text):
    text = str(text)
    text = text.lower()
    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!?.\/'+]", " ", text)
    text = re.sub(r"\+", " ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ", text)
    text = re.sub(r"\?", " ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"[0-9]", " ", text)
    doc = nlp(text)
    lemma_text = ''
    for token in doc:
        text_token = token.lemma_
        lemma_text = lemma_text + text_token + " "
    text = lemma_text
    final_text = ""
    for x in text.split():
        if x not in domain_stop_word:
            final_text = final_text + x  +" "
    return final_text

df['Des'] = df['Des'].apply(lambda x: clean_text(x))
df.head()
#%%
tokens = word_tokenize(df)
print(tokens)

#%%
mask = ( df.loc[: , "Title"] == "Signs" )
new_df = df[mask]
#%%
vectorizer = CountVectorizer(stop_words='english')
doc_word = vectorizer.fit_transform(new_df["Des"])
print(doc_word.shape)
#%%
pd_vectorizer = pd.DataFrame(doc_word.toarray(), index=new_df["Des"], columns=vectorizer.get_feature_names())
#%%


lsa = TruncatedSVD(num_of_topic)
doc_topic = lsa.fit_transform(doc_word)
print(lsa.explained_variance_ratio_)
#%%
topic_word = pd.DataFrame(lsa.components_.round(3),
             index = ['component'+str(i) for i in range(num_of_topic)],
             columns = vectorizer.get_feature_names())
#%%
print(topic_word)
#%%
def display_topics(model, feature_names, no_top_words, topic_names=None):
    for ix, topic in enumerate(model.components_):
        if not topic_names or not topic_names[ix]:
            print("\nTopic ", ix)
        else:
            print("\nTopic: '",topic_names[ix],"'")
        print(", ".join([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]]))
#%%
display_topics(lsa, vectorizer.get_feature_names(), 20)
#%%
Vt = pd.DataFrame(doc_topic.round(5),
             index = df["Des"],
             columns = ['component'+str(i) for i in range(num_of_topic)])
#%%






