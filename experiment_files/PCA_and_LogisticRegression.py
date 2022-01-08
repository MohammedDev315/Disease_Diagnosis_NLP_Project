#%%
# make predictions using pca with logistic regression
from sklearn.datasets import make_classification
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
import re
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD


#%%
file_path = "/Volumes/Lexar/DataScience/Disease_Diagnosis_NLP_Project/experiment_files/Data/ch13_label_sub_m1.csv"
df = pd.read_csv(file_path , names=["Label", "Disease"])
print(df.head())

##########################
###   Pre-Processing   ###
##########################

#%%

domain_stop_word = [
'patient', 'packing', 'surgery', 'infection',  'allergic', 'mouth',
'usually', 'relieve', 'treatment', 'place', 'symptoms', 'discharge',
'include' , 'pain', 'chronic', 'fever', 'produces',  'anterior', 'pack', 'posterior',
'sickness', 'traveling',  'blood', 'hours', 'adults', 'commonly', 'inserting', 'patch',
'tell', 'children', 'chronic', 'fever', 'produces',  'anterior', 'pack',
'posterior', 'sickness', 'traveling',  'blood', 'hours', 'adults', 'commonly',
'inserting', 'patch', 'tell', 'children', 'causes', 'longer' , 'weeks', 'form',
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
    final_text = ""
    for x in text.split():
        if x not in domain_stop_word:
            final_text = final_text + x  +" "
    return final_text
df['Disease'] = df['Disease'].apply(lambda x: clean_text(x))
df.head()

#%%
X = df.Disease
y = df.Label

#%%
cv = CountVectorizer( binary=True, stop_words='english')
X_cv = cv.fit_transform(X)
pd_cv = pd.DataFrame(X_cv.toarray(), columns=cv.get_feature_names())
#%%

# define the model
steps = [('lsa', TruncatedSVD(20)), ('m', LogisticRegression())]
model = Pipeline(steps=steps)
# fit the model on the whole dataset
model.fit( X_cv , y)


#%%
X_test = ["dizziness loss of balance  vomiting tinnitus of hearing in the high frequency range in one ear difficulty focusing your eyes "]
# X_test_cv3  = cv2.transform(X_test)
X_test_cv3  = cv.transform(X_test)
y_pred_cv3 = model.predict(X_test_cv3)
print(y_pred_cv3)

