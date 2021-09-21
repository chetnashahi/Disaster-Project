#!/usr/bin/env python
# coding: utf-8

# # ML Pipeline Preparation
# Follow the instructions below to help you create your ML pipeline.
# ### 1. Import libraries and load data from database.
# - Import Python libraries
# - Load dataset from database with [`read_sql_table`](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_sql_table.html)
# - Define feature and target variables X and Y

# In[66]:


# import libraries
import sys
import pandas as pd
import re
import nltk
nltk.download(['punkt','stopwords','wordnet'])
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sqlalchemy.engine import create_engine
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
#from sklearn import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import pickle


# In[51]:


# load data from database
engine = create_engine('sqlite:///data//DisasterResponse.db')
df = pd.read_sql_table('data',engine)  
X = df['message'].values
y = df[df.columns[4:]]


# In[52]:


df.head()


# In[53]:


y.head()


# ### 2. Write a tokenization function to process your text data

# In[54]:


def tokenize(text):
    '''Process text into clean tokens
    Text is processed by keeping it in lower case & words lemmatized into their original stem
    
    Input:
    text (str) : message in text form
    
    Output:
    clean_tokens (array): array of words after processing
    '''
    url_regex= 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls= re.findall(url_regex,text)
    for url in detected_urls:
        text= text.replace(url,"urlplaceholder")
    text=re.sub(r"[^a-zA-Z0-9]"," ", text.lower())
    tokens=word_tokenize(text)
    stop=stopwords.words("english")
    words= [t for t in tokens if t not in stop]
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens=[]
    for tok in words:
        clean_tok=lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens


# In[55]:


print(tokenize(X[3]))


# ### 3. Build a machine learning pipeline
# This machine pipeline should take in the `message` column as input and output classification results on the other 36 categories in the dataset. You may find the [MultiOutputClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.multioutput.MultiOutputClassifier.html) helpful for predicting multiple target variables.

# In[56]:


pipeline = Pipeline([
    ('vect',CountVectorizer(tokenizer=tokenize)),
    ('tfidf',TfidfTransformer()),
    ('clf',MultiOutputClassifier(RandomForestClassifier(n_jobs=-1)))
])


# In[57]:


pipeline.get_params()


# ### 4. Train pipeline
# - Split data into train and test sets
# - Train pipeline

# In[58]:



X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=42)


# In[59]:


pipeline.fit(X_train,y_train)
y_pred=pipeline.predict(X_test)

accuracy = (y_pred == y_test).mean()
print("Accuracy:", accuracy)


# In[60]:


#print(classification_report(y_test,y_pred,target_names = df.columns[4:]))
#y_pred.shape,y_test.shape,len(list(y.columns))
df.head


# ### 5. Test your model
# Report the f1 score, precision and recall for each output category of the dataset. You can do this by iterating through the columns and calling sklearn's `classification_report` on each.

# In[61]:


y_test2=np.asarray(y_test)
print(classification_report(y_test2,y_pred,target_names=y))


# ### 6. Improve your model
# Use grid search to find better parameters. 

# In[62]:


parameters = parameters = {
    'clf__estimator__max_depth':[15,25],
    'clf__estimator__n_estimators':[100,200],
    }

cv = GridSearchCV(pipeline,param_grid=parameters)


# In[9]:


cv.fit(X_train,y_train)
y_pred=cv.predict(X_test)


# In[63]:


import numpy as np
y_test2=np.asarray(y_test)
print(y_pred.shape,y_test2.shape,y_test.shape)
type(y_pred)
print(len(y.columns))


# In[64]:


target_names=y.columns
print(classification_report(y_test2,y_pred,target_names=target_names))


# In[ ]:





# ### 7. Test your model
# Show the accuracy, precision, and recall of the tuned model.  
# 
# Since this project focuses on code quality, process, and  pipelines, there is no minimum performance metric needed to pass. However, make sure to fine tune your models for accuracy, precision and recall to make your project stand out - especially for your portfolio!

# In[ ]:


def display_results(y_test,y_pred):

    labels=np.unique(y_pred)
    confusion_mat=confusion_matrix(y_test,y_pred,labels=labels)
    accuracy=(y_pred==y_test).mean()
    
    print("Labels:",labels)
    print("Confusion Matrix \n:", confusion_mat)
    print("Accuracy:", accuracy)


# ### 8. Try improving your model further. Here are a few ideas:
# * try other machine learning algorithms
# * add other features besides the TF-IDF

# In[ ]:





# In[ ]:





# ### 9. Export your model as a pickle file

# In[67]:


with open('Classifier.pkl','wb') as file:
    pickle.dump(cv,file)


# ### 10. Use this notebook to complete `train.py`
# Use the template file attached in the Resources folder to write a script that runs the steps above to create a database and export a model based on a new dataset specified by the user.

# In[ ]:




