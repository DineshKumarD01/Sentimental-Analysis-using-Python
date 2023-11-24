#!/usr/bin/env python
# coding: utf-8

# ###Importing libraries

# In[1]:


import numpy as np
import pandas as pd


# ### Importing dataset
# 
# 
# 

# In[5]:


dataset = pd.read_csv('RestaurantReviews_1.tsv', delimiter = '\t', quoting = 3)


# In[6]:


dataset.shape


# In[7]:


dataset.head()


# ### Data Preprocessing

# In[8]:


import re
import nltk

nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

all_stopwords = stopwords.words('english')
all_stopwords.remove('not')


# In[9]:


corpus=[]

for i in range(0, 900):
  review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
  review = review.lower()
  review = review.split()
  review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
  review = ' '.join(review)
  corpus.append(review)


# In[10]:


corpus


# ### Data transformation

# In[11]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1420)


# In[12]:


X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, -1].values


# In[13]:


# Saving BoW dictionary to later use in prediction
import pickle
bow_path = 'c1_BoW_Sentiment_Model.pkl'
pickle.dump(cv, open(bow_path, "wb"))


# ### Dividing dataset into training and test set

# In[14]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)


# ### Model fitting (Naive Bayes)

# In[15]:


from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)


# In[16]:


# Exporting NB Classifier to later use in prediction
import joblib
joblib.dump(classifier, 'c2_Classifier_Sentiment_Model') 


# ###Model performance

# In[17]:


y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)

accuracy_score(y_test, y_pred)

