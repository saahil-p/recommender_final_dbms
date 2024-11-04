#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# In[ ]:


courses_df = pd.read_csv('/Users/saahil/Desktop/College/Sem 5/DBMS Lab/Project/coursera_courses.csv')

# job_df.head()

job_df = pd.read_csv('/Users/saahil/Desktop/College/Sem 5/DBMS Lab/Project/job_data_filtered.csv')
# courses_df.head()


# In[ ]:


job_df.head()


# In[ ]:


courses_df.head()


# In[ ]:


# Initialize the stemmer
stemmer = PorterStemmer()

stop_words = set(stopwords.words('english'))
# Define a function to stem words in each course description
def stem_text(text):
    if isinstance(text, str):  # Check if the entry is a string
        words = nltk.word_tokenize(text)
        filtered_words = [word for word in words if word.lower() not in stop_words]
        return ' '.join(stemmer.stem(word) for word in filtered_words)
    else:
        return ""  # Return an empty string if it's not a valid string


# Apply the function to the course_description column
courses_df['course_description_stemmed'] = courses_df['course_description'].apply(stem_text)

courses_df.drop(columns = ['course_description'], inplace = True)


# In[ ]:


courses_df.head()


# In[ ]:


# job_df.head()
job_df['job_description'] = job_df['job_description'].apply(stem_text)


# In[ ]:


job_df.head()


# In[ ]:


def vectorise_text(df,col,lang):
  tfidf_vectorizer = TfidfVectorizer(stop_words=lang)
  tfidf_matrix = tfidf_vectorizer.fit_transform(df[col].fillna(''))
  cosine_sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
  return cosine_sim_matrix


# In[ ]:


course_desc_vectorised = vectorise_text(courses_df,'course_description_stemmed','english')
job_desc_vectorised = vectorise_text(job_df,'job_description','english')


# In[ ]:


print(course_desc_vectorised)
print(job_desc_vectorised)


# In[ ]:


tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(courses_df['course_description_stemmed'].fillna(''))


def recommend_courses(student_input, tfidf_matrix, courses_df):
    # Transform the student input into a TF-IDF vector using the same vectorizer
    student_input_vector = tfidf_vectorizer.transform([student_input])
    
    # Compute cosine similarity between the student input vector and all course descriptions
    similarity_scores = cosine_similarity(student_input_vector, tfidf_matrix).flatten()
    
    # Get indices of the top 5 highest similarity scores
    top_indices = similarity_scores.argsort()[-5:][::-1]
    
    print(top_indices)
    # Retrieve the recommended courses
    recommended_courses = courses_df.iloc[top_indices]
    
    return recommended_courses


# In[ ]:


student_input = "I am interested in data science and machine learning"

# Get the top 5 recommended courses
recommended_courses = recommend_courses(student_input, tfidf_matrix, courses_df)
# print(recommended_courses[['course_title']])


# In[ ]:




