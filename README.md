# Remibot-AI ML project
## Introduction
RemitiBot uses machine learning technology to address the problem of diagnosing health problems from symptoms. By utilizing this feature, RemiBot distinguishes itself by being able to identify possible illnesses and offer workable remedies. In contrast to conventional symptom checkers, RemiBot provides a special benefit by recommending Ayurvedic treatments that are easily found in your kitchen and that can aid in symptom relief and the advancement of natural healing. With the help of easily obtainable and efficient home remedies, this novel approach enables users to take control of their health.

## Problem Statment
Develop a disease predictor that can accurately predict diseases based on user-input symptoms and medical history, and recommend appropriate remedies or treatments for the predicted diseases.

## Solution
### Disease Prediction
Develop a model that accurately predicts diseases based on symptoms and medical history, considering shared symptoms between diseases
### User Interface
Create a user-friendly interface for inputting symptoms, viewing predictions, and providing feedback.
### Remedy Recommendation
Design an algorithm that recommends safe and effective remedies for predicted diseases, considering severity and user's medical history.
### Data Quality
Collect and maintain a reliable dataset of diseases, symptoms, and remedies.

## Technology
1. Python: Python is a versatile programming language widely used for data analysis, machine learning, and natural language processing (NLP) tasks due to its simplicity and readabilty.
2. Pandas: Pandas is a powerful library in Python used for data manipulation and analysis. It provides data structures like Data Frames, which are useful for handling structured data.
3.NLTK (Natural Language Toolkit): NLTK is a library in Python used for natural language processing tasks such as tokenization, stemming, lemmatization, part-of-speech tagging, and more
4. NumPy: NumPy is a fundamental package for scientific computing in Python. It provides support for large, multi-dimensional arrays and matrices, along with a collection of mathematical functions to operate on these arrays.
5. Regular Expressions (re): The re module in Python provides support for regular expressions which are used for pattern matching in text data. They are useful for tasks such as cleaning and preprocessing text.
6. Scikit-leam: Scikit-lean is a machine learning library in Python that provides simple and efficient tools for data mining and data analysis. It includes various algorithms for clustering, classification, regression, and more.

## Approach
1. Data Preprocessing: The code likely involves loading a dataset containing text data, such as sentences or documents. The data is then preprocessed to remove any irrelevant information, such as punctuation, special characters, and stopwords (common words like "and," "the," etc.).
2. Tokenization: Tokenization is the process of splitting text into individual words or tokens. NLTK's word_tokenize function is likely used for this purpose.
3. Lemmatization: Lemmatization is the process of reducing words to their base or root form. The NLTK library's WordNetLemmatizer is likely used for this task. 4. Vectorization: Vectorization is the process of converting text data into numerical form so that it can be used in machine
leaming algorithms. This is often done using techniques like CountVectorizer or TfidfVectorizer, which convert text into a
matrix of token counts or term frequencies, respectively. 5. Similarity Calculation: The code may involve calculating the similarity between text documents or sentences. This is typically done using techniques like cosine similarity, which measures the cosine of the angle between two vectors.
6. Machine Learning: Depending on the specific task, the code may involve using machine learning algorithms to classify text, cluster documents, or perform other NLP tasks. Scikit-lean provides a range of algorithms that can be used for these purposes.
