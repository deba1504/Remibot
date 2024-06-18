import pandas as pd
import nltk
import numpy as np
import re
import random
from nltk.stem import wordnet
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import pos_tag
from sklearn.metrics import pairwise_distances
from nltk import word_tokenize
from nltk.corpus import stopwords
import warnings
warnings.filterwarnings('ignore')

df=pd.read_excel("remibot_dataset.xlsx")

def text_normalization(text):
    text=str(text).lower()
    special_char_text=re.sub(r'[^a-z]',' ',text)
    tokenized_text=nltk.word_tokenize(special_char_text)
    lemmatized_text=wordnet.WordNetLemmatizer()
    tags_list=pos_tag(tokenized_text,tagset=None)
    lema_words=[]
    for token,pos_token in tags_list:
        if pos_token.startswith('V'):
            pos_val='v'
        elif pos_token.startswith('J'):
            pos_val='a'
        elif pos_token.startswith('R'):
            pos_val='r'
        else:
            pos_val='n'
        lema_token=lemmatized_text.lemmatize(token,pos_val)
        lema_words.append(lema_token)
        
    return " ".join(lema_words)
    



def splitting_of_Input(u_input):
    global b
    q=[]
    a=u_input.split()
    for i in a:
        if i in stop:
            continue
        else:
            q.append(i)
        
        b=" ".join(q)         


Greeting_Inputs = ["hello", "hi", "greetings", "sup", "what's up", "hey", "hai", "hallo"]
Greeting_Responses = ["Hi. I will assist you.", "Hey. I will assist you.", "*nods*", "Hi there. I will assist you.", "Hello. I will assist you.", "I am glad you are talking to me. I will assist you."]
def greeting(sentence):
    for word in sentence.split():
        if word.lower() in Greeting_Inputs:
            return random.choice(Greeting_Responses)        

df['lemmatized_symptoms']=df['Symptom'].apply(text_normalization)



stop=stopwords.words('english')
stop_words = ['?',',','.','!','(',')']
for i in stop_words:
    stop.append(i)

def main_f(user_response):
    global tfidf
    global df_tfidf
    cv=CountVectorizer()
    x=cv.fit_transform(df['lemmatized_symptoms']).toarray()

    features=cv.get_feature_names_out()
    df_bow=pd.DataFrame(x,columns=features)

    splitting_of_Input(user_response)
        
    ques_lema=text_normalization(b)
    ques_bow=cv.transform([ques_lema]).toarray() 
       
    tfidf=TfidfVectorizer()
    x_tfidf=tfidf.fit_transform(df['lemmatized_symptoms']).toarray()
    
    df_tfidf=pd.DataFrame(x_tfidf,columns=tfidf.get_feature_names_out())
    ques_tfidf=tfidf.transform([ques_lema]).toarray()
    
    lemma=text_normalization(user_response)
    tf=tfidf.transform([lemma]).toarray()
    cos=1-pairwise_distances(df_tfidf,tf,metric='cosine')
    index_value=cos.argmax()
    inde_value=cos.argmin()
    
    if 'bye' not in user_response:
        if greeting(user_response) != None:
            return greeting(user_response)
        elif (index_value == inde_value):
            i_d = f"I don't understand that."
            return i_d
        else:
            for k in chat_tfidf(user_response):
                var_n = (f"The Natural Remedies for the following Symptoms are:\n {k}")  
            return var_n


def chat_tfidf(text):
    lemma=text_normalization(text)
    tf=tfidf.transform([lemma]).toarray()
    cos=1-pairwise_distances(df_tfidf,tf,metric='cosine')
    index_value=cos.argmax()
    inde_value=cos.argmin()
    return [df['Remedies'].loc[index_value]]

def chat(user_response):
    user_response = user_response.lower()
    if 'bye' not in user_response:
        if (user_response == "thanks") or (user_response == "thank you") or (user_response == "thank you so much") or (user_response == "thank you very much"):
            return ("Most Welcome\nBye Take Care...")
            flag = False
        else:
            return main_f(user_response)
    else:
        flag=False
        return "Bye! take care.."








