import nltk
import numpy as np
import random
import string 
import warnings
import re, string, unicodedata
import pandas as pd
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer
import wikipedia as wk
from collections import defaultdict
warnings.filterwarnings('ignore')
'''nltk.download('punkt') # first-time use only
nltk.download('wordnet') # first-time use only'''

data=open('dataset.txt','r',errors = 'ignore')
raw=data.read()
raw = raw.lower()

sent_tokens = nltk.sent_tokenize(raw)

def Normalize(text):
    remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)

    word_token = nltk.word_tokenize(text.lower().translate(remove_punct_dict))

    new_words = []
    for word in word_token:
        new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        new_words.append(new_word)

    rmv = []
    for w in new_words:
        text=re.sub("&lt;/?.*?&gt;","&lt;&gt;",w)
        rmv.append(text)

    tag_map = defaultdict(lambda : wn.NOUN)
    tag_map['J'] = wn.ADJ
    tag_map['V'] = wn.VERB
    tag_map['R'] = wn.ADV
    lmtzr = WordNetLemmatizer()
    lemma_list = []
    rmv = [i for i in rmv if i]
    for token, tag in nltk.pos_tag(rmv):
        lemma = lmtzr.lemmatize(token, tag_map[tag[0]])
        lemma_list.append(lemma)
    return lemma_list

GREETING_INPUTS = ("hello", "hi","hey")
GREETING_RESPONSES = ["hi, how can i help you?", "hey, how can i help you?", "hi there, how can i help you?", "hello , how can i help you?", "I am glad! You are talking to me"]
def greeting(your_response):
    for word in your_response.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel

def response(your_response):
    ranger_response=''
    sent_tokens.append(your_response)
    TfidfVec = TfidfVectorizer(tokenizer=Normalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = linear_kernel(tfidf[-1], tfidf)
    idx=vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if(req_tfidf==0) or "tell me about" in your_response:
        print("\n\nChecking The Web...")
        if your_response:
            ranger_response = wikipedia_data(your_response)
            return ranger_response
    else:
        ranger_response = ranger_response+sent_tokens[idx]
        return ranger_response

def wikipedia_data(input):
    reg_ex = re.search('tell me about (.*)', input)
    try:
        if reg_ex:
            topic = reg_ex.group(1)
            ny = wk.summary(topic, sentences = 3)
            return ny
    except Exception as e:
            print(e)

flag=True
print("\n\nRanger : Hey! I am Corona Ranger and I am a chatbot. How Can i help?\nTo get answered, type the corresponding keyword.")
print("1. What is Corona Virus ? - cvirus")
print("2. FullForm of COVID-19 - fullform")
print("3. Local News - local")
print("4. International News - inter")
print("5. Total Cases - totca")
print("6. Total Deaths - totda")
print("7. Total Recovered - totre")
print("8. Self Diagnosis - selfda")
print("9. In case you want to search anything else type - 'tell me about xxxxx'  where xxxx is your keyword")
print("10. To exit - bye")
while(flag==True):
    your_response = input("You : ")
    your_response=your_response.lower()
    if(your_response!='bye'):
        if(your_response=='thanks' or your_response=='thank you' ):
            flag=False
            print("\nRanger : You are welcome...")
        else:
            if(greeting(your_response)!=None):
                print("\nRanger : "+greeting(your_response))
            else:
                print("\nRanger : "+response(your_response))
                print("")
                sent_tokens.remove(your_response)
    else:
        flag=False
        print("\nRanger : Bye! take care...")