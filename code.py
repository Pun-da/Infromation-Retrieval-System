#!/usr/bin/env python
# coding: utf-8

# # 1. Collecting data set and Importing necessary libraries 

# #### 1.1 Import necessary libraries

# In[1]:


#importing all the libraries
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
import string
import numpy as np


# In[2]:


nltk.download('stopwords')
nltk.download('punkt')
nltk.download('popular')


# #### 1.2 Collecting the data

# In[3]:


#Reading input text files
text = []
names = []
for root, dir, files in os.walk('shakespeares-works_TXT_FolgerShakespeare'):
    for file in files:
        with open(os.path.join(root, file), 'r') as rd:
            text.append(rd.read())     #appending all the documents to text list
            names.append(file)        #appending all the document names to name list


# # 2. Removal of punctuation and stop words

# In[4]:


# remove punctuations
# tokenise the document
def tokenize(sentence):
    words = nltk.word_tokenize(sentence)
    token_words= [word for word in words if word.isalnum()]     #takes only the charecters which are either numbers or alphabets
    return token_words


# In[5]:


# remove stop words from tokens
stopwords = stopwords.words('english')
def stopwords_clr(sentence):
    tokens_clr= [token for token in sentence if token.lower() not in stopwords] #takes only the words which are not in stopwords
    return tokens_clr


# # 3. Normalization using Porter Stemmer

# In[6]:


#stemming the words to root form
stem = PorterStemmer()

def stem_tokens(sentence):
    tokens_stem = []
    for token in sentence:
        tokens_stem.append(stem.stem(token))     #stems the token and appends into tokens_stem list
    return tokens_stem


# # 4. Preprocessing data

# In[7]:


#tokenizes 'cont', removes stopwords and stems 'cont' 
def preprocess(cont):
    return " ".join(stopwords_clr(stem_tokens(stopwords_clr(tokenize(cont)))))      


# In[8]:


processed_data = []    #This contains the pre-processed data of each document


# In[9]:


for i in range(len(text)):                       #goes through all the documents in text list
  processed_data.append(preprocess(text[i]))     #appends the preprocessed document to preprocessed_data list 


# # 5. Construct inverted index

# In[10]:


inv_index = {}      #creating inverted index


# In[11]:


#Indexing the inputted document
def indexing(document, index):
    words = nltk.word_tokenize(document)          #tokenizes the document
    for word in words:                          
        if(inv_index.get(word) is None):          #check whether word is there in inv_index or not
            inv_index[word] = [index]               
        elif not index in inv_index.get(word):     
            inv_index.get(word).append(index)     


# In[12]:


for x in range(len(processed_data)):
    indexing(processed_data[x], x)          #indexing the preprocessd data of documents


# In[13]:


keys = list(inv_index.keys())       #Keys contains a list of all terms in the dictionary
postings = list(inv_index.values())    #Postings contain the posting list of all terms in the dictionary


# # 6. Spelling Correction

# In[14]:


#To find Levenshtein distance of two terms
def levenshtein_distance(term1, term2):
    term1 = term1.lower()
    term2 = term2.lower()
    dyn_mat = [[0 for x in range(len(term2) + 1)] for x in range(len(term1) + 1)]

    for x in range(len(term1) + 1):
        dyn_mat[x][0] = x
    for y in range(len(term2) + 1):
        dyn_mat[0][y] = y

    for x in range(1, len(term1) + 1):
        for y in range(1, len(term2) + 1):
            if term1[x - 1] == term2[y - 1]:
                dyn_mat[x][y] = min(
                    dyn_mat[x - 1][y] + 1,
                    dyn_mat[x - 1][y - 1],
                    dyn_mat[x][y - 1] + 1
                )
            else:
                dyn_mat[x][y] = min(
                    dyn_mat[x - 1][y] + 1,
                    dyn_mat[x - 1][y - 1] + 1,
                    dyn_mat[x][y - 1] + 1
                )

    return dyn_mat[len(term1)][len(term2)]


# In[15]:


#To find the nearest word in the dictionary to a misspelled word
def nearest_word(word):
  min = 100
  near = ''
  for key in keys:
    leven = levenshtein_distance(word, key)     #finding the levenshtein distance of word and key
    if leven == 0:                              
      min = leven                               
      near = key
      return near
    if leven < min:
      near = key
      min = leven
  return near


# In[16]:


def gen_posting(term, inv_index):        
  near = nearest_word(term)
  posting = inv_index[near]
  return posting


# # 7. Wildcard Queries

# In[17]:


def rotate(s, n):
    return s[n:] + s[:n]


# In[18]:


def bit_and(X, Y):
    return set(X).intersection(Y)


# In[19]:


def bit_or(X, Y):
    return set(X).union(Y)


# In[20]:


#Generating all permuterms for a term
def gen_perm(keys, per_index):
  for key in keys:
    okey = key + "$"
    for i in range(len(okey),0,-1):
      rot = rotate(okey, i)
      per_index[rot] = key
  return per_index


# In[21]:


#Find all appropriate permuterms and original terms
def find_perm(term, prefix):
    req_terms = []
    for key in term.keys():
        if key.startswith(prefix):
            req_terms.append(term[key])
    return req_terms


# In[22]:


#For cases 1, 2 and 3
def processQuery1(query, per_index):    
    req_terms = find_perm(per_index, query)
    #print(req_terms)

    post_ID = []
    for term in req_terms:
        post_ID.append(inv_index[term])
    #print(post_ID)

    coll = []
    for x in post_ID:
        for y in x:
            coll.append(y)   

    coll = [int(x) for x in coll]
    per = set(coll)
    per = list(per) 
    #print(per)    

    return per


# In[23]:


#For case 4 (X*Y*Z)
def processQuery2(que_part1, que_part2, per_index):

  #Part 1 = Z$X
  req_terms1 = find_perm(per_index, que_part1)
  #print(req_terms1)

  post_ID1 = []
  for term in req_terms1:
      post_ID1.append(inv_index[term])
  #print(post_ID1)

  coll1 = []
  for x in post_ID1:
      for y in x:
          coll1.append(y) 
  #print(coll1)  

  #Part 2 = Y
  req_terms2 = find_perm(per_index, que_part2)
  #print(req_terms2)

  post_ID2 = []
  for term in req_terms2:
      post_ID2.append(inv_index[term])
  #print(post_ID2)

  coll2 = []
  for x in post_ID2:
      for y in x:
          coll2.append(y) 
  #print(coll2)  

  #Intersecting the two posting lists obtained above

  coll1 = [int(x) for x in coll1]
  coll2 = [int(x) for x in coll2]

  coll_final = bit_and(coll1, coll2)
  per_final = set(coll_final)
  per_final = list(per_final)
  #print(per_final)

  return per_final


# In[24]:


#Decide case and process the wildcard query accordingly
def wildcard_process(query):
  out = []
  per_index = []
  comps = query.split('*')
  case = 0

  if len(comps) == 3:
    case = 4
  elif comps[1] == '':
    case = 1
  elif comps[0] == '':
    case = 2
  elif comps[0] != '' and comps[1] != '':
    case = 3

  per_index = {}
  per_index = gen_perm(keys, per_index)

  if case == 1:
    query = "$" + comps[0]
  elif case == 2:
    query = comps[1] + "$"
  elif case == 3:
    query = comps[1] + "$" + comps[0]
  elif case == 4:
    que_part1 = comps[2] + "$" + comps[0]
    que_part2 = comps[1]
    #print(que_part1, que_part2)

  if case != 4:
    out = processQuery1(query, per_index)
  elif case == 4:
    out = processQuery2(que_part1, que_part2, per_index)

  return out


# # 8. Boolean query

# In[25]:


#Processing a boolean query and finding the appropriate documents
def boolean_query(query, inv_index):
  terms = query.split(' ')
  bool_words = []
  diff_words = []

  for term in terms:
    if term.lower() != 'and' and term.lower() != 'or' and term.lower() != 'not':
      diff_words.append(term)
    else:
      bool_words.append(term)
  
  #print(bool_words, diff_words)
  
  posting_term = []
  posting_comb = []

  for term in diff_words:
    if '*' in term:
      posting_term = wildcard_process(term)
      posting_comb.append(posting_term)
    else:
      posting_term = gen_posting(term, inv_index)
      posting_comb.append(posting_term)

  #print(posting_comb)


  i = 0
  x = 0
  z = len(bool_words)
    
  while i < z:
    
    if bool_words[x] == 'not':
      all_docs = set(list(range(len(processed_data))))
      res = list(all_docs - set(posting_comb[x]))
      posting_comb.remove(posting_comb[x])
      posting_comb.insert(x, res)
      bool_words.remove(bool_words[x])
      i = i + 1
    
    elif bool_words[x] == 'and':
        if (x + 1) < len(bool_words) and bool_words[x + 1] == 'not':
            all_docs = set(list(range(len(processed_data))))
            res = list(all_docs - set(posting_comb[x + 1]))
            bool_words.remove(bool_words[x + 1])
            i = i + 1
        else:
            res = posting_comb[x + 1]
        intersection = list(set(posting_comb[x]).intersection(res))
        posting_comb.remove(posting_comb[x])
        posting_comb.remove(posting_comb[x])
        posting_comb.insert(x, intersection)
        bool_words.remove(bool_words[x])
        i = i + 1
        
    elif bool_words[x] == 'or':
        x = x + 1
        i = i + 1
        
  #print(posting_comb)
  #print(bool_words)
    
  i = 0      
  while i < len(bool_words):
    union = posting_comb[0] + list(set(posting_comb[1]) - set(posting_comb[0]))
    #print(union)
    posting_comb.remove(posting_comb[0])
    posting_comb.remove(posting_comb[0])
    posting_comb.insert(0, union)
    i = i + 1
         
      
  #print(posting_comb)
  return posting_comb[0]


# In[26]:


inp_query = input()

out = []
out = boolean_query(inp_query, inv_index)
output = open("OUTPUT_Documents.txt","w")

for x in out:
  output.write(names[x] + '\n')

output.close()

