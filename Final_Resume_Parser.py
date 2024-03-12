#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import spacy
import io
from spacy.matcher import Matcher
import pandas as pd
import re
from nltk.corpus import stopwords
import constants as cs
import datetime
from datetime import datetime
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize 
import docx2txt
import subprocess
import docxpy
import glob
import numpy as np
# load pre-trained model
nlp = spacy.load('en_core_web_sm')
import warnings
warnings.filterwarnings('ignore')


# In[2]:


cs


# In[29]:


# FOR INDIAN RESUME RUN THE BELOW FUNCTION TO EXTRACT MOBILE NUMBER
def extract_mobile_number(text):
    phone= re.findall(r'[8-9]{1}[0-9]{9}',text)
    
    if len(phone) > 10:
        return '+' + phone
    else:
        return phone

def extract_email(text):
        email = re.findall("([^@|\s]+@[^@]+\.[^@|\s]+)", text)
        if email:
            try:
                return email[0].split()[0].strip(';')
            except IndexError:
                return None

# Function to remove punctuation and tokenize the text
def tokenText(extText):
   
    # Remove punctuation marks
    punc = '''!()-[]{};:'"\,.<>/?@#$%^&*_~'''
    for ele in extText:
        if ele in punc:
            puncText = extText.replace(ele, "")
            
    # Tokenize the text and remove stop words
    stop_words = set(stopwords.words('english'))
    puncText.split()
    word_tokens = word_tokenize(puncText)
    TokenizedText = [w for w in word_tokens if not w.lower() in stop_words]
    TokenizedText = []
  
    for w in word_tokens:
        if w not in stop_words:
            TokenizedText.append(w)
    return(TokenizedText)            

# Function to extract Name and contact details
def extract_name(Text):
    name = ''  
    for i in range(0,3):
        name = " ".join([name, Text[i]])
    return(name)

# Grad all general stop words
STOPWORDS = set(stopwords.words('english'))

# Education Degrees
EDUCATION = ['BE','B.E.', 'B.E', 'BS','B.S','B.Com','BCA','ME','M.E', 'M.E.', 'M.S','B.com','10','10+2','BTECH', 'B.TECH', 'M.TECH', 'MTECH', 'SSC', 'HSC', 'C.B.S.E','CBSE','ICSE', 'X', 'XII','10th','12th',' 10th',' 12th','Bachelor of Arts in Mathematics','Master of Science in Analytics','Bachelor of Business Administration','Major: Business Management']

def extract_education(text):
    nlp_text = nlp(text)

    # Sentence Tokenizer
    nlp_text = [sent.text.strip() for sent in nlp_text.sents]


    edu = {}
    # Extract education degree
    for index, t in enumerate(nlp_text):
        for tex in t.split():
            # Replace all special symbols
            tex = re.sub(r'[?|$|.|!|,]', r'', tex)
            if tex in EDUCATION and tex not in STOPWORDS:
                edu[tex] = t + nlp_text[index + 1]

    # Extract year
    education = []
    for key in edu.keys():
        year = re.search(re.compile(r'(((20|19)(\d{2})))'), edu[key])
        if year:
            education.append((key, ''.join(year[0])))
        else:
            education.append(key)
    return education

def extract_skills(resume_text):

        nlp_text = nlp(resume_text)
        noun_chunks = nlp_text.noun_chunks

        # removing stop words and implementing word tokenization
        tokens = [token.text for token in nlp_text if not token.is_stop]
        
        # reading the csv file
        data = pd.read_csv("skills.csv") 
        
        # extract values
        skills = list(data.columns.values)
        
        skillset = []
        
        # check for one-grams (example: python)
        for token in tokens:
            if token.lower() in skills:
                skillset.append(token)
        
        # check for bi-grams and tri-grams (example: machine learning)
        for token in noun_chunks:
            token = token.text.lower().strip()
            if token in skills:
                skillset.append(token)
        
        return [i.capitalize() for i in set([i.lower() for i in skillset])]



def string_found(string1, string2):
        if re.search(r"\b" + re.escape(string1) + r"\b", string2):
            return True
        return False

def extract_entity_sections_grad(text):
    '''
    Helper function to extract all the raw text from sections of resume specifically for 
    graduates and undergraduates
    :param text: Raw text of resume
    :return: dictionary of entities
    '''
    text_split = [i.strip() for i in text.split('\n')]
    entities = {}
    key = False
    for phrase in text_split:
        if len(phrase) == 1:
            p_key = phrase
        else:
            p_key = set(phrase.lower().split()) & set(cs.RESUME_SECTIONS_GRAD)
        try:
            p_key = list(p_key)[0]
        except IndexError:
            pass
        if p_key in cs.RESUME_SECTIONS_GRAD:
            entities[p_key] = []
            key = p_key
        elif key and phrase.strip():
            entities[key].append(phrase)
    return entities 

# Function to extract experience details
def expDetails(Text):
    global sent
   
    Text = Text.split()
   
    for i in range(len(Text)-2):
        Text[i].lower()
        
        if Text[i] ==  'years':
            sent =  Text[i-2] + ' ' + Text[i-1] +' ' + Text[i] +' '+ Text[i+1] +' ' + Text[i+2]
            l = re.findall('\d*\.?\d+',sent)
            for i in l:
                a = float(i)
            return(a)
            return (sent)

def extract_experience(resume_text):
    '''
    Helper function to extract experience from resume text
    :param resume_text: Plain resume text
    :return: list of experience
    '''
    wordnet_lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    # word tokenization 
    word_tokens = nltk.word_tokenize(resume_text)

    # remove stop words and lemmatize  
    filtered_sentence = [w for w in word_tokens if not w in stop_words and wordnet_lemmatizer.lemmatize(w) not in stop_words] 
    sent = nltk.pos_tag(filtered_sentence)

    # parse regex
    cp = nltk.RegexpParser('P: {<NNP>+}')
    cs = cp.parse(sent)
    
    # for i in cs.subtrees(filter=lambda x: x.label() == 'P'):
    #     print(i)
    
    test = []
    
    for vp in list(cs.subtrees(filter=lambda x: x.label()=='P')):
        test.append(" ".join([i[0] for i in vp.leaves() if len(vp.leaves()) >= 2]))

    # Search the word 'experience' in the chunk and then print out the text after it
    x = [x[x.lower().index('experience') + 10:] for i, x in enumerate(test) if x and 'experience' in x.lower()]
    return x

def string_found(string1, string2):
        if re.search(r"\b" + re.escape(string1) + r"\b", string2):
            return True
        return False

def get_score(_dict):
    _len = len(_dict)
    if _len >= 5:
        return 1
    elif _len < 5 and _len > 2:
        return 0.5
    elif _len  == 1:
        return 0.2
    else:
        return 0

def extract_competencies(text, experience_list):
    '''
    Helper function to extract competencies from resume text
    :param resume_text: Plain resume text
    :return: dictionary of competencies
    '''
    experience_text = ' '.join(experience_list)
    competency_dict = {}
    score = 0

    percentage = (100 // len(cs.COMPETENCIES.keys()))

    for competency in cs.COMPETENCIES.keys():
        matches = {}
        for item in cs.COMPETENCIES[competency]:
            if string_found(item, experience_text):
                if competency not in competency_dict.keys():
                    match = re.search(r'([^.|,]*' + item + '[^.|,]*)', experience_text)
                    if item not in matches.keys():
                        matches[item] = [match.group(0)]
                    else:
                        for i in match.groups():
                            matches[item].append(i)    
                    competency_dict[competency] = matches
                else:
                    match = re.search(r'([^.|,]*' + item + '[^.|,]*)', experience_text)
                    if item not in matches.keys():
                        matches[item] = [match.group(0)]
                    else:
                        for i in match.groups():
                            matches[item].append(i)
                    competency_dict[competency] = matches
                score += get_score(competency_dict[competency]) * percentage
    
    competency_dict['score'] = score 
    list=list(competency_dict.keys())
    return(list)

def extract_competencies_score(text, experience_list):
        '''
        Helper function to extract competencies from resume text
        :param resume_text: Plain resume text
        :return: dictionary of competencies
        '''
        experience_text = ' '.join(experience_list)
        competency_dict = {}
        score = 0

        percentage = (100 // len(cs.COMPETENCIES.keys()))

        for competency in cs.COMPETENCIES.keys():
            matches = {}
            for item in cs.COMPETENCIES[competency]:
                if string_found(item, experience_text):
                    if competency not in competency_dict.keys():
                        match = re.search(r'([^.|,]*' + item + '[^.|,]*)', experience_text)
                        if item not in matches.keys():
                            matches[item] = [match.group(0)]
                        else:
                            for i in match.groups():
                                matches[item].append(i)    
                        competency_dict[competency] = matches
                    else:
                        match = re.search(r'([^.|,]*' + item + '[^.|,]*)', experience_text)
                        if item not in matches.keys():
                            matches[item] = [match.group(0)]
                        else:
                            for i in match.groups():
                                matches[item].append(i)
                        competency_dict[competency] = matches
                    score += get_score(competency_dict[competency]) * percentage
        
        competency_dict['score'] = score 
        return(competency_dict['score'])

def extract_dob(text):
        
    result1=re.findall(r"[\d]{1,2}/[\d]{1,2}/[\d]{4}",text)
    result2=re.findall(r"[\d]{1,2}-[\d]{1,2}-[\d]{4}",text)           
    result3= re.findall(r"[\d]{1,2} [ADFJMNOSadfjmnos]\w* [\d]{4}",text)
    result4=re.findall(r"([\d]{1,2})\.([\d]{1,2})\.([\d]{4})",text)
                
    l=[result1,result2,result3,result4]
    for i in l:
        if i==[]:
            continue
        else:
            return i


def extract_text_from_docx(path):
    '''
    Helper function to extract plain text from .docx files
    :param doc_path: path to .docx file to be extracted
    :return: string of extracted text
    '''
    try:
        temp = docx2txt.process(path)
        return temp
    except KeyError:
        return ' '


# In[30]:


df = pd.DataFrame(columns=['Name','Mobile No.', 'Email','DOB','Education Qualifications','Skills','Experience (Years)','Last Position','Competence','competence score'], dtype=object)


# ### For Single Files

# In[31]:


i=0
path_input = r"C:/Users\Moin Dalvi\Data_Science\Projects\Resume_Classification/Resumes_docx/Peoplesoft/Peoplesoft Admin_AnubhavSingh.docx"
if path_input.endswith('.docx'):
    text = extract_text_from_docx(path_input)
    tokText = tokenText(text)
    df.loc[i,'Name']=extract_name(tokText)
    df.loc[i,'Mobile No.']=extract_mobile_number(text)
    df.loc[i,'Email']=extract_email(text)
    df.loc[i,'DOB']=extract_dob(text)
    df.loc[i,'Education Qualifications']=extract_education(text)
    df.loc[i,'Skills']=extract_skills(text)
    df.loc[i,'Experience (Years)']=expDetails(text) 
    experience_list1=extract_entity_sections_grad(text) 

    if 'experience' in experience_list1:
        i=0
        experience_list=experience_list1['experience']
        df.loc[i,'Last Position']=extract_experience(text)
        df.loc[i,'Competence']=extract_competencies(text,experience_list)
        df.loc[i,'competence score']=extract_competencies_score(text,experience_list)

    else:
        df.loc[i,'Last Position']='NA'
        df.loc[i,'Competence']='NA'
        df.loc[i,'competence score']='NA'
    i+=1


# In[32]:


df


# In[ ]:




