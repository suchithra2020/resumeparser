#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().system('pip install docx')


# In[2]:


# Importing 
# Libraries
import plotly.io as pio
import seaborn as sns
sns.set_style('darkgrid')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from pdfminer.converter import TextConverter
from pdfminer.pdfinterp import PDFPageInterpreter
from pdfminer.pdfinterp import PDFResourceManager
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
import spacy
import io
from spacy.matcher import Matcher
import pandas as pd
import re
from nltk.corpus import stopwords
import constants as cs
import datetime
from datetime import datetime
from dateutil import relativedelta
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
nltk.download('stopwords')
import docx2txt
import subprocess
import docxpy
import shutil

import re
import time
import string
import warnings
from tqdm.notebook import tqdm_notebook
import os,re
import textract as tr

# for all NLP related operations on text 
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import *
from nltk.stem.porter import PorterStemmer
from wordcloud import WordCloud, STOPWORDS
from collections import Counter

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# To identify the sentiment of text
from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer
from textblob.np_extractors import ConllExtractor

from sklearn.feature_extraction.text import CountVectorizer 
from nltk.tokenize import RegexpTokenizer

# ignoring all the warnings
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings("ignore", category=DeprecationWarning)


# In[2]:


file_path = r'C:/Users\Moin Dalvi\Documents\Data Science Material\Projects\Resume Classification\Resumes/'
doc_file = []
pdf_file = []
docx_file = []
folder_name = []
for folder in os.listdir(file_path):
    folder_path = file_path+folder
    for file in os.listdir(folder_path):
        if file.endswith('.doc'):
            doc_file.append(file)
            folder_name.append(folder)
        elif file.endswith('.docx'):
            docx_file.append(file)
            folder_name.append(folder)
        else:
            pdf_file.append(file)
            folder_name.append(folder)


# In[3]:


print('Number of .doc files = {}'.format(len(doc_file)),'\n'
     'Number of .pdf files = {}'.format(len(pdf_file)),'\n'
     'Number of .docx files = {}'.format(len(docx_file)))


# In[4]:


len(docx_file)+len(doc_file)+len(pdf_file)


# In[5]:


['docx_file','doc_file','pdf_file']


# In[6]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[7]:


f, axe = plt.subplots(1,1, figsize=(8,8), dpi=100)
ax = sns.barplot(x=['docx_file','doc_file','pdf_file'], y=[len(docx_file),len(doc_file),len(pdf_file)], ax = axe
            ,label='Total Resumes = {}'.format(len(docx_file)+len(doc_file)+len(pdf_file)))
axe.set_xlabel('Extensions', size=12,fontweight = 'bold')
axe.set_ylabel('Frequency', size=12,fontweight = 'bold')
plt.yticks(fontsize=12,fontweight = 'bold')
plt.xticks(fontsize=12,fontweight = 'bold')
plt.legend(loc='best', fontsize  = 'large')
plt.title('Type of Files in Resumes', fontsize = 14, fontweight = 'bold')
for i in ax.containers:
    ax.bar_label(i,color = 'black', fontweight = 'bold', fontsize= 12)
plt.show()


# In[8]:


plt.figure(figsize=(12,8), dpi = 100)
# Setting size in Chart based on 
# given values

sizes = [len(docx_file),len(doc_file),len(pdf_file)]
  
# Setting labels for items in Chart
labels = ['docx_file','doc_file','pdf_file']
  
# colors
colors = ['#e3342f', '#f6993f', '#ffed4a']
  
# explosion
explode = (0.02, 0.03, 0.04)
  
# Pie Chart
plt.pie(sizes, colors=colors, labels=labels,
        autopct='%1.0f%%', shadow=True,
        pctdistance=0.85, 
        explode=explode,
        startangle=0,
        textprops = {'size':'x-large',
                   'fontweight':'bold',
                    'rotation':'0',
                   'color':'black'})
  
# draw circle
centre_circle = plt.Circle((0, 0), 0.70, fc='white')
fig = plt.gcf()
  
# Adding Circle in Pie chart
fig.gca().add_artist(centre_circle)

# Adding Title of chart
plt.title('Percentage of Extensions in Resumes', fontsize = 16, fontweight = 'bold')
  
# Add Legends
plt.legend(labels, loc="center")
  
# Displaying Chart

plt.show()


# In[2]:


file_path = r'C:/Users\Moin Dalvi\Data_Science\Projects\Resume_Classification\Resumes_docx/'
file_name = []
profile = []
for folder in os.listdir(file_path):
    folder_path = file_path+folder
    for file in os.listdir(folder_path):
        if file.endswith('.docx'):
            profile.append(folder)
            file_name.append(file)


# In[3]:


data = pd.DataFrame()
data['Resumes'] = file_name
data['Profile'] = profile
data


# In[15]:


data.Profile.value_counts().index


# In[16]:


data.Profile.value_counts()


# In[13]:


plt.figure(figsize=(12,8)) 
# Setting size in Chart based on 
# given values

sizes = data.Profile.value_counts()
  
# Setting labels for items in Chart
labels = data.Profile.value_counts().index
  
# colors
colors = ['#e3342f', '#f6993f', '#ffed4a', '#38c172', '#4dc0b5', '#3490dc', '#6574cd', '#9561e2', '#f66d9b' ]
  
# explosion
explode = (0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1)
  
# Pie Chart
plt.pie(sizes, colors=colors, labels=labels,
        autopct=lambda x: '{:.0f}'.format(x*sizes.sum()/100), shadow=True,
        pctdistance=0.85, 
        explode=explode,
        startangle=0,
        textprops = {'size':'large',
                   'fontweight':'bold',
                    'rotation':'0',
                   'color':'black'})
  
# draw circle
centre_circle = plt.Circle((0, 0), 0.70, fc='white')
fig = plt.gcf()
  
# Adding Circle in Pie chart
fig.gca().add_artist(centre_circle)

# Adding Title of chart
plt.title('Number of Profiles in Resumes', fontsize = 16, fontweight = 'bold')
  
# Add Legends
plt.legend(labels, loc="center")
  
# Displaying Chart

plt.show()


# In[14]:


f, axe = plt.subplots(1,1, figsize=(18,6), dpi=200)
ax = sns.barplot(x=data.Profile.value_counts().index, y=data.Profile.value_counts(), data=data, ax = axe,
            label='Total Pofile Types = {}'.format(len(data.Profile.unique())))
axe.set_xlabel('Profiles', size=12,fontweight = 'bold')
axe.set_ylabel('Frequency', size=12,fontweight = 'bold')
plt.yticks(fontsize=14,fontweight = 'bold')
plt.xticks(fontsize=14,fontweight = 'bold', rotation = 45)
plt.legend(loc='best', fontsize  = 'x-large')
plt.title('Number of Profiles in Resumes', fontsize = 16, fontweight = 'bold')
for i in ax.containers:
    ax.bar_label(i,color = 'black', fontweight = 'bold', fontsize= 14)
plt.show()


# In[15]:


plt.figure(figsize=(12,8),dpi=100) 
# Setting size in Chart based on 
# given values

sizes = data.Profile.value_counts()
  
# Setting labels for items in Chart
labels = data.Profile.value_counts().index
  
# colors
colors = ['#e3342f', '#f6993f', '#ffed4a', '#38c172', '#4dc0b5', '#3490dc', '#6574cd', '#9561e2', '#f66d9b' ]
  
# explosion
explode = (0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1)
  
# Pie Chart
plt.pie(sizes, colors=colors, labels=labels,
        autopct='%1.0f%%', shadow=True,
        pctdistance=0.85, 
        explode=explode,
        startangle=0,
        textprops = {'size':'large',
                   'fontweight':'bold',
                    'rotation':'0',
                   'color':'black'})
  
# draw circle
centre_circle = plt.Circle((0, 0), 0.70, fc='white')
fig = plt.gcf()
  
# Adding Circle in Pie chart
fig.gca().add_artist(centre_circle)

# Adding Title of chart
plt.title('Percentage of Profiles in Resumes', fontsize = 16, fontweight = 'bold')
  
# Add Legends
plt.legend(labels, loc="center")
  
# Displaying Chart

plt.show()


# In[6]:


plt.figure(figsize=(12,8)) 
# Setting size in Chart based on 
# given values

sizes = data.Profile.value_counts()
  
# Setting labels for items in Chart
labels = data.Profile.value_counts().index
  
# colors
colors = ['#3490dc', '#f6993f','#38c172', '#e3342f']
  
# explosion
explode = (0.01, 0.01, 0.01, 0.01)
  
# Pie Chart
plt.pie(sizes, colors=colors, labels=labels,
        autopct=lambda x: '{:.0f}'.format(x*sizes.sum()/100), shadow=True,
        pctdistance=0.85, 
        explode=explode,
        startangle=0,
        textprops = {'size':'large',
                   'fontweight':'bold',
                    'rotation':'0',
                   'color':'black'})
  
# draw circle
centre_circle = plt.Circle((0, 0), 0.70, fc='white')
fig = plt.gcf()
  
# Adding Circle in Pie chart
fig.gca().add_artist(centre_circle)

# Adding Title of chart
plt.title('Number of Classes in Dataframe', fontsize = 16, fontweight = 'bold')
  
# Add Legends
plt.legend(labels, loc="center")
  
# Displaying Chart

plt.show()


# In[5]:


f, axe = plt.subplots(1,1, figsize=(18,6), dpi=200)
#colors = ['#e3342f', '#f6993f', '#3490dc', '#38c172']
ax = sns.barplot(x=data.Profile.value_counts().index, y=data.Profile.value_counts(), data=data, ax = axe,# color=colors,
            label='Total Pofile Types = {}'.format(len(data.Profile.unique())))
axe.set_xlabel('Profiles', size=12,fontweight = 'bold')
axe.set_ylabel('Frequency', size=12,fontweight = 'bold')
plt.yticks(fontsize=14,fontweight = 'bold')
plt.xticks(fontsize=14,fontweight = 'bold', rotation = 45)
plt.legend(loc='best', fontsize  = 'x-large')
plt.title('Number of Classes in Dataframe', fontsize = 16, fontweight = 'bold')
for i in ax.containers:
    ax.bar_label(i,color = 'black', fontweight = 'bold', fontsize= 14)
plt.show()


# In[7]:


plt.figure(figsize=(12,8),dpi=100) 
# Setting size in Chart based on 
# given values

sizes = data.Profile.value_counts()
  
# Setting labels for items in Chart
labels = data.Profile.value_counts().index
  
# colors
colors = ['#3490dc', '#f6993f','#38c172', '#e3342f']
  
# explosion
explode = (0.0, 0.0, 0.0, 0.0)
  
# Pie Chart
plt.pie(sizes, colors=colors, labels=labels,
        autopct='%1.0f%%', shadow=True,
        pctdistance=0.85, 
        explode=explode,
        startangle=0,
        textprops = {'size':'large',
                   'fontweight':'bold',
                    'rotation':'0',
                   'color':'black'})
  
# draw circle
centre_circle = plt.Circle((0, 0), 0.70, fc='white')
fig = plt.gcf()
  
# Adding Circle in Pie chart
fig.gca().add_artist(centre_circle)

# Adding Title of chart
plt.title('Percentage of Classes in Dataframe', fontsize = 16, fontweight = 'bold')
  
# Add Legends
plt.legend(labels, loc="center")
  
# Displaying Chart

plt.show()


# ### Reading a Resume File

# In[124]:


import docx2txt
 
def extract_text_from_docx(docx_path):
    txt = docx2txt.process(docx_path)
    if txt:
        return txt.replace('\t', ' ')
    return None
 
if __name__ == '__main__':
    print(extract_text_from_docx('C:/Users\Moin Dalvi\Documents\Data Science Material\Projects\Resume Classification\Resumes\React JS Developer/React JS Developer_KotaniDurgaprasad[3_1] (1)-converted.docx'))


# In[4]:


df = pd.read_csv('Resume2Text_Extracted.csv')


# In[5]:


df


# In[6]:


df[df.Profiles=='SQL Developer']


# ## <a id='2'>Data Exploration</a>

# In[15]:


df[df.duplicated()].shape


# In[16]:


df[df.duplicated()]


# In[17]:


df.isnull().any()


# In[18]:


df.isnull().sum()


# #### Number of Words

# In[19]:


df['word_count'] = df['Resumes'].apply(lambda x: len(str(x).split(" ")))
df[['Resumes','word_count']].head()


# In[20]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[21]:


skew = df['word_count'].skew()
kurt = df['word_count'].kurt()
mean = df['word_count'].mean()
std = df['word_count'].std()
plt.figure(figsize=(12,8))
sns.set(font_scale=1.2)
sns.distplot(df["word_count"], kde= True, label='Skewness = {:.2f},\nkurtosis = {:.2f},\nmean = {:.2f},\nStandard Deviation = {:.2f}'.format(skew, kurt, mean, std), bins=30).set(title='Word Count in Resumes')
plt.legend(loc='best')
plt.show()


# #### Number of characters

# In[22]:


df['char_count'] = df['Resumes'].str.len() ## this also includes spaces
df[['Resumes','char_count']].head()


# In[45]:


skew = df['char_count'].skew()
kurt = df['char_count'].kurt()
mean = df['char_count'].mean()
std = df['char_count'].std()
plt.figure(figsize=(12,8))
sns.set(font_scale=1.2)
sns.distplot(df["char_count"], kde= True, label='Skewness = {:.2f},\nkurtosis = {:.2f},\nmean = {:.2f},\nStandard Deviation = {:.2f}'.format(skew, kurt, mean, std), bins=30).set(title='Characters in Resumes')
plt.legend(loc='best')
plt.show()


# #### Average Word Length in each Resumes

# In[23]:


def avg_word(sentence):
    words = sentence.split()
    return (sum(len(word) for word in words)/len(words))

df['avg_word'] = df['Resumes'].apply(lambda x: avg_word(x))
df[['Resumes','avg_word']].head()


# In[49]:


skew = df['avg_word'].skew()
kurt = df['avg_word'].kurt()
mean = df['avg_word'].mean()
std = df['avg_word'].std()
plt.figure(figsize=(12,8))
sns.set(font_scale=1.2)
sns.distplot(df["avg_word"], kde= True, label='Skewness = {:.2f},\nkurtosis = {:.2f},\nmean = {:.2f},\nStandard Deviation = {:.2f}'.format(skew, kurt, mean, std), bins=30).set(title='Average Words in Resumes')
plt.legend(loc='best')
plt.show()


# #### Number of stopwords

# In[24]:


from nltk.corpus import stopwords
stop = stopwords.words('english')

df['stopwords'] = df['Resumes'].apply(lambda x: len([x for x in x.split() if x in stop]))
df[['Resumes','stopwords']].head()


# In[51]:


skew = df['stopwords'].skew()
kurt = df['stopwords'].kurt()
mean = df['stopwords'].mean()
std = df['stopwords'].std()
plt.figure(figsize=(12,8))
sns.set(font_scale=1.2)
sns.distplot(df["stopwords"], kde= True, label='Skewness = {:.2f},\nkurtosis = {:.2f},\nmean = {:.2f},\nStandard Deviation = {:.2f}'.format(skew, kurt, mean, std), bins=30).set(title='Stopwords in Resumes')
plt.legend(loc='best')
plt.show()


# #### Number of numerics

# In[25]:


df['numerics'] = df['Resumes'].apply(lambda x: len([x for x in x.split() if x.isdigit()]))
df[['Resumes','numerics']].head()


# In[53]:


skew = df['numerics'].skew()
kurt = df['numerics'].kurt()
mean = df['numerics'].mean()
std = df['numerics'].std()
plt.figure(figsize=(12,8))
sns.set(font_scale=1.2)
sns.distplot(df["numerics"], kde= True, label='Skewness = {:.2f},\nkurtosis = {:.2f},\nmean = {:.2f},\nStandard Deviation = {:.2f}'.format(skew, kurt, mean, std), bins=30).set(title='Numerics in Resumes')
plt.legend(loc='best')
plt.show()


# #### Number of Uppercase words

# In[26]:


df['uppercase'] = df['Resumes'].apply(lambda x: len([x for x in x.split() if x.isupper()]))
df[['Resumes','uppercase']].head()


# In[55]:


skew = df['uppercase'].skew()
kurt = df['uppercase'].kurt()
mean = df['uppercase'].mean()
std = df['uppercase'].std()
plt.figure(figsize=(12,8))
sns.set(font_scale=1.2)
sns.distplot(df["uppercase"], kde= True, label='Skewness = {:.2f},\nkurtosis = {:.2f},\nmean = {:.2f},\nStandard Deviation = {:.2f}'.format(skew, kurt, mean, std), bins=30).set(title='Uppercase in Resumes')
plt.legend(loc='best')
plt.show()


# In[27]:


df.to_csv('Counts_extracted.csv', index=False)
df = pd.read_csv('Counts_extracted.csv')
df


# #### Percentage of Links attached in the Resumes

# In[28]:


(df.Resumes.str.contains('https://').value_counts() / len(df))*100


# In[58]:


links = (df.Resumes.str.contains('https://').value_counts() / len(df))*100
No_links = links[0]
Yes_links = links[1]


# In[59]:


plt.figure(figsize=(12,8)) 
# Setting size in Chart based on 
# given values

sizes = [No_links, Yes_links]
  
# Setting labels for items in Chart
labels = ['False', 'True']
  
# colors
colors = ['#007500', '#FF0000']
  
# explosion
explode = (0.0, 0.05)
  
# Pie Chart
plt.pie(sizes, colors=colors, labels=labels,
        autopct='%1.0f%%', shadow=True,
        pctdistance=0.85, 
        explode=explode,
        startangle=0,
        textprops = {'size':'x-large',
                   'fontweight':'bold',
                    'rotation':'0',
                   'color':'black'})
  
# draw circle
centre_circle = plt.Circle((0, 0), 0.70, fc='white')
fig = plt.gcf()
  
# Adding Circle in Pie chart
fig.gca().add_artist(centre_circle)

# Adding Title of chart
plt.title('Percentage of Hyperlinks Present in the Resumes', fontsize = 16, fontweight = 'bold')
  
# Add Legends
plt.legend(labels, loc="upper right")
  
# Displaying Chart
plt.show()


# ### Text Pre-Processing

# #### <a id='5Af'>Removing Punctuations, Numbers and Special characters</a>
# This step should not be followed if we also want to do sentiment analysis on __key phrases__ as well, because semantic meaning in a sentence needs to be present. So here we will create one additional column 'absolute_tidy_tweets' which will contain absolute tidy words which can be further used for sentiment analysis on __key words__.

# <a id='5Ah'>h. Removing Stop words</a>
# With the same reason we mentioned above, we won't perform this on 'tidy_tweets' column, because it needs to be used for __key_phrases__ sentiment analysis.
# 
# #### <a id='5Ai'> Tokenize *'Resumes'*</a>  
# 
# #### <a id='5Aj'> Converting words to Lemma</a>

# In[29]:


import re
import string
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer,PorterStemmer
from nltk.corpus import stopwords
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer() 
nltk.download('wordnet')
nltk.download('stopwords')


# In[30]:


def preprocess(sentence):
    sentence=str(sentence)
    sentence = sentence.lower()
    sentence=sentence.replace('{html}',"") 
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', sentence)
    rem_url=re.sub(r'http\S+', '',cleantext)
    rem_num = re.sub('[0-9]+', '', rem_url)
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(rem_num)  
    filtered_words = [w for w in tokens if len(w) > 2 if not w in stopwords.words('english')]
    stem_words=[stemmer.stem(w) for w in filtered_words]
    lemma_words=[lemmatizer.lemmatize(w) for w in stem_words]
    return " ".join(filtered_words)


# In[31]:


df = pd.read_csv('Resume2Text_Extracted.csv')
df['Clean_Resumes']=df.Resumes.apply(lambda x: preprocess(x))
df.to_csv('Cleaned_Resumes.csv', index = False)


# In[3]:


df = pd.read_csv('Cleaned_Resumes.csv')
df


# In[33]:


df.Resumes[20]


# In[34]:


df.Clean_Resumes[20]


# ### Data Exploration
# 
# Now its time to explore the preprocessed and cleaned text reviews. Textual data can be explored using Word Clouds. These are visual representations of the frequency of different words present in text. Importance of words are represented by size of the word. Bigger size represents more frequently occuring words.$

# ### N-grams
# N-grams are the combination of multiple words used together. Ngrams with N=1 are called unigrams. Similarly, bigrams (N=2), trigrams (N=3) and so on can also be used.
# 
# Unigrams do not usually contain as much information as compared to bigrams and trigrams. The basic principle behind n-grams is that they capture the language structure, like what letter or word is likely to follow the given one. The longer the n-gram (the higher the n), the more context you have to work with. Optimum length really depends on the application – if your n-grams are too short, you may fail to capture important differences. On the other hand, if they are too long, you may fail to capture the “general knowledge” and only stick to particular cases.
# 
# So, let’s quickly extract bigrams from our tweets using the ngrams function of the textblob library.

# In[37]:


TextBlob(df['Resumes'][1]).ngrams(1)[:20]


# In[38]:


TextBlob(df['Resumes'][1]).ngrams(2)[:20]


# In[39]:


TextBlob(df['Resumes'][1]).ngrams(3)[:20]


# In[65]:


df['Clean_Resumes']


# In[165]:


from sklearn.feature_extraction.text import CountVectorizer
c_vec = CountVectorizer(stop_words=stopwords.words('english'), ngram_range=(1,2))
# matrix of ngrams
ngrams = c_vec.fit_transform(df['Clean_Resumes'])
# count frequency of ngrams
count_values = ngrams.toarray().sum(axis=0)
# list of ngrams
vocab = c_vec.vocabulary_
df_ngram = pd.DataFrame(sorted([(count_values[i],k) for k,i in vocab.items()], reverse=True)
            ).rename(columns={0: 'frequency', 1:'unigram/bigram'})


# In[167]:


f, axe = plt.subplots(1,1, figsize=(18,6), dpi=200)
ax = sns.barplot(x=df_ngram['unigram/bigram'].head(20), y=df_ngram.frequency.head(20), data=data, ax = axe,
            label='Total Pofile Types = {}'.format(len(data.Profile.unique())))
axe.set_xlabel('Words', size=12,fontweight = 'bold')
axe.set_ylabel('Frequency', size=12,fontweight = 'bold')
plt.yticks(fontsize=14,fontweight = 'bold')
plt.xticks(fontsize=14,fontweight = 'bold', rotation = 45)
plt.legend(loc='best', fontsize  = 'x-large')
plt.title('Top 20 Most used Words in Resumes', fontsize = 16, fontweight = 'bold')
for i in ax.containers:
    ax.bar_label(i,color = 'black', fontweight = 'bold', fontsize= 14)
plt.show()


# ### Generate Word Cloud

# In[61]:


oneSetOfStopWords = set(stopwords.words('english')+['``',"''"])
totalWords =[]
Sentences = df['Clean_Resumes'].values
cleanedSentences = ""
for records in Sentences:
    cleanedText = preprocess(records)
    cleanedSentences += cleanedText
    requiredWords = nltk.word_tokenize(cleanedText)
    for word in requiredWords:
        if word not in oneSetOfStopWords and word not in string.punctuation:
            totalWords.append(word)
    
wordfreqdist = nltk.FreqDist(totalWords)
mostcommon = wordfreqdist.most_common(50)
print(mostcommon)


# In[62]:


from wordcloud import WordCloud
# Define a function to plot word cloud
def plot_cloud(wordcloud):
    plt.figure(figsize=(40,30))
    plt.imshow(wordcloud)
    plt.title('Most Common Words in Resumes', fontsize = 50, fontweight = 'bold')
    plt.axis('off')
    plt.show()
    
# Generate Word Cloud

#STOPWORDS.add('pron')
#STOPWORDS.add('rt')
#STOPWORDS.add('yeah')
wordcloud=WordCloud(width=3000,height=2000,background_color='black',max_words=200,
                   colormap='Set1').generate(cleanedSentences)

plot_cloud(wordcloud)


# ### <a id='9'> Story Generation and Visualization</a>

# #### <a id='9A'>A. Most common words in Different Profile Resumes</a>
# Answer can be best found using WordCloud

# In[21]:


def generate_wordcloud(all_words, Profile):
    wordcloud = WordCloud(width=800, height=500,background_color='black',max_words=200,
                   colormap='Set1').generate(all_words)

    plt.figure(figsize=(14, 10), dpi=200)
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.title('Most Common Words in {} Resumes'.format(Profile), fontsize = 16, fontweight = 'bold')
    plt.show()


# In[68]:


for profiles in df.Profiles.unique():
    all_words = ' '.join([text for text in df['Clean_Resumes'][df.Profiles == profiles]])
    generate_wordcloud(all_words, profiles)


# In[93]:


for profiles in df.Profiles.unique():
    all_words = ' '.join([text for text in df['Clean_Resumes'][df.Profiles == profiles]])
    generate_wordcloud(all_words, profiles)


# In[3]:


categories = np.sort(df['Profiles'].unique())
categories


# In[4]:


df_categories = [df[df['Profiles'] == category].loc[:, ['Clean_Resumes', 'Profiles']] for category in categories]


# In[6]:


def wordfreq(df):
    count = df['Clean_Resumes'].str.split(expand=True).stack().value_counts().reset_index()
    count.columns = ['Word', 'Frequency']

    return count.head(10)


# In[7]:


for i, category in enumerate(categories):
    print(category)


# In[8]:


fig = plt.figure(figsize=(16, 32), dpi=200)

for i, category in enumerate(categories):
    wf = wordfreq(df_categories[i])

    fig.add_subplot(2, 2, i + 1).set_title(category)
    plt.bar(wf['Word'], wf['Frequency'])
    plt.xticks(rotation=90)

plt.show()


# In[113]:


fig = plt.figure(figsize=(16, 32), dpi=200)

for i, category in enumerate(categories):
    wf = wordfreq(df_categories[i])

    fig.add_subplot(3, 3, i + 1).set_title(category)
    plt.bar(wf['Word'], wf['Frequency'])
    plt.xticks(rotation=90)

plt.show()


# In[ ]:


fig = plt.figure(figsize=(22, 64), dpi=200)

for i, category in enumerate(categories):
    wf = wordfreq(df_categories[i])

    fig.add_subplot(3, 3, i + 1).set_title(category)
    plt.bar(wf['Word'], wf['Frequency'])
    plt.xticks(rotation=45)

plt.show()


# ### <a id='6Da'> Named Entity Recognition (NER)</a>

# In[85]:


# Parts Of Speech (POS) Tagging
nlp=spacy.load('en_core_web_lg')

one_block=cleanedSentences[:2000]
doc_block=nlp(one_block)
spacy.displacy.render(doc_block,style='ent',jupyter=True)


# In[80]:


for token in doc_block[:50]:
    print(token,token.pos_)  


# #### Filtering out only the nouns and verbs from the Text to Tokens

# In[81]:


# Filtering the nouns and verbs only
one_block=cleanedSentences
doc_block=nlp(one_block)
nouns_verbs=[token.text for token in doc_block if token.pos_ in ('NOUN','VERB')]
print(nouns_verbs[100:200])


# #### Counting all the nouns and verbs present in the Tokens of words

# In[82]:


# Counting the noun & verb tokens
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()

X=cv.fit_transform(nouns_verbs)
sum_words=X.sum(axis=0)

words_freq=[(word,sum_words[0,idx]) for word,idx in cv.vocabulary_.items()]
words_freq=sorted(words_freq, key=lambda x: x[1], reverse=True)

wd_df=pd.DataFrame(words_freq)
wd_df.columns=['word','count']
wd_df[0:10] # viewing top ten results


# #### Visualizing the Result of Top 10 nouns and verbs most frequently present in the Resumes

# In[84]:


# Visualizing results (Barchart for top 10 nouns + verbs)
wd_df[0:10].plot.bar(x='word',figsize=(12,8),title='Top 10 nouns and verbs')
plt.show()


# ### <a id='6'> Basic Feature Extaction</a>

# #### <a id='6Aa'> **Applying bag of Words without N grams**</a>

# In[49]:


from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
tweetscv=cv.fit_transform(requiredWords)


# In[51]:


print(cv.vocabulary_)


# In[52]:


print(cv.get_feature_names()[109:200])


# In[53]:


print(cv.get_feature_names()[:100])


# In[65]:


print(tweetscv.toarray()[100:200])


# ### <a id='6Ba'>**CountVectorizer with N-grams (Bigrams & Trigrams)**</a>

# In[3]:


from nltk.corpus import stopwords
ps = PorterStemmer()
corpus = []
for i in tqdm_notebook(range(0, len(df))):
    review = re.sub('[^a-zA-Z]', ' ', df['Clean_Resumes'][i])
    review = review.lower()
    review = review.split()
    
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)


# In[28]:


df.Clean_Resumes[7]


# In[29]:


corpus[7]


# In[59]:


## Applying Countvectorizer
# Creating the Bag of Words model
cv = CountVectorizer(max_features=5000,ngram_range=(1,3))
X = cv.fit_transform(corpus).toarray()


# In[60]:


X.shape


# In[61]:


cv.get_feature_names()[:20]


# In[62]:


cv.get_params()


# In[63]:


count_df = pd.DataFrame(X, columns=cv.get_feature_names())
count_df


# ### <a id='6Ca'> **TF-IDF Vectorizer**</a>

# In[75]:


from nltk.corpus import stopwords
ps = PorterStemmer()
corpus = []
for i in tqdm_notebook(range(0, len(df))):
    review = re.sub('[^a-zA-Z]', ' ', df['Clean_Resumes'][i])
    review = review.lower()
    review = review.split()
    
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)


# In[4]:


corpus[1]


# In[76]:


## TFidf Vectorizer
tfidf_v=TfidfVectorizer(max_features=5000,ngram_range=(1,3))
X=tfidf_v.fit_transform(corpus).toarray()


# In[77]:


X.shape


# In[7]:


tfidf_v.get_feature_names()[:20]


# In[9]:


tfidf_v.get_feature_names()[4980:]


# In[10]:


tfidf_v.get_params()


# In[78]:


tfidf_df = pd.DataFrame(X, columns=tfidf_v.get_feature_names())
tfidf_df


# In[3]:


from nltk.corpus import stopwords
ps = PorterStemmer()
corpus = []
for i in tqdm_notebook(range(0, len(df))):
    review = re.sub('[^a-zA-Z]', ' ', df['Clean_Resumes'][i])
    review = review.lower()
    review = review.split()
    
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)

# requiredText = df['Clean_Resumes'].values

word_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    stop_words='english',
    max_features=5000,
    ngram_range=(1,3))
tfidf=word_vectorizer.fit(corpus)
WordFeatures = word_vectorizer.transform(corpus).toarray()

tfidf_df = pd.DataFrame(WordFeatures, columns=tfidf.get_feature_names())
tfidf_df


# In[45]:


tfidf_df.to_csv('tfidf.csv', index = False)


# In[46]:


import pickle

pickle.dump(tfidf, open("tfidf.pkl", "wb"))


# ## <a id='7'> Feature Extraction</a>
# 
# We need to convert textual representation in the form on numeric features. We have two popular techniques to perform feature extraction:
# 
# 1. __Bag of words (Simple vectorization)__
# 2. __TF-IDF (Term Frequency - Inverse Document Frequency)__
# 
# We will use extracted features from both one by one to perform sentiment analysis and will compare the result at last.
# 

# ### <a id='7Aa'>A. Feature Extraction for 'Key Words'</a>

# In[86]:


# BOW features
bow_word_vectorizer = CountVectorizer(max_df=0.90, min_df=2, stop_words='english')
# bag-of-words feature matrix
bow_word_feature = bow_word_vectorizer.fit_transform(df['Clean_Resumes'])

# TF-IDF features
tfidf_word_vectorizer = TfidfVectorizer(max_df=0.90, min_df=2, stop_words='english')
# TF-IDF feature matrix
tfidf_word_feature = tfidf_word_vectorizer.fit_transform(df['Clean_Resumes'])

