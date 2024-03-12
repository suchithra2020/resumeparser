#!/usr/bin/env python
# coding: utf-8

# # Resume Classification
# ## Business objective:
# ### The document classification solution should significantly reduce the manual human effort in the HRM. It should achieve a higher level of accuracy and automation with minimal human intervention

# In[1]:


import numpy as np 
import pandas as pd 
import re
import time
import string
import warnings
from tqdm.notebook import tqdm_notebook
import os,re
import textract as tr
import docx2txt

# ignoring all the warnings
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings("ignore", category=DeprecationWarning)


# ### Extracting data from all the resumes

# In[2]:


path_input = r"C:/Users\Moin Dalvi\Data_Science\Projects\Resume_Classification/Resumes_docx/"
for folder in os.listdir(path_input):
    folder_path = os.path.join(path_input, folder)
    for file in os.listdir(folder_path):
        if file.endswith('.docx'):
            final_path = os.path.join(folder_path, file)
            print(final_path)


# In[4]:


file_path = r"C:/Users\Moin Dalvi\Data_Science\Projects\Resume_Classification/Resumes_docx"
extracted_data = []
profile_names = []

for folder in os.listdir(file_path):
    folder_path = os.path.join(file_path, folder)
    for file in os.listdir(folder_path):
        if file.endswith('.docx'):
            final_path = os.path.join(folder_path, file)
            extracted_data.append((docx2txt.process(final_path)))
            profile_names.append(folder)


# In[5]:


final_path


# In[6]:


len(extracted_data)


# In[7]:


extracted_data[:1]


# In[8]:


type(extracted_data)


# In[9]:


data = pd.DataFrame()
data['Resumes'] = extracted_data
data['Profiles'] = profile_names
data


# In[10]:


data.Profiles.unique()


# In[13]:


data.to_csv('Resume2Text_Extracted.csv', index = False )


# In[12]:


data.Resumes[0]


# In[ ]:





# In[ ]:




