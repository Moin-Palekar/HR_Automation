#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import random
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[10]:


#gathering preferences
df = pd.read_csv("preferences.csv")

#features
cgpa = 0
no_of_internships = 0
no_of_projects = 0 #personal projects
hired = 0 # 0 is yes, 1 is no
record = {} #the new record to be added in each iteration
record_list = [] # the list of records to be shown to the user to gauge preferences

#gathering preferences
for i in range(15):
    cgpa = random.randrange(6,10)
    no_of_internships = random.randrange(0,5)
    no_of_projects = random.randrange(0,30)
    print("would you hire someone with the following profile:")
    print("cgpa {}".format(cgpa))
    print("number of internships {}".format(no_of_internships))
    print("number of projects {}".format(no_of_projects))
    hired = int(input("hired or not?: "))
    
    record = {"cgpa":cgpa, "no_of_internships":no_of_internships, "no_of_projects":no_of_projects, "hired":hired}
    record_list.append(record)

#loading preferences onto dataframe to be used for training
for i in range(15):
    df = df.append(record_list[i], ignore_index = True)
    
    


# In[11]:


df


# In[12]:


plt.scatter(df.cgpa,df.no_of_internships, no_of_projects, marker="+", color ='red')


# In[13]:


from sklearn.model_selection import train_test_split


# In[16]:


x_train, x_test, y_train, y_test = train_test_split(df[["cgpa","no_of_internships","no_of_projects"]],df.hired, train_size = 0.9)


# In[17]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(x_train, y_train)


# In[18]:


model.predict(x_test)


# In[19]:


x_test


# In[20]:


model.score(x_test,y_test)


# In[23]:


test_data = [[8, 0, 10],
             [0, 1, 5],
             [9, 0, 30]]
model.predict(test_data)


# In[ ]:




