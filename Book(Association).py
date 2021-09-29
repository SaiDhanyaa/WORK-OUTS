#!/usr/bin/env python
# coding: utf-8

# # Import Libraries

# In[1]:


#conda install -c conda -forge mixtend
get_ipython().run_line_magic('pip', 'install mlxtend --upgrade')


# In[2]:


conda install -c conda-forge mlxtend


# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.frequent_patterns import apriori,association_rules
from mlxtend.preprocessing import TransactionEncoder


# # Load Data

# In[5]:


book=pd.read_csv('book(1).csv')
book.head()


# In[6]:


book.info()


# In[7]:


te=TransactionEncoder()


# In[8]:


te_ary=te.fit(book).transform(book)
te_ary


# # Visualization
# 

# In[9]:


sns.distplot(book['ChildBks'])


# In[10]:


from mlxtend.frequent_patterns import apriori,association_rules
frequent_itemsets=apriori(book,min_support=0.05,use_colnames=True,max_len=3)
plt.bar(x = list(range(1,11)),height = frequent_itemsets.support[1:11],color='rgmyk');
plt.xticks(list(range(1,11)),frequent_itemsets.itemsets[1:11])
plt.xlabel('item-sets');plt.ylabel('support')
rules=association_rules(frequent_itemsets,metric='lift',min_threshold=1)


# In[11]:


sns.pairplot(book)


# In[12]:


sns.barplot(book['CookBks'], book['ChildBks'])


# In[13]:


sns.boxplot(book['ChildBks'], book['CookBks'], hue=book['Florence'])


# In[14]:


sns.lmplot(x='ChildBks', y='CookBks', data=book)


# In[15]:


sns.jointplot(book['ChildBks'],book['CookBks'], kind="kde")


# ## Preprocessing

# In[17]:


df=pd.get_dummies(book)
df


# In[18]:


df.describe()


# # Apriori Algorithm

# In[20]:


frequent_itemsets = apriori(df, min_support=0.2, use_colnames=True)
frequent_itemsets


# In[22]:


rules = association_rules(frequent_itemsets, metric="lift", min_threshold=0.5)
rules


# In[23]:


len(rules)


# In[25]:


rules1 = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)
rules1


# In[26]:


rules.sort_values('lift',ascending = False)


# In[27]:


rules[rules.lift>1]


# In[ ]:




