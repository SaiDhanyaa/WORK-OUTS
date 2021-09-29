#!/usr/bin/env python
# coding: utf-8

# # Import Libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.frequent_patterns import apriori,association_rules
from mlxtend.preprocessing import TransactionEncoder


# # Load Data

# In[2]:


groceries=[]
#as data is transaction data we will be reading it directly
with open('groceries.csv') as f:
    groceries=f.read()


# In[3]:


groceries=groceries.split('\n')


# In[4]:


groceries_list=[]
for i in groceries:
    groceries_list.append(i.split(','))
    


# In[5]:


all_groceries_list=[]
all_groceries_list=[i for item in groceries_list for  i in item]
all_groceries_list


# In[6]:


from collections import Counter
item_frequencies=Counter(all_groceries_list)
item_frequencies=sorted(item_frequencies.items(),key=lambda x:x[1])


# In[7]:


frequencies = list(reversed([i[1] for i in item_frequencies]))
items = list(reversed([i[0] for i in item_frequencies]))
items


# In[8]:


plt.bar(height = frequencies[0:11],
        x = list(range(0,11)),color='rgbkymc');
plt.xticks(list(range(0,11),),items[0:11]);
plt.xlabel("items")
plt.ylabel("Count")


# # Creating DataFrame

# In[9]:


groceries_series=pd.DataFrame(pd.Series(groceries_list))
groceries_series


# In[10]:


groceries_series=groceries_series.iloc[:9835,:]
groceries_series


# In[11]:


groceries_series.columns=['transactions']
groceries_series


# In[12]:


x=groceries_series['transactions'].str.join(sep='*').str.get_dummies(sep='*')
x


# In[13]:


x1=x.dropna().reset_index(drop=True)
x1


# In[14]:


df=pd.get_dummies(x1)
df


# In[15]:


df.describe()


# # Apriori Algorithm

# In[16]:


frequent_itemsets = apriori(df, min_support=0.02, use_colnames=True)
frequent_itemsets


# In[17]:


frequent_itemsets.sort_values('support',ascending=False,inplace=True)
plt.bar(x = list(range(1,11)),height = frequent_itemsets.support[1:11],color='rgmyk');plt.xticks(list(range(1,11)),frequent_itemsets.itemsets[1:11])
plt.xlabel('item-sets');plt.ylabel('support')


# In[18]:


rules=association_rules(frequent_itemsets,metric='lift',min_threshold=1)
rules.shape


# In[19]:


rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.2)
rules


# In[20]:


len(rules)


# In[21]:


rules.sort_values('lift',ascending = False)


# In[22]:


rules[rules.lift>1]


# In[23]:


rules[rules.lift<1]


# # Visualization

# In[24]:


x.info()


# In[25]:


sns.barplot(x['bottled water'], x['yogurt'])


# In[ ]:




