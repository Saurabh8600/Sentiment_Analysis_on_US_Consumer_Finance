#!/usr/bin/env python
# coding: utf-8

# # DATA DESCRIPTION

# This dataset is about consumer complaints.Customer service is an essential part of any organization since it help develop a customer base. Our main aim is to implement a sentiment analysis with the main customer service issues with some of the organizations. We want to determine what is the opinion "out there" of the main issues and "recommend" several solutions to improve this sentiment some customers have towards a specific final institution.

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import nltk

import warnings
warnings.filterwarnings("ignore")




# # LOADING DATA
# 

# In[2]:


dataset = pd.read_csv("consumer_complaints.csv")
dataset


# # EDA

# In[3]:


dataset.head()


# In[4]:


dataset.tail()


# In[5]:


dataset.info()


# In[6]:


dataset.shape


# In[7]:


dataset.nunique()


# In[8]:


dataset.isna().sum()


# In[9]:


dataset['consumer_disputed?'].value_counts().plot(kind='bar',title="Consumer Dispute count")
plt.show()


# In[10]:


dataset['company'].nunique()


# There are total 3605 Companies

# # Goal 1: Determine the top companies that received more disputes

# In[11]:


top_companies = dataset[dataset['consumer_disputed?']=="Yes"]['company'].value_counts().head(10)
print("Top Companies with Most Disputes:\n",top_companies)


# In[12]:


sns.barplot(x=top_companies,y=top_companies.index)
plt.title("Top Companies with Most Disputes",size=15)
plt.xlabel('Dispute count')
plt.show()


# ## Bank of America Company has the Maximum No. of Complaints

# # Goal 2: Plot the trend of disputes over time
# 

# In[13]:


dataset['date_received'] = pd.to_datetime(dataset['date_received'])
disputes_over_time = dataset.groupby(dataset['date_received'].dt.to_period('M')).size()
disputes_over_time.plot(kind='line', marker='o')
plt.xlabel('Date')
plt.ylabel('Number of Disputes')
plt.title('Trend of Disputes Over Time')
plt.xticks(rotation=45)
plt.show()


# ## - Dispute trend is increasing over the period of time

# # Goal 3: Sentiment Analysis

# In[14]:


dataset.isna().sum()


# In[15]:


dataset.dropna(axis=0,inplace=True)


# In[16]:


dataset


# In[17]:


dataset.isna().sum()


# In[18]:


dataset.info()


# In[19]:


dataset['consumer_disputed?'].value_counts()


# In[20]:


from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')
from tqdm.notebook import tqdm
from collections import defaultdict


# In[21]:


# Initialize the VADER sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Create a defaultdict to store sentiment scores for each company
company_sentiments = defaultdict(list)

# Calculate sentiment scores for each complaint narrative
for index, row in tqdm(dataset.iterrows(), total=len(dataset)):
    complaint = row['consumer_complaint_narrative']
    comp_id= row['complaint_id']
    sentiment_scores = sia.polarity_scores(complaint)
    
    # Store the sentiment score in the defaultdict
    company_sentiments[comp_id].append(sentiment_scores['compound'])


# In[22]:


df=pd.DataFrame(company_sentiments).T.reset_index().rename(columns={'index': 'complaint_id',0:'Sentiment_score'})
df


# In[23]:


df= df.merge(dataset, how='left')
df


# In[24]:


def Tagging(Sentiment_score):
    if Sentiment_score<0:
        return "Negative"
    else:
        return "Positive"
    
df['Sentiment']=df['Sentiment_score'].apply(Tagging)


# # TESTING SENTIMENTS WITH RANDOM ROWS

# In[25]:


a = df['consumer_complaint_narrative'].values[1]
a


# In[26]:


sia.polarity_scores("I received services from a healthcare provider XXXX years ago. Recently I got a call from a Collection Agency who said the debt has not been paid. The healthcare provider and I were in agreement that they 'd take care of billing but somehow my name was thrown into the mix. The healthcare agency does not respond to my inquiries about the debt or to the collection agency 's inquiries about the debt. This has a possibility of ruining my credit. \n")


# In[27]:


b = df['consumer_complaint_narrative'].values[3]
b


# In[28]:


sia.polarity_scores('Reached out to them however, never got a response. Unfortunately, I was the victim of identity theft. I have done everything required of me by law to dispute the fraudulent items on my credit report. I sent a letter to " Flagship Credit \'\' on XXXX via XXXX certified mail. The company refuses to honor my rights under the FDCPA & the FCRA as a victim of identity theft to remove the fraudulent inquiry that was submitted without my knowledge, consent and/or authorization. \n\nPlease find attached the notice that was sent out along with the attached certified receipt. Information can be tracked online using referenced information. Thank you for you time and attention regarding the matter. \n')


# In[32]:


c = df['consumer_complaint_narrative'].values[13]
c


# In[33]:


sia.polarity_scores('I chose # XXXX because I did n\'t see an appropriate point under " False statements or representation. \'\' NATIONAL CREDIT SYSTEMS , INC \'\' told me I had to pay the debt by giving them my CHECKING ACCOUNT NUMBER in order to " Protect your credit. \'\' And when I said I was n\'t giving them my acct number she said " Well, like I told you, giving me your checking account number is THE only way to protect your credit. \'\' I called within XXXX days and said I wanted a letter statin that if I XXXX they would contact the credit bureaus to say the debt was XXXX, and that I was going to send an official money order for the XXXX payment. AGAIN she said if I did that they could not ensure my credit. \n')


# In[29]:


df


# In[30]:


df['Sentiment'].value_counts()


# So the overall sentiments of the issues is Negative

# In[31]:


df['Sentiment'].value_counts().plot(kind = 'pie', autopct = '%0.1f%%')

