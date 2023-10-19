#!/usr/bin/env python
# coding: utf-8

# # DATA DESCRIPTION
This dataset is about consumer complaints.Customer service is an essential part of any organization since it help develop a customer base. Our main aim is to implement a sentiment analysis with the main customer service issues with some of the organizations. We want to determine what is the opinion "out there" of the main issues and "recommend" several solutions to improve this sentiment some customers have towards a specific final institution.
# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import nltk


# # LOADING DATA

# In[2]:


dataset = pd.read_csv("consumer_complaints.csv")


# In[3]:


dataset


# # EXPLORING THE DATA

# In[4]:


dataset.shape


# In[5]:


dataset.nunique()


# In[6]:


dataset.head()


# In[7]:


dataset.state.value_counts()


# In[8]:


dataset.tail()


# In[9]:


dataset.columns


# In[10]:


dataset.info()


# In[11]:


dataset.isna().sum()


# In[12]:


dataset.consumer_complaint_narrative.unique()


# In[13]:


print(dataset.tags.unique())
print(dataset.consumer_consent_provided.unique())


# In[14]:


#Removing the null values

dataset['sub_product'] = dataset['sub_product'].fillna('Others')
dataset['sub_issue'] = dataset['sub_issue'].fillna('Others')
dataset['company_public_response'] = dataset['company_public_response'].fillna('Others')
dataset['tags'] = dataset['tags'].fillna('Others')


# In[15]:


dataset.isna().sum() 


# In[16]:


#dataset = pd.DataFrame(dataset)


# In[17]:


#drop null values from state and zipcode
column_name_to_drop_nulls_from = 'state'
dataset = dataset.dropna(subset=[column_name_to_drop_nulls_from])

column_name_to_drop_nulls_from = 'zipcode'
dataset = dataset.dropna(subset=[column_name_to_drop_nulls_from])


# In[18]:


dataset.isna().sum()


# In[19]:


with pd.ExcelWriter("cleaned_data2.xlsx") as writer:
    dataset.to_excel(writer)


# In[20]:


dataset.shape


# In[21]:


dataset['consumer_disputed?'].unique()


# In[22]:


# Changing the datatype of date_received and date_sent_to_company
dataset.date_received = pd.to_datetime(dataset.date_received)
dataset.date_sent_to_company = pd.to_datetime(dataset.date_sent_to_company)


# In[23]:


unique_years = dataset['date_received'].dt.year.unique()
print(unique_years)


# In[24]:


# Minimum & Maximun date
print('Minimum date :',dataset.date_received.min(),'')
print('Maximum date :',dataset.date_received.max())


# # EXPLORATORY DATA ANALYSIS

# In[25]:


#data not used for 2011 as it has only 1 month data and 2016 has only 4 month data

df2012 = dataset[dataset.date_received.dt.year == 2012]
df2013 = dataset[dataset.date_received.dt.year == 2013]
df2014 = dataset[dataset.date_received.dt.year == 2014]
df2015 = dataset[dataset.date_received.dt.year == 2015]


# In[26]:


# Monthwise complaints for the Year 2012
Complaints_2012 = df2012.groupby(df2012.date_received.dt.month).agg({'complaint_id':'count'})
Complaints_2012['month'] = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
Complaints_2012.set_index('month',drop=True,inplace=True)
Complaints_2012.rename(columns={'complaint_id':'no_of_complaints'},inplace=True)

print('Complaints for the Year 2012 :','\n',Complaints_2012)

# Monthwise complaints for the Year 2013
Complaints_2013 = df2013.groupby(df2013.date_received.dt.month).agg({'complaint_id':'count'})
Complaints_2013['month'] = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
Complaints_2013.set_index('month',drop=True,inplace=True)
Complaints_2013.rename(columns={'complaint_id':'no_of_complaints'},inplace=True)

print('\nComplaints for the Year 2013 :','\n',Complaints_2013) 

# Monthwise complaints for the Year 2014
Complaints_2014 = df2014.groupby(df2014.date_received.dt.month).agg({'complaint_id':'count'})
Complaints_2014['month'] = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
Complaints_2014.set_index('month',drop=True,inplace=True)
Complaints_2014.rename(columns={'complaint_id':'no_of_complaints'},inplace=True)

print('\nComplaints for the Year 2014 :','\n',Complaints_2014)
 
# Monthwise complaints for the Year 2015
Complaints_2015 = df2015.groupby(df2015.date_received.dt.month).agg({'complaint_id':'count'})
Complaints_2015['month'] = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
Complaints_2015.set_index('month',drop=True,inplace=True)
Complaints_2015.rename(columns={'complaint_id':'no_of_complaints'},inplace=True)

print('\nComplaints for the Year 2015 :','\n',Complaints_2015)


# In[27]:


# Monthwise complaints trend for the year 2012,2013,2014 & 2015

plt.figure(figsize=(20,10))
plt.plot(Complaints_2012,marker='o',label='2012')
plt.plot(Complaints_2013,marker='o',label='2013')
plt.plot(Complaints_2014,marker='o',label='2014')
plt.plot(Complaints_2015,marker='o',label='2015')
plt.grid()
plt.xlabel('Month', size= 15)
plt.ylabel('No of Complaints',size=15)
plt.legend(loc='upper right')
plt.title('Monthwise complaints trend for the year 2012,2013,2014 & 2015\n', size=15)


# In[28]:


#Yearwise Trend

plt.figure(figsize=(5,5))
plt.plot(dataset.groupby(dataset['date_received'].dt.year).agg({'complaint_id':'count'}),marker='o')
plt.grid()
plt.xlabel('Year')
plt.ylabel('No of Complaints')
plt.title('Yearwise complaints trends\n')


# In[29]:


plt.figure(figsize=(25,10))
plt.bar(dataset['product'].unique(),dataset['product'].value_counts())
plt.ylabel('No of Complaints', size=15)
plt.xlabel('Products',size=15)
plt.title('Product-wise complaints count\n', size=15)


# In[30]:


plt.figure(figsize=(10,5)) 
sns.countplot(x="product", hue="consumer_disputed?", data=dataset)
plt.xlabel('Products', size = 10)
plt.ylabel('Count', size = 10)
plt.title('Product vs Consumer Disputed ', size = 10)
plt.xticks(rotation=90)
plt.show()


# In[31]:


plt.figure(figsize=(5,3))
sns.countplot(x="timely_response",data=dataset)
plt.title("Count Plot of Timely Response\n")
plt.show


# In[32]:


plt.figure(figsize=(5,4))
sns.countplot(x="consumer_disputed?",data=dataset)
plt.title("Count Plot of Consumer Disputed\n")
plt.show


# In[33]:


# Group and aggregate the data
df = dataset.groupby(['company', 'timely_response']).size().reset_index(name='No_of_Complaints')

# Separate timely and late responses
timely_response = df[df['timely_response'] == 'Yes']
late_response = df[df['timely_response'] == 'No']


# In[34]:


timely_response = timely_response.drop('timely_response',axis=1)
late_response = late_response.drop('timely_response',axis=1)

timely_response.rename(columns = {'No_of_Complaints':'timely_response'},inplace=True)
late_response.rename(columns = {'No_of_Complaints':'late_response'},inplace=True)

timely_response.set_index('company',inplace=True)
late_response.set_index('company',inplace=True)

timely_or_late_response = pd.concat([timely_response,late_response],axis=1)
timely_or_late_response['total_complaints'] = timely_response.timely_response+late_response.late_response


# In[35]:


timely_or_late_response.sort_values('total_complaints',ascending=False,inplace=True)
timely_or_late_response.head(10)


# In[36]:


# Stacked Bar Chart
# Select the top 10 companies
top_companies = timely_or_late_response.head(10)

plt.figure(figsize=(15, 9))

# Create the bar plot using Seaborn
sns.barplot(data=top_companies, x='timely_response', y=top_companies.index, color='blue', label='Timely Response')
sns.barplot(data=top_companies, x='late_response', y=top_companies.index, color='orange', label='Late Response')
plt.xlabel('No of Complaints')
plt.ylabel('Company')
plt.title('Top 10 Companies with Most Complaints - Timely vs Late Responses')
plt.legend()
plt.show()


# In[37]:


percentage_distribution = (dataset['company_response_to_consumer'].value_counts() / len(df)) * 100

plt.figure(figsize=(6, 5))
sns.barplot(y=percentage_distribution.index, x=percentage_distribution.values, color='blue')
plt.xlabel('Percentage of Complaints')
plt.ylabel('Response provided to the consumers')
plt.title('Response provided for the complaints\n')
plt.show()


# In[38]:


#To see Average No of days for complaint to reach the company through different modes of submission

df3 = dataset[['submitted_via', 'date_received', 'date_sent_to_company']].copy()

df3['no_of_days'] = (df3['date_sent_to_company'] - df3['date_received']).dt.days

print(df3)


# In[39]:


df3.drop(['date_received','date_sent_to_company'],axis=1,inplace=True)


# In[40]:


df3 = df3.groupby('submitted_via').agg({'no_of_days':'mean'})
df3.no_of_days = round(df3.no_of_days,0)
df3.sort_values('no_of_days',ascending=True,inplace=True)
df3


# In[41]:


plt.figure(figsize=(8, 8))
sns.barplot(data=df3, x=df3.index, y='no_of_days', color='blue')
plt.xlabel('Mode of submitting the complaint')
plt.ylabel('Avg no. of days for the complaint to reach the company')
plt.title('Avg no. of days taken for the complaint to reach the company depending upon the mode of submission\n')
plt.show()


# Through the above data we can see that 
# 
# 1] The shortest mode of submission for the complaint to reach to the company is web.
# 
# 2] Maximum Number of complaints were been closed with an explanation by the company.
# 
# 3] Bank of America has the Maximum No. of Complaints.

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[44]:


X = dataset.iloc[:,[12,3,4,5,6,12,13]].values
y = dataset.iloc[:,-2].values


# In[45]:


X


# In[46]:


y


# In[47]:


# train test split

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2, random_state=0)


# In[48]:


#creating a linear regressor object

from sklearn.linear_model import LinearRegression
regressor= LinearRegression()
regressor.fit(X_train,y_train)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




