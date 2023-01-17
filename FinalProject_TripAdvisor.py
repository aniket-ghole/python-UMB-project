#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Course: MSIS 615 Business Programming
#Professor: Wei Zhang
#ProjectTopic: TripAdvisor Airline Review Analysis 
#Group members: Aniket Ghole, Rashmi Bajwan, Utsavi Waingankar

get_ipython().system('pip install selenium')
get_ipython().system('pip install webdriver-manager')


# In[2]:


import re
import time
import sqlite3
from selenium import webdriver
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager #downloading chrome webdriver

#Step 1: Get the webpage (using webdrive)
driver = webdriver.Chrome(ChromeDriverManager().install()) 
URLpattern_str="https://www.tripadvisor.com/Airlines"

page_URL=URLpattern_str.replace("$NUM$",str(1))
print("Starting with: "+page_URL)
driver.get(page_URL)
pageContent = driver.page_source
print(pageContent)

#It takes time to launch the browser


# In[3]:


from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
import time

#Parsing the first page to calculate the maximal number of review pages
main = driver.find_elements(By.XPATH, '//div[@class="mainColumnContent"]')
page_numbers = driver.find_elements(By.XPATH, "(//div[@class='ui_columns']//div[@class='airlineList ui_column is-12-tablet is-9-desktop is-12-mobile']//div[@id='AIRLINE_INDEX']//div[@class='mainColumnContent']//div[@class='pagination']//div[@class='prw_rup prw_common_standard_pagination_resp']//div[@class='unified ui_pagination standard_pagination']//div[@class='pageNumbers']/span)") 

#finding page numbers
i_page_numbers = int(page_numbers[len(page_numbers)-2].text)
airlinenames = []
reviews1 = []
ratings1 = []
count = 1
while count <= i_page_numbers:
    print(str(count) + ": airline detail:")
    airline_names = driver.find_elements(By.XPATH, "(//div[@class='ui_columns']//div[@class='airlineList ui_column is-12-tablet is-9-desktop is-12-mobile']//div[@id='AIRLINE_INDEX']//div[@class='mainColumnContent']//div[@class='prw_rup prw_airlines_airline_lander_card']//div[@class='airlineData']//div[@class='cell left']//div[@class='airlineSummary']/a)") 
    for airline_name in range(len(airline_names)):
        airlinenames.append(airline_names[airline_name].text)

    reviews = driver.find_elements(By.XPATH, "(//div[@class='ui_columns']//div[@class='airlineList ui_column is-12-tablet is-9-desktop is-12-mobile']//div[@id='AIRLINE_INDEX']//div[@class='mainColumnContent']//div[@class='prw_rup prw_airlines_airline_lander_card']//div[@class='airlineData']//div[@class='cell left']//div[@class='wrapper']//div[@class='reviews']/p)") 
    for review in range(len(reviews)):
        reviews1.append(reviews[review].text)

    ratings = driver.find_elements(By.XPATH, "(//div[@class='ui_columns']//div[@class='airlineList ui_column is-12-tablet is-9-desktop is-12-mobile']//div[@id='AIRLINE_INDEX']//div[@class='mainColumnContent']//div[@class='prw_rup prw_airlines_airline_lander_card']//div[@class='airlineData']//div[@class='cell left']//div[@class='airlineSummary']//a[@class='detailsLink']//div[@class='prw_rup prw_common_bubble_rating']/span)")
    for rating in range(len(ratings)):
        ratings1.append(ratings[rating].get_attribute("alt"))
    
    if(count<i_page_numbers):
        nextpage = driver.find_element(By.XPATH, "(//div[@class='ui_columns']//span[@data-page-number="+ str(count+1) + "])")
        nextpage.click()
        driver.wait = WebDriverWait(driver, 10)
        time.sleep(10)

    page_numbers = driver.find_elements(By.XPATH, "(//div[@class='ui_columns']//div[@class='airlineList ui_column is-12-tablet is-9-desktop is-12-mobile']//div[@id='AIRLINE_INDEX']//div[@class='mainColumnContent']//div[@class='pagination']//div[@class='prw_rup prw_common_standard_pagination_resp']//div[@class='unified ui_pagination standard_pagination']//div[@class='pageNumbers']/span)") 
    if page_numbers[len(page_numbers)-1].text == "…":
        i_page_numbers = int(page_numbers[len(page_numbers)-2].text)
    else:
        i_page_numbers = int(page_numbers[len(page_numbers)-1].text)
    print(str(i_page_numbers) + ": total pages:")
    count = count + 1

#Need to wait until 63pages gets readed


# In[4]:


print(len(airlinenames)) 
print(len(reviews1)) 
print(len(ratings1))


# In[5]:


#prepare the database for storing results
conn = sqlite3.connect('reviews.db', timeout=10)
c = conn.cursor()
c.execute("DROP TABLE IF EXISTS Reviews")
c = conn.cursor()
c.execute("CREATE TABLE Reviews(              AirlineName varchar(100),               NumberOfReviews varchar(20),               reviewDate date,               traveller_rating varchar(20),               reviewContent text)")
count1=0
count2=0
for count in range(0,len(airlinenames),2):
    AirlineName = ""
    NumberOfReviews = ""
    reviewContent = ""
    reviewDate = ""
    traveller_rating = ""
    AirlineName = airlinenames[count]
    NumberOfReviews = airlinenames[count+1].replace(',', '')
    NumberOfReviews = NumberOfReviews.replace(' reviews', '')
    NumberOfReviews = NumberOfReviews.replace(' review', '')
    if NumberOfReviews == "":
        NumberOfReviews = "0"
    traveller_rating = ratings1[count2]
    count2 = count2 + 1
    if int(NumberOfReviews.split(" ")[0])==0 or (count1 >= len(reviews1)) :
        #Saving the extracted data into the database
        query = "INSERT INTO Reviews(AirlineName,NumberOfReviews,traveller_rating,reviewContent) VALUES (?, ?, ?, ?)"
        c.execute(query, (AirlineName, NumberOfReviews, traveller_rating, reviewContent))
    elif int(NumberOfReviews.split(" ")[0])==1 :
        reviewDate = reviews1[count1][-10:]
        reviewContent = reviews1[count1][:-10]
        query = "INSERT INTO Reviews(AirlineName,NumberOfReviews,reviewDate,traveller_rating,reviewContent) VALUES (?, ?, ?, ?, ?)"
        c.execute(query, (AirlineName, NumberOfReviews, reviewDate, traveller_rating, reviewContent))
        count1 = count1 + 1
    else:
        reviewDate = reviews1[count1][-10:]
        reviewContent = reviews1[count1][:-10]
        query = "INSERT INTO Reviews(AirlineName,NumberOfReviews,reviewDate,traveller_rating,reviewContent) VALUES (?, ?, ?, ?, ?)"
        c.execute(query, (AirlineName, NumberOfReviews, reviewDate, traveller_rating, reviewContent))
        count1 = count1 + 1
        if (count1 < len(reviews1)) :
            reviewDate = reviews1[count1][-10:]
            reviewContent = reviews1[count1][:-10]
            query = "INSERT INTO Reviews(AirlineName,NumberOfReviews,reviewDate,traveller_rating,reviewContent) VALUES (?, ?, ?, ?, ?)"
            c.execute(query, (AirlineName, NumberOfReviews, reviewDate, traveller_rating, reviewContent))
            count1 = count1 + 1

c.close()
conn.commit()

try:
    driver.close()
    print("\n\nCollection Finished!")  
except:
    print("\n\nCollection Finished!")


# In[6]:


import pandas as pd

c = conn.cursor()
query = c.execute("Select * FROM Reviews")
cols = [column[0] for column in query.description]
data = pd.DataFrame.from_records(data = query.fetchall(), columns = cols)
c.close

#Saving the dataframe
data.to_csv('reviews.csv')


# In[7]:


#Reading training data from csv
data=pd.read_csv('reviews.csv', sep=',',header=0)

data.head()


# In[8]:


#find the unique elements of an array
data["traveller_rating"].unique()


# In[9]:


# extracting rating from text
data["traveller_rating"] = data["traveller_rating"].astype(str).str.replace(' of 5 bubbles', '')

data['AirlineName'] = data['AirlineName'].astype('|S')
data['NumberOfReviews'] = data['NumberOfReviews'].astype('int')
data['traveller_rating'] = data['traveller_rating'].astype('float')
data['reviewContent'] = data['reviewContent'].astype('str')


# In[10]:


#return description of the data in the DataFrame
data.describe()


# In[11]:


#print information about the DataFrame
data.info()


# In[12]:


#return object containing counts of unique values
data['reviewContent'].value_counts()


# In[13]:


#returns a DataFrame object where all the values are replaced with a Boolean value True for NULL values,
#otherwise False
data.isnull()


# In[14]:


data.isnull().sum()


# In[15]:


import numpy as np
from matplotlib import pyplot
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm


# In[16]:


#plotting a BoxPlot
print("Number of words: ")
print(len(np.unique(np.hstack(data['reviewContent']))))
print("Review length: ")
result = [len(x) for x in data['reviewContent']]
print("Mean %.2f words (%f)" % (np.mean(result), np.std(result)))

plt.figure(figsize = (10,6))

#plot review length
pyplot.boxplot(result)
pyplot.show()


# In[17]:


#Plotting NumberOfReviews against Count in the resulted dataframe
x=data[['Unnamed: 0']].to_numpy()
y=data[['NumberOfReviews']].to_numpy()

plt.figure(figsize = (12,6))
ax=plt.axes()
ax.set_xlabel('Count')
ax.set_ylabel('NumberOfReviews')
ax.scatter(x, y)
plt.show()


# In[18]:


#Running an OLS linear regression of NumberOfReviews(the Y) on Count(the X)
#Finding out the coefficient for count and the R‐squared(R2) value

X = data[['Unnamed: 0']].to_numpy()   
Y = data[['NumberOfReviews']].to_numpy()

X = sm.add_constant(X)
model = sm.OLS(Y,X)
results = model.fit()
print(results.summary())

#The co-efficient is 3192.1879 and the R-square is 0.00


# In[19]:


#Linear regression model

mod = LinearRegression() #Initialize the model
mod.fit(x, y) # fit the model

y_est = mod.predict(x) #now predict

#Plotting the estimated line along with scattered raw data on figure
plt.figure(figsize =(11,10))
ax=plt.axes()
ax.set_xlabel('Count')
ax.set_ylabel('Number of Reviews')
ax.scatter(x, y)
ax.plot(x, y_est)


# In[20]:


get_ipython().system('pip install keras')
get_ipython().system('pip install tensorflow')

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from keras.preprocessing.text import Tokenizer
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.tokenize import word_tokenize,sent_tokenize
from bs4 import BeautifulSoup
nltk.download('punkt')


# In[21]:


tokenizer=ToktokTokenizer()
stopword_list=nltk.corpus.stopwords.words('english')


# In[22]:


def strip_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

def remove_between_square_brackets(text):
    return re.sub('\[[^]]*\]', '', text)

def denoise_text(text):
    text = strip_html(text)
    text = remove_between_square_brackets(text)
    return text

#Apply function on review column
data['reviewContent']=data['reviewContent'].apply(denoise_text)


# In[23]:


def remove_special_characters(text, remove_digits=True):
    pattern=r'[^a-zA-z0-9\s]'
    text=re.sub(pattern,'',text)
    return text

#Apply function on review column
data['reviewContent']=data['reviewContent'].apply(remove_special_characters)


# In[24]:


#Stemming the text
def simple_stemmer(text):
    ps=nltk.porter.PorterStemmer()
    text= ' '.join([ps.stem(word) for word in text.split()])
    return text

#Apply function on review column
data['reviewContent']=data['reviewContent'].apply(simple_stemmer)


# In[25]:


#set stopwords to english
stop=set(stopwords.words('english'))
print(stop)

#removing the stopwords
def remove_stopwords(text, is_lower_case=False):
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    if is_lower_case:
        filtered_tokens = [token for token in tokens if token not in stopword_list]
    else:
        filtered_tokens = [token for token in tokens if token.lower() not in stopword_list]
    filtered_text = ' '.join(filtered_tokens)    
    return filtered_text

#Apply function on review column
data['reviewContent']=data['reviewContent'].apply(remove_stopwords)


# In[26]:


# Replacing ratings of -1,1,1.5,2,2.5,3,3.5 with 0 (bad) and 4,5 with 1 (good)
def sentiment_rating(rating):
    if(int(rating) == -1 or int(rating) == 1 or int(rating) == 1.5 or int(rating) == 2 or int(rating) == 2.5 or int(rating) == 3 or int(rating) == 3.5):
        return 0
    else: 
        return 1
    
data.traveller_rating = data.traveller_rating.apply(sentiment_rating)


# In[27]:


data.traveller_rating.value_counts() 
#negative wordcount is 891 and positive wordcount is 252


# In[28]:


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(data['reviewContent'], data.traveller_rating, test_size=0.2, random_state=1)


# In[29]:


good = x_train[y_train[y_train == 1].index]
bad = x_train[y_train[y_train == 0].index]
x_train.shape,good.shape,bad.shape


# In[30]:


#word cloud for negative review words
get_ipython().system('pip install wordcloud')

import matplotlib.pyplot as plt
from wordcloud import WordCloud

plt.figure(figsize = (20,20)) #Text Reviews with Poor Ratings
WC = WordCloud(min_font_size = 3,  max_words = 3000 , width = 1600 , height = 800).generate(" ".join(bad))
plt.imshow(WC,interpolation = 'bilinear')


# In[31]:


#Word cloud for positive review words

plt.figure(figsize = (20,20)) #Text Reviews with Good Ratings
WC = WordCloud(min_font_size = 3,  max_words = 3000 , width = 1600 , height = 800).generate(" ".join(good))
plt.imshow(WC,interpolation = 'bilinear')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




