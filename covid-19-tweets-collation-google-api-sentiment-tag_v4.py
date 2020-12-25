# -*- coding: utf-8 -*-
"""
Created on Fri May  1 17:21:21 2020

@author: Ashwini
"""

import pandas as pd
import google_sentiment_example
import math


############ defining all functions ################################3
	

def collate_data():
    dt_range = pd.date_range(start="2020-03-12",end="2020-04-15")
    
    tweets_df = pd.DataFrame(columns=['status_id', 'user_id', 'created_at', 'screen_name', 'text', 'source',
           'reply_to_status_id', 'reply_to_user_id', 'reply_to_screen_name',
           'is_quote', 'is_retweet', 'favourites_count', 'retweet_count',
           'country_code', 'place_full_name', 'place_type', 'followers_count',
           'friends_count', 'account_lang', 'account_created_at', 'verified',
           'lang'])
    
    for date in dt_range:
        path= "Tweets Data\\" + str(date.date()) + " Coronavirus Tweets.csv"
        temp_df=pd.read_csv(path)
        temp_df=temp_df[temp_df['lang']=="en"].sample(n = 2858) 
        tweets_df= tweets_df.append(temp_df,ignore_index=True)
        if (len(tweets_df)>500000):
            break
    
    tweets_df.to_csv('Translated_tweets_new_v2.1.csv',encoding='utf-8')
    return tweets_df

def count_records():
    dt_range = pd.date_range(start="2020-03-12",end="2020-04-15")
    total_len=0
    rec_l=[]
    for date in dt_range:
        path= "Tweets Data\\" + str(date.date()) + " Coronavirus Tweets.csv"
        temp_df=pd.read_csv(path)
        l=len(temp_df)
        #eng_l=len(temp_df[temp_df['lang']=="en"])
        #total_len=total_len+l
        rec_l.append(list(temp_df['lang'].unique()))
        #print(str(date.date())+" "+str(l)+" "+str(eng_l))
        print(str(date.date())+" "+str(l))
        #rec_l.append([date.date(),l,eng_l])
    print(set(rec_l))
    return total_len


def read_data():
    tweets_df=pd.read_csv('Translated_tweets_new_v2.1.csv')
    return tweets_df

def save_data(df):
    df.to_csv('Translated_tweets_new_v2.1.csv',encoding='utf-8')
    
def assign_google_sentiments(tweets_df):
    t_unit_len=0 #unitl length
    ov_len=0 #overall number of characters
    n_units=0#number of units per google doc 100characters each
    l=len(tweets_df)
    lim=500000
    
    ##sample_analyze_sentiment(tweet)
    ##sample_analyze_sentiment(tweet,1)
    
    for i in range(0,l):
        tweet=tweets_df.iloc[i]['text']
        t_unit_len=len(tweet)
        ov_len=ov_len+t_unit_len
        n_units=n_units+int(math.ceil(t_unit_len/1000))
        response=sample_analyze_sentiment(tweet)
        sentences = [sentence.text.content for sentence in response.sentences]
        sentence_sen_magnitude=[sentence.sentiment.score for sentence in response.sentences]
        sentence_sen_scores=[sentence.sentiment.magnitude for sentence in response.sentences]
        tweets_df.at[i,str('overall_sentiment_magnitude')]=response.document_sentiment.magnitude
        tweets_df.at[i,str('overall_sentiment_score')]=response.document_sentiment.score
        tweets_df.at[i,str('sentences')] = str(sentences)
        tweets_df.at[i,str('sentence_sen_magnitude')]= str(sentence_sen_magnitude)
        tweets_df.at[i,str('sentence_sen_scores')]= str(sentence_sen_scores)
		if(n_units%1000==0): # counting every 1000 units
            print("total units processed ",n_units)
            print("total characters processed ",ov_len)
            if(n_units%lim==0):
                break
    return tweets_df


###################End of functions definition ###############################





################## Code for each section of the analysis ############################

#################### Section 1, 2 & 3: Data Collection, Collation & Preparation ##########################
"""
This consists of donwloading covid 19 related tweets data from Kaggle public URL and then collating the downloaded tweets into a single file
Data was manually downloaded from https://www.kaggle.com/smid80/coronavirus-covid19-tweets (for 13-March-20 to 28-March-20) and
from https://www.kaggle.com/smid80/coronavirus-covid19-tweets-early-april (for 29-march-20 to 15-April-20)
and saved in Tweets Data subfolder under the same directory as this code

The code reads and collates in 11.32 MM tweets and creates a simple random sample of 110K tweets using simple random sampling in section 3

"""

tweets_df=collate_data()

############################### End of Section 1,2 & 3 Data Collection, Collation and Preparation ########################





#################### Section 4: Sentiment_tag using Google NLP API ##########################

tweets_df = read_data()
tweets_df=assign_google_sentiments(tweets_df)
tweets_df.to_csv('Translated_tweets_new_v2.1.csv',encoding='utf-8')

"""
added sentiment score, polarity of the sentiment (strength) to the file
"""

twt_file1=pd.read_csv('Translated_tweets_new_v2.1.csv')

#drop unmaned columns if needed
"""twt_file1 = twt_file1.loc[:, ~twt_file1.columns.str.contains('^Unnamed')]

twt_file2=pd.read_csv('Translated_tweets_new_v2.2.csv')
twt_file2 = twt_file2.loc[:, ~twt_file2.columns.str.contains('^Unnamed')]

twt_file1['text'].head(100000).to_csv('Tweets_new_v1.csv',encoding='utf-8')
twt_file1=twt_file1.head(100000)
twt_file1.to_csv('Translated_tweets_new_v2.1.csv',encoding='utf-8')
"""


######## Tagging the setiment to four types, positive, negative, neutral and mixed ##############

save_data(twt_file1)

#sample=twt_file1[twt_file1['overall_sentiment_score']==0.0 and twt_file1['overall_sentiment_magnitude']>0.5].head()

c=0
for i in range(0,l):
    score=twt_file1.at[i,str('overall_sentiment_score')]
    magnitude=twt_file1.at[i,str('overall_sentiment_magnitude')]
    if (magnitude<=0.25):
        twt_file1.at[i,str('Sentiment_tag')]='Neutral'
    else:
        if (score>=0.2):
            twt_file1.at[i,str('Sentiment_tag')]='Positive'
        elif (score<=-0.2):
            twt_file1.at[i,str('Sentiment_tag')]='Negative'
        else:
            twt_file1.at[i,str('Sentiment_tag')]='Mixed'
    c=c+1
    if(c%20000==0):
        print("c=",c)
        
save_data(twt_file1)
    

#################################################################################################
## adding tweets to separate files by sentiment tag for further analysis #######################

#positive
sample=twt_file1[twt_file1['Sentiment_tag']=="Positive"][['text','Sentiment_tag']]
sample.to_csv('Translated_tweets_new_v2.1_Positive.csv',encoding='utf-8')
sample.to_excel('Translated_tweets_new_v2.1_Positive.xlsx')


#negative
sample=twt_file1[twt_file1['Sentiment_tag']=="Negative"][['text','Sentiment_tag']]
sample.to_csv('Translated_tweets_new_v2.1_Negative.csv',encoding='utf-8')
sample.to_excel('Translated_tweets_new_v2.1_Negative.xlsx')

        
#neutral
sample=twt_file1[twt_file1['Sentiment_tag']=="Neutral"][['text','Sentiment_tag']]
sample.to_csv('Translated_tweets_new_v2.1_Neutral.csv',encoding='utf-8')
sample.to_excel('Translated_tweets_new_v2.1_Neutral.xlsx')


#mixed
sample=twt_file1[twt_file1['Sentiment_tag']=="Mixed"][['text','Sentiment_tag']]
sample.to_csv('Translated_tweets_new_v2.1_Mixed.csv',encoding='utf-8')
sample.to_excel('Translated_tweets_new_v2.1_Mixed.xlsx')


#################### Section 4: Sentiment_tag using Google NLP API ##########################


####################### Section 5 done on Excel and Tableau ###############################################
