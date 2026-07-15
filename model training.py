import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.metrics import classification_report

#load dataset
df=pd.read_csv(r"C:\Users\SHIVANSHI SHRIVASTAV\Desktop\credit_card_fraud_project\data\credit_card_fraud_custom (1).csv")
#feature engineering
df['transaction_time'] = pd.to_datetime(df['transaction_time'])
df['hour'] = df['transaction_time'].dt.hour
df['transaction_ratio']= df['transaction_amount'] / df['account_balance']
df['high_amount_flag'] = df['transaction_amount'].apply(lambda i :1 if i>50000 else 0)
#drop useless column
df.drop(['transaction_id','transaction_time'],axis = 1,inplace = True)
#encoder
le =LabelEncoder()
df ['user_gender'] = le.fit_transform(df['user_gender'])
df['device_type']=le.fit_transform(df['device_type'])
df =pd.get_dummies(df,columns=['merchant','category','user_city'],dtype=int, drop_first=True)
# split data
x = df.drop('is_fraud',axis = 1)
y = df['is_fraud']

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42)

#scaling
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
# model training
model = RandomForestClassifier(random_state=42)
model.fit(x_train, y_train)

# model
with open('model/model.pkl','wb') as f:
    pickle.dump(model,f)

    # scaler
with open('model/scaler.pkl','wb') as f:
    pickle.dump(scaler,f)