# -*- coding: utf-8 -*-

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# imputer lib
from sklearn.preprocessing import Imputer

# Encoding categorical data lib
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Dimension reduce
from sklearn.decomposition import PCA

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split

# Feature Scaling
from sklearn.preprocessing import StandardScaler

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import regularizers

import matplotlib.pyplot as plt


def convertNan(num, falseResult):
    return num if pd.isnull(num) is False else falseResult

def processData(dataframe):
    for i, row in dataframe.iterrows():
        areaValue = int(row['area'].split()[0])
        price = row['price']
        legalStatus = row['legal_status']
        title = row['title']
        priceUnit = price.split()[1]
        if(priceUnit == 'Triệu'):
            price = float(price.split()[0].replace(',','.')) * 1000000/22000
        elif(priceUnit == 'Tỷ'):
            price = float(price[0].replace(',','.')) * 1000000000/22000
        else:
            price = np.nan
        
        if("cao cấp" in title.lower() and areaValue >= 70):
            dataframe.at[i, 'premium'] = 1
        else:
            dataframe.at[i,'premium'] = 0
        dataframe.at[i,'area'] = areaValue
        dataframe.at[i,'price'] = price
        dataframe.at[i,'legal_status'] = convertNan(legalStatus, 'Không')
        
    return dataframe;

# Importing the dataset
dataframe = pd.read_csv('ApartmentCSV.csv')
dataframe = dataframe.iloc[:,:12]

# Drop unrelated columns
dataframe.drop(dataframe.columns[[3,5,6,10]], axis=1, inplace=True)

# convert to numeric values
dataset = processData(dataframe)

# data immputation (filling missing data)
imp = Imputer(missing_values='NaN', strategy='mean')
dataset[['bathrooms', 'bedrooms','num_of_floor', 'price']] = imp.fit_transform(dataset[['bathrooms', 'bedrooms','num_of_floor','price']])

# convert from dataframe to numpy arrays
dataset_np = dataset.values

# filter out any appartment that has more than 200m2
dataset_np = dataset_np[(dataset_np[:,0]<200)]

# categorize appartment standard
premium = dataset_np[(dataset_np[:,8] == 1)]
standard =  dataset_np[(dataset_np[:,8] == 0)]

# calculate appartment category properties
minimum_premium_area = np.min(premium[:,0])
minimum_standard_area = np.min(standard[:,0])

# get the max area of each kind
max_premium_area = np.max(premium[:,0])
max_standard_area = np.max(standard[:,0])

# get the cost per meter of each kind
cost_per_meter_premium = premium[:,6]/premium[:,0]
cost_per_meter_standard =  standard[:,6]/standard[:,0]

# get the median cost per meter of each kind
median_pre = np.median(cost_per_meter_premium)
median_sta = np.median(cost_per_meter_standard)

# looking for potential data for premium appartment
pre_standard = standard[((standard[:,6]/standard[:,0]) >= median_pre)]


standard = standard[((standard[:,6]/standard[:,0]) < median_pre)]

# combine two premium dataset
premium = np.concatenate([premium, pre_standard])

for item in premium:
    item[8] = 1
    
for item in standard:
    item[8] = 0

# finalize the dataset, prepare for training
processData = np.concatenate([premium, standard]);

processData = np.delete(processData, 7,1)

# split the features and outputs
X = np.delete(processData, 6,1)
y = processData[:, 6]

# Encoding categorical data
labelencoder_X_3 = LabelEncoder()
X[:, 3] = labelencoder_X_3.fit_transform(X[:, 3])
#
labelencoder_X_4 = LabelEncoder()
X[:, 4] = labelencoder_X_4.fit_transform(X[:, 4])

# flatten categorical data
onehotencoder = OneHotEncoder(categorical_features=[3,4])
X = onehotencoder.fit_transform(X).toarray()

# eliminate unnecessary features
X = X[:,1:]
X = np.delete(X, 21, 1)

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Initialising the ANN
model = Sequential()

# Adding the input layer and the first hidden layer
model.add(Dense(units = 100, kernel_initializer = 'normal', activation = 'relu', input_dim = 29
                ,kernel_regularizer=regularizers.l2(0.01),
                activity_regularizer=regularizers.l1(0.01)))

# drop out features to prevent overfitting
model.add(Dropout(0.5, name='dropout_1'))

## Adding the second hidden layer
model.add(Dense(units = 100, kernel_initializer = 'normal', activation = 'linear'))

# drop out features to prevent overfitting
model.add(Dropout(0.7, name='dropout_2'))

model.add(Dense(units = 50, kernel_initializer = 'normal', activation = 'relu'))

# Adding the output layer
model.add(Dense(units = 1, kernel_initializer = 'normal'))

# Compiling the ANN
model.compile(optimizer = 'adam', loss = 'mean_absolute_error', metrics = ['accuracy'])

# Fitting the ANN to the Training set
model.fit(X_train, y_train, batch_size = 10, epochs = 200)

# Predicting the Test set results
y_pred = model.predict(X_test)


plottingData = pd.DataFrame({'x':y_test, 'y':y_pred[:,0]});
plt.plot(plottingData)

mean_entropy = np.sum(abs(y_test-y_pred[:,0]))/len(y_test)

max_deviation = np.max(abs(y_test-y_pred[:,0]))

deviation = np.array(abs(y_test-y_pred[:,0])).astype(np.float64)

# predicted data that are below 5000 dollars difference
diff = abs(y_pred[:,0] - y_test) < 5000


count = 0
for element in diff:
    if element == True:
        count = count + 1

# percentage of <5000 difference data in the whole testing data set
percentage = count/len(diff)

rmse = np.sqrt(np.sum(y_pred[:,0] - y_test)**2)
