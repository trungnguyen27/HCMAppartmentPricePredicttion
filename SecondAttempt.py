# Artificial Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# pip install tensorflow

# Installing Keras
# pip install --upgrade keras

# Part 1 - Data Preprocessing

#lib imports

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


def convertNan(num, falseResult):
    return num if pd.isnull(num) is False else falseResult

def processData(dataframe):
    for i, row in dataframe.iterrows():
        areaValue = int(row['area'].split()[0])
        #frontStreetLength = row['front_street_length']
        price = row['price']
        legalStatus = row['legal_status']
        #numOfFloor = row['num_of_floor']
        #bathrooms = row['bathrooms']
        priceUnit = price.split()[1]
        if(priceUnit == 'Triệu'):
            price = float(price.split()[0].replace(',','.')) * 1000000/22000
        elif(priceUnit == 'Tỷ'):
            price = float(price[0].replace(',','.')) * 1000000000/22000
        else:
            price = np.nan
        #frontStreetLength = np.nan if pd.isnull(frontStreetLength) is True else float(frontStreetLength.split()[0].replace(',','.'))
        dataframe.at[i,'area'] = areaValue
        #dataframe.at[i,'front_street_length'] = frontStreetLength
        dataframe.at[i,'price'] = price
        dataset.at[i,'legal_status'] = convertNan(legalStatus, 'Không')
        #dataset.at[i,'num_of_floor'] = float(convertNan(numOfFloor, 0))
        #dataset.at[i,'bathrooms'] = float(convertNan(bathrooms, 0))
    return dataframe;

# Importing the dataset
dataset = pd.read_csv('ApartmentCSV.csv')
dataset = dataset.iloc[:,:10]
dataset.drop(dataset.columns[[3,5,6]], axis=1, inplace=True)

# dataset = dataset.query('district=="Huyện Bình Chánh"')

# dataset = dataset.query('house_type == "Nhà phố"')

# dataset.drop(['house_type','legal_status', 'district'], axis=1, inplace=True)

dataset = processData(dataset)

# data immputation

imp = Imputer(missing_values='NaN', strategy='mean')
dataset[['bathrooms', 'bedrooms','num_of_floor', 'price']] = imp.fit_transform(dataset[['bathrooms', 'bedrooms','num_of_floor','price']])


X = dataset.iloc[:, 0:6].values
y = dataset.iloc[:, 6].values


# Encoding categorical data
labelencoder_X_3 = LabelEncoder()
X[:, 3] = labelencoder_X_3.fit_transform(X[:, 3])
#
labelencoder_X_5 = LabelEncoder()
X[:, 4] = labelencoder_X_5.fit_transform(X[:, 4])
##
#labelencoder_X_6 = LabelEncoder()
#X[:, 6] = labelencoder_X_6.fit_transform(X[:, 6])
#
#
onehotencoder = OneHotEncoder(categorical_features=[3,4])
X = onehotencoder.fit_transform(X).toarray()

X = X[:,1:]

X = np.delete(X, 21, 1)

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# define base model
def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(units = 30, input_dim=30, kernel_initializer='normal', activation='relu'))
	model.add(Dense(units = 1, kernel_initializer='normal'))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model


from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)
# evaluate model with standardized dataset
estimator = KerasRegressor(build_fn=baseline_model, epochs=100, batch_size=5, verbose=0)

kfold = KFold(n_splits=10, random_state=seed)
results = cross_val_score(estimator, X, y, cv=kfold)

# Dimension reduce
pca = PCA(n_components=20)
X = pca.fit_transform(X)

# Part 2 - Now let's make the ANN!

# Importing the Keras libraries and packages

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 30, kernel_initializer = 'normal', activation = 'relu', input_dim = 28,kernel_regularizer=regularizers.l2(0.01),
            activity_regularizer=regularizers.l1(0.01)))

classifier.add(Dropout(0.5, name='dropout_1'))

# Adding the second hidden layer
classifier.add(Dense(units = 30, kernel_initializer = 'normal', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'normal'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 5, epochs = 100)

# Part 3 - Making predictions and evaluating the model

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

kfold = KFold(n_splits=10, random_state=seed)
results = cross_val_score(estimator, X, Y, cv=kfold)

print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))

# Predicting the Test set results
y_pred = classifier.predict(X_test)

rounded_pred = np.round(y_pred[:,0],0).astype(int)

diff = abs(y_pred[:,0] - y_train) < 5000

y_pred = y_pred[(y_pred < 500000)]
y_test = y_test[(y_test<500000)][:159]

rmse = np.sqrt(np.sum(y_pred[:,0] - y_test)**2)

count = 0
for element in diff:
    if element == True:
        count = count + 1

percentage = count/len(diff)

new_prediction = classifier.predict(sc.transform(np.array([[0,0,0,0, 0,0,0,0 ,0,0,0,0 ,0,0,0,0 ,0,0,0,0 ,0 ,1,0,0 ,62,2,2,17.24]])))

difference = y_test/((y_pred[1] - y_test)/1000000000)

score = classifier.evaluate(X_test, y_test, batch_size=16)

# Plot
 import matplotlib.pyplot as plt
 
 plottingData = pd.DataFrame({'x':y_test, 'y':y_pred});
plt.plot(y_test, y_pred[:,0])
plt.plot(plottingData)
plt.plot(y_pred)
plt.plot(rounded_pred)