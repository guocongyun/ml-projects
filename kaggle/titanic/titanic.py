import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import string
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
import tensorflow.keras as keras
# os.environ['CUDA_VISIBLE_DEVICES'] = "0"
#%%
pd.options.display.max_columns = None


#%%

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
data_set = pd.concat([train_data, test_data])
# IMPORTANT apply data engineering to both train_data and test_data
# IMPORTANT this can also be done with train_data = np.genfromtxt('./kaggle/titanic/train.csv',delimiter=',') 
# if txt is in correct shape, otherwise will get not constant columns

def print_stats(data):
    print(data)
    print(f'{data.isnull().sum(axis=0)}')
    print(f'{list(map(lambda dta: type(dta),data.iloc[0]))}')

def cleaning_data(data):
    data['Age'].fillna(data['Age'].mean(), inplace=True)
    data['Fare'].fillna(data['Fare'].mean(), inplace=True)
    data['Embarked'].fillna('S', inplace=True)
#%%

def feature_engineering(data):

    title_list = np.array(['Mr', 'Mrs', 'Miss', 'Master', 'Don', 'Rev', 'Dr', 'Mme', 'Ms', 'Major', 'Lady', 'Sir', 'Mlle', 'Col', 'Capt', 'the Countess', 'Jonkheer'])
    titles = np.full(data['Name'].shape, 'Misc', dtype=object)
    # IMPORTANT If you try to assign a long string to a normal numpy array, it truncates the string unless use dtype=object
    for title in title_list:
        mask = list(map(lambda name: title in name,data['Name']))
        # can also use string.find
        if sum(mask) > 10: titles[mask] = title
    data['Title'] = titles

cleaning_data(data_set)
# print_stats(data_set)
feature_engineering(data_set)
# print(data_set['Title'].value_counts()) # IMPORTANT
# TODO find their family and use the dropped data

train_data = data_set.loc[data_set['Survived'].notnull(), :]
test_data = data_set.loc[data_set['Survived'].isnull(), :]
data_set = [train_data, test_data]
print_stats(train_data)
# IMPORTANT train_data['Age'] == train_data.loc[:,'Age']
# NaN != NaN
#%%

# Bar plot for discret features
# fig, ax = plt.subplots(2,2)
# sns.barplot(x=train_data['Sex'], y=train_data['Survived'], ax=ax[0,0])
# sns.barplot(x=train_data['Embarked'], y=train_data['Survived'], ax=ax[0,1])
# sns.barplot(x=train_data['Pclass'], y=train_data['Survived'], ax=ax[1,0])
# sns.barplot(x=train_data['Title'], y=train_data['Survived'], ax=ax[1,1])

# Box plot for continuos features
# fig, ax = plt.subplots(1,3)
# sns.boxplot(x = 'Fare', orient='v', ax=ax[0], data=train_data)
# sns.boxplot(x = 'Age', orient='v', ax=ax[1], data=train_data)
# sns.boxplot(x = 'SibSp', orient='v', ax=ax[2], data=train_data)

# Histogram for continuous  features
# fig, ax = plt.subplots(1,3)
# sns.distplot(train_data[train_data['Survived']==1]['Fare'], ax=ax[0], color='g', label='Survived')
# sns.distplot(train_data[train_data['Survived']==0]['Fare'], ax=ax[0], color='r', label='Died')
# sns.distplot(train_data[train_data['Survived']==1]['Age'], ax=ax[1], color='g', label='Survived')
# sns.distplot(train_data[train_data['Survived']==0]['Age'], ax=ax[1], color='r', label='Died')
# sns.distplot(train_data[train_data['Survived']==1]['SibSp'], ax=ax[2], color='g', label='Survived')
# sns.distplot(train_data[train_data['Survived']==0]['SibSp'], ax=ax[2], color='r', label='Died')

# Pointplot or LinePlot for features that are ordinal
# fig, ax = plt.subplots(1,2)
# sns.pointplot(x='Embarked',y='Survived', hue='Sex', ax=ax[0], data=train_data)
# sns.pointplot(x='Pclass',y='Survived', hue='Sex', ax=ax[1], data=train_data)

# plt.show()

#%%

for df in data_set:
    # IMPORTANT range(4)!=(1,2,3,4)
    df['Fare_cat'] = pd.qcut(df['Fare'], q=4, labels=range(4)).astype(np.int32)
    df['Age_cat'] = pd.qcut(df['Age'], q=4, labels=range(4)).astype(np.int32)
    df['SibSp_cat'] = df['SibSp'].apply(lambda x: 'Alone' if x==0 else('Small' if x>0 and x<5 else('Medium' if x>=5 and x<7 else 'Large')))
    # IMPORTANT df['Title'] == df.Title
    df.Title.replace({'Mr':1, 'Mrs':2, 'Miss':3, 'Master':4, 'Dr':5, 'Col':6, 'Misc':7}, inplace=True)
    df.Sex.replace({'female':0, 'male': 1}, inplace=True)
    df.Embarked.replace({'S':1, 'C':2, 'Q':3}, inplace=True)

# fig, ax = plt.subplots(1,3)
# sns.pointplot(x='Fare_cat', y='Survived', ax=ax[0], data=data_set[0])
# sns.pointplot(x='Age_cat', y='Survived', ax=ax[1], data=data_set[0])
# sns.pointplot(x='SibSp_cat', y='Survived', ax=ax[2], data=data_set[0])
# plt.show()

# IMPORTANT pd.qcut is based on fix frequency,  pd.cut is based on fix distance

#%%

features = ['Age_cat', 'Fare_cat', 'Pclass', 'Sex', 'Embarked', 'Title', 'SibSp_cat']
encoded_features = []

for df in data_set:
    for feature in features:
        # type(df[feature]) == <class 'pandas.core.series.Series'> type(df[feature]).values == np.array
        encoded = OneHotEncoder().fit_transform(df[feature].values.reshape(-1, 1)).toarray()
        n = df[feature].nunique()
        cols = [f'{feature}_{n}' for n in range(1, n + 1)]
        encoded_df = pd.DataFrame(encoded, index=df.index, columns=cols)
        encoded_features.append(encoded_df)



train_encoded = pd.concat([train_data, *encoded_features[:7]], axis=1)
test_encoded = pd.concat([test_data, *encoded_features[7:]], axis=1)
data_set = [train_encoded, test_encoded]

for df in data_set:
    df.drop(['PassengerId', 'Pclass', 'SibSp_cat', 'Cabin', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Embarked', 'SibSp', 'Title', 'Fare_cat', 'Age_cat' ], axis=1,inplace=True) # drop default is axis=0, i.e. drop row

# VERY IMPORTANT type(encoded_features[:7]) == list, type(*encoded_features[:7]) == np.array

#%% 

train_label = data_set[0]['Survived'].to_numpy()
train_data = data_set[0].drop(['Survived'], axis=1)
seed = 1400

train_data, vali_data, train_label, vali_label = train_test_split(train_data, train_label, train_size=0.95, random_state=seed)


#%% 

clf = RandomForestClassifier(criterion='gini',n_estimators=300,max_depth=4,min_samples_split=4,min_samples_leaf=7)
clf.fit(train_data,train_label)
prediction = clf.predict(vali_data)
cm = confusion_matrix(vali_label, prediction)
# print(cm)
# print(classification_report(vali_label, prediction))

# min_samples_split is min(internal node, whether node can continue split) min_samples_leaf is min(leaf, whether splits are legal)

#%% 
#IMPORTANT seed usually generate the same output for different random generators

tf.random.set_seed(seed)
init = keras.initializers.glorot_uniform(seed=seed)
model = keras.models.Sequential()
model.add(keras.layers.Input(shape=(train_data.shape[1],)))
model.add(keras.layers.Dense(360, activation='selu', kernel_initializer=init))

model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Dense(360, activation='selu', kernel_initializer=init))

model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Dense(360, activation='selu', kernel_initializer=init))

model.add(keras.layers.Dense(1, activation='sigmoid'))
model.summary()

model.compile(optimizer='adam', loss = keras.losses.BinaryCrossentropy(), metrics=['accuracy'])

#%%

early_stopping = keras.callbacks.EarlyStopping(monitor='accuracy', patience=3, mode='max', restore_best_weights=True)
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='accuracy', factor=0.1, patience=3, mode='max', min_lr=0)

model.fit(train_data, train_label, epochs = 50, batch_size = 2, callbacks=[reduce_lr, early_stopping], verbose = 1)

# val_loss, val_acc = model.evaluate(vali_data, vali_label, verbose=1)
# print('\nValidation accuracy:', val_acc)

test_data = data_set[1].drop(['Survived'], axis=1) # drop survived==nan

prediction = model.predict(test_data) # since output layer is sigmoid we have to map it to int
prediction = np.array(list(map(lambda data: int(np.round(data)), prediction)))

output = pd.DataFrame({'PassengerId' : range(1,len(prediction)+1), 'Survived' : prediction})
print(output)
output.to_csv('my_submission.csv', index=False)