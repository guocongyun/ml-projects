import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import string
#%%
pd.options.display.max_columns = None


#%%

train_data = pd.read_csv('./kaggle/titanic/train.csv')
test_data = pd.read_csv('./kaggle/titanic/test.csv')
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
    data.drop('PassengerId', axis=1,inplace=True) # drop default is axis=0, i.e. drop row
    data.drop('Cabin', axis=1,inplace=True)
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


# IMPORTANT pd.qcut is based on fix frequency,  pd.cut is based on fix distance
