from category_encoders import OrdinalEncoder
from pandas import value_counts
from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import LabelEncoder

"""To read the dataset and print first 10 lines"""
data_set=pd.read_csv("car_price_dataset.csv")
print(data_set.head(10))

"""To read the dataset and print last 10 lines"""
data_set=pd.read_csv("car_price_dataset.csv")
print(data_set.tail(10))

"""slicing """
print(data_set[4:10])
print(data_set[10:])
print(data_set.loc[5:10])
print(data_set.loc[2:10],"Year")
print(data_set.iloc[5:10,0:2])

"""Read Column names"""
column_names=data_set.columns
print(column_names)

"""to know columns datatypes"""
column_type=data_set.dtypes
print(column_type)

column_type=data_set[["Year","Brand","Price","Engine_Size"]].dtypes
print(column_type)

"""detailing about entire dataset & information &Shape of dataset"""
print(data_set.describe())
print(data_set.info())
print(data_set.shape)
print(data_set.ndim)

"""Remove unwanted columns"""
data_set.drop(columns=['Fuel_Type', 'Transmission','Model','Doors'],inplace=True)
print(data_set.head(10))

"""Rename of Columns"""
data_set=data_set.rename(columns={"Engine_Size":"Size","Owner_Count":"Owners"})
print(data_set.head(10))

"""To Find Null Data"""
print(data_set.isnull())
print(data_set.isnull().sum().sum())

"""Find the categorical data count"""
categorical_data_count=data_set['Brand'].unique()
print( categorical_data_count)

categorical_data_count=data_set['Brand'].nunique()
print( categorical_data_count)

categorical_data_count=data_set['Brand'].value_counts()
print( categorical_data_count)

"""Encoding using ordinal encoder for converting categorical to numerical value for brand"""
label_encoding_object =OrdinalEncoder()
data_set['Brand']=label_encoding_object.fit_transform(data_set[['Brand']])
print(data_set)

"""Min Max Scaler values from 0 to 1 """
# min_max_scaler_object=MinMaxScaler()
# data_set['Price'] = min_max_scaler_object.fit_transform(data_set[['Price']])
# print(data_set.head(10))
#
# min_max_scaler_object=MinMaxScaler()
# data_set['Mileage'] = min_max_scaler_object.fit_transform(data_set[['Mileage']])
# print(data_set.head(10))

"""Standard scaler values from -1 to 1"""
# standard_scaler_object=StandardScaler()
# data_set['Year'] = standard_scaler_object.fit_transform(data_set[['Year']])
# print(data_set.head(10))

# standard_scaler_object=StandardScaler()
# data_set['Mileage'] = standard_scaler_object.fit_transform(data_set[['Mileage']])
# print(data_set.head(10))

"""Fit the model"""
df=pd.DataFrame(data_set)
x=df[['Year']]
y=df['Price']
print(df)
model=LinearRegression()
model.fit(x,y)
future_prediction=model.predict(pd.DataFrame({"Year":[2030,2032]}))
print("Future:",future_prediction)

"""Pyplot"""
fig,axes=plt.subplots(3,1,figsize=(5,5))
axes[0].boxplot(df['Year'])
axes[0].boxplot(df['Mileage'])
axes[0].boxplot(df['Price'])
plt.tight_layout()
plt.show()

"""Pairplot"""
sns.pairplot(df,x_vars=["Year","Mileage"],y_vars=['Price'],height=4,aspect=1,kind="scatter")
plt.show()

"""Heatmap"""
sns.heatmap(df.corr(),cmap="magma",annot=True)
plt.show()


"""Train and Test of Data"""
x=df[['Year']]
y=df['Price']
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8,random_state=10)
model=LinearRegression()
model.fit(x_train,y_train)
output=model.predict(x_test)
print(output)

"""Testing Accuracy"""
testing_accuracy=r2_score(y_test,output)
print("Accuracy",testing_accuracy)

