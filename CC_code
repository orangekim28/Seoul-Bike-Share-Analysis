# %%
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

# %%
df = pd.read_csv("SeoulBikeData.csv", encoding= 'unicode_escape')

df.columns = ['Date', 'BikeCount', 'Hour', 'Temp', 'Humid', 'WindSpeed', 'Visibility', 
              'DewPTemp', 'SolarRad', 'Rainfall', 'Snowfall', 'Season', 'Holiday', 'Functioning']

#%% 
#Converting the Date column in Datetime Dtype
df['Date']=pd.to_datetime(df['Date'], format='%d/%m/%Y')

#Breaking Down the Date into 3 Components
df['Day']=df['Date'].dt.day
df['Month']=df['Date'].dt.month
df['Year']=df['Date'].dt.year

df
# %%
df.head()
df.info()
# %%
df.isnull().sum() # have no null values

# %%
# Convert 'Season' column to categorical data type with specific categories
df['Season'] = pd.Categorical(df['Season'], categories=['Winter', 'Spring', 'Summer', 'Autumn'])
df['Holiday'] = pd.Categorical(df['Holiday'], categories=['No Holiday', 'Holiday'])
df['Functioning'] = pd.Categorical(df['Functioning'], categories=['Yes', 'No'])


# Plot the count of rented bikes according to seasons
sns.countplot(data=df, x='Season', palette='pastel')
plt.title('Rented Bike count according to Seasons')
plt.show()

sns.countplot(data=df, x='Holiday', palette='pastel')
plt.title('Rented Bike count according to Holiday')
plt.show()


#%%
sns.displot(df['BikeCount'],kde=True,color='g')

plt.figure(figsize=(8,6))
lis=['Hour','Temp',	'Humid','WindSpeed','Visibility','DewPTemp','SolarRad',	'Rainfall','Snowfall']
for i in lis:
  sns.displot(df[i],kde=True)


# %%
plt.figure(figsize=(12,8))
sns.barplot(data=df, x='Hour', y='BikeCount')
plt.title('Hourly Bike Rents')
plt.show()

# %%
discrete = ['Season','Holiday','Functioning','Month','Year']
for i in discrete:
  plt.figure(figsize=(10,6))
  sns.barplot(x=i,y='BikeCount',data=df,palette='magma')

#%%
df['Month'].value_counts()


# %%
def corr_heatmap(df):
    # Exclude non-numeric columns
    numeric_columns = df.select_dtypes(include=np.number)
    
    plt.figure(figsize=(8, 8))
    mask = np.triu(np.ones_like(numeric_columns.corr(), dtype=bool))
    sns.heatmap(numeric_columns.corr(), mask=mask, vmin=-1, vmax=1, annot=True, fmt='.2f', cmap='icefire')

corr_heatmap(df)

# %%%%%%%%%%%%%%%%%%%%%%%%% Model Building %%%%%%%%%%%%%%%%%%%%%%%%%%%

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.pipeline import make_pipeline
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error

# %% Feature and target
X = df.drop(columns=['BikeCount', 'Date'])
y = df.BikeCount

# Encoding
le = LabelEncoder()
categ = ['Holiday', 'Functioning', 'Season']
X[categ] = X[categ].apply(le.fit_transform)

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


#%%%%%%%%%%%%%%%%%%%%% Linear Regression %%%%%%%%%%%%%%%%%%%

# Initialize the linear regression model
model1 = LinearRegression()

# Train the model
model1.fit(X_train, y_train)

# Make predictions on the test set
y_pred1 = model1.predict(X_test)

# Evaluate the model
r2_lr = r2_score(y_test, y_pred1)

cv_lr = cross_val_score(model1, X_train, y_train, cv=10, scoring='r2')
mean_r2_lr = np.mean(cv_lr)
print('Linear Regression Mean R2:', mean_r2_lr)


#%%%%%%%%%%%%%%%%%%%%% Random Forest %%%%%%%%%%%%%%%%%%%
model2 = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
model2.fit(X_train, y_train)

# Make predictions on the test set
y_pred2 = model2.predict(X_test)

# Evaluate the model
r2_rf = r2_score(y_test, y_pred2)
cv_rf = cross_val_score(model2, X_train, y_train, cv=10, scoring='r2')
mean_r2_rf = np.mean(cv_rf)
print('Random Forest Mean R2:', mean_r2_rf)


# %%%%%%%%%%%%%%%%%%%%%% KNN %%%%%%%%%%%%%%%%%%%%%%%%%
knn = KNeighborsRegressor()
knn.fit(X_train, y_train)

y_pred_knn = knn.predict(X_test)
# graph['knn_pred'] = y_pred
# plt.figure(figsize=(12,8))
# sns.scatterplot(data=graph, x='y_pred', y='BikeCount')
# sns.lineplot(x=[0,200], y=[0,200], color='red', linestyle='--')
# plt.show()

# Evaluate the model
r2_knn = r2_score(y_test, y_pred_knn)
cv_knn = cross_val_score(knn, X_train, y_train, cv=10, scoring='r2')
mean_r2_knn = np.mean(cv_knn)
print('K-Nearest Neighbors Mean R2:', mean_r2_knn)

# %%%%%%%%%%%%%%%%%%%%%%%%%% baseline XGBoost %%%%%%%%%%%%%%%%%%%%
pipe = make_pipeline(StandardScaler(), XGBRegressor())

# Cross-validation
cv = cross_val_score(pipe, X_train, y_train, cv=10)
print('Mean R2', np.mean(cv))


# %%%%%%%%%%%%%%%%%%%%%%%%%%% tuned XGBoost %%%%%%%%%%%%%%%%%%%%%%%%%
pipe = make_pipeline(StandardScaler(), XGBRegressor(n_estimators=70, max_depth=10, eta=0.08,
                                                    subsample=0.8, reg_lambda=1.2))

# Cross-validation
cv = cross_val_score(pipe, X_train, y_train, cv=10)
print('Mean R2', np.mean(cv))


#%%
# Fit and predict
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
plt.figure(figsize=(7,7))

# 1:1 line
x = np.linspace(0,3500,10)
y = x

# Scatter plot predicted vs. actual
plt.scatter(y_pred, y_test)
plt.plot(x, y, c='r')
plt.gca().set_aspect('equal')
plt.xlabel('Predicted Bike Rents', size=12)
plt.ylabel('Actual Bike Rents', size=12)
plt.title('Predicted vs. Actual Bike Rents', size=20)
plt.xlim(0,3500)
plt.ylim(0,3500)

#%%
# Create a pd.Series of features importances
fimp = pipe.steps[1][1].feature_importances_
importances = pd.Series(data=fimp,
                        index= X_train.columns)

# Sort importances
importances_sorted = importances.sort_values()

# Draw a horizontal barplot of importances_sorted
importances_sorted.plot(kind='barh', color='red')
plt.title('Features Importances')
plt.show()

# Best Mean R2 score is with XG Boost model.