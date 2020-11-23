from sklearn.svm import SVR
import pandas as pd
import matplotlib.pyplot as plt
from plotnine import *

df = pd.read_csv("Tadawul_stcks.csv")
print(df)

# The number of null values in each column
print(df.isnull().sum())

# Data cleaning - dropping columns which are not necessary to process the data
df = df.drop(['symbol', 'name', 'sectoer'], axis=1)
print(df)


# For trial and experiment purposes
df_trial = df[5:24]
true_data = df.loc[[24]]
print(true_data)
print(df_trial)

days = []
close = []

df_days = df_trial.loc[:, 'date']
df_close = df_trial.loc[:, 'close']

for day in df_days:
    days.append([int(day.split('-')[2])])

for close_price in df_close:
    close.append(float(close_price))

svr_with_radial_kernel = SVR(kernel='rbf', C=1000.0, gamma=0.15)
svr_with_radial_kernel.fit(days, close)

# print(ggplot(df, aes(x=day, y=close)) + geom_point())

plt.figure(figsize=(16, 8))
plt.scatter(days, close, color='red', label='Data')
plt.plot(days, svr_with_radial_kernel.predict(days), color='green', label='RBF model')
plt.legend()
plt.show()

predicted_date = [[2]]
print("The predicted date is: ", svr_with_radial_kernel.predict(predicted_date))

