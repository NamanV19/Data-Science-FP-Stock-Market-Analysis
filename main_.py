from sklearn.svm import SVR
import pandas as pd
import matplotlib.pyplot as plt
from plotnine import *

plt.style.use("fivethirtyeight")

df = pd.read_csv("aaap.us.txt")
print(df.isnull().sum())

# Data cleaning - remove the column openInt as every value in this column is 0
df = df.drop(['OpenInt'], axis=1)
print(df)

true_data = df.tail(1)
print(true_data)
df = df.head(len(df)-1)

days = []
close = []

df_days = df.loc[:, 'Date']
df_close = df.loc[:, 'Close']

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

predicted_date = [[31]]
print("The predicted date is: ", svr_with_radial_kernel.predict(predicted_date))




