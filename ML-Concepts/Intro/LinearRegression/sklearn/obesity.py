import  numpy as np
import pandas as pd
import sklearn as sk
import matplotlib.pyplot as plt
import pickle
from matplotlib import style, pyplot

df = pd.read_csv('../Data/ObesityDataSet_raw_and_data_sinthetic.csv', sep=',')
df['Gender'] = df['Gender'].astype('category')
df['Gender_numeric'] = df['Gender'].cat.codes

df['FAVC'] = df['FAVC'].replace({'yes': 1, 'no': 0})

df['CAEC'] = df['CAEC'].astype('category')
df['CAEC_numeric'] = df['CAEC'].cat.codes

df['SMOKE'] = df['SMOKE'].replace({'yes': 1, 'no': 0})
df['SCC'] = df['SCC'].replace({'yes': 1, 'no': 0})

df['CALC'] = df['CALC'].astype('category')
df['CALC_numeric'] = df['CALC'].cat.codes


df['MTRANS'] = df['MTRANS'].astype('category')
df['MTRANS_numeric'] = df['MTRANS'].cat.codes

df['NObeyesdad'] = df['NObeyesdad'].astype('category')
df['NObeyesdad_numeric'] = df['NObeyesdad'].cat.codes

# print('data head\n', data.head())
#
# print(data.shape)

print(df.head())

predict_variables = ['NObeyesdad_numeric']
input_variables = ['FAF', 'TUE', 'CALC_numeric', 'MTRANS_numeric']

x = np.array(df[input_variables])
print(x[:5])
y = np.array(df[predict_variables])
print(y[:5])

x_train, x_test, y_train, y_test = sk.model_selection.train_test_split(x, y, test_size=0.2)
best_model_so_far = 0

# for _ in range(20):
#     linear = sk.linear_model.LinearRegression()
#     linear.fit(x_train, y_train)
#     acc = linear.score(x_test, y_test)
#     print(acc)
#     if acc > best_model_so_far:
#         with open("saved_model.pkl", "wb") as f:
#             pickle.dump(linear, f)

# saved_model = pickle.load(open("obesity.pkl", "rb"))
#
#
# print('Coefficient: \n', saved_model.coef_)
# print('Intercept: \n', saved_model.intercept_)
# acc = saved_model.score(x_test, y_test)
# print('Model Accuracy: \n', acc)

# predictions = saved_model.predict(x_test)
#
# for x in range(len(predictions)):
#     print('Predicted value: ', predictions[x], 'Input Value: ', x_test[x], 'Actual Value: ', y_test[x])


#scattter diagram
x_axis = 'CALC'
y_axis = 'NObeyesdad'
style.use('ggplot')
pyplot.scatter(df[x_axis], df[y_axis])
pyplot.xlabel(x_axis)
pyplot.ylabel("Obesity Levels")
pyplot.show()


# #bar chart
# x_axis = 'Gender'
# y_axis = 'NObeyesdad'
# #pyplot.scatter(df[x_axis], df[y_axis])
# plt.bar(df[x_axis], df[y_axis])  # For vertical bars
# # plt.barh(categories, values) # For horizontal bars
# pyplot.xlabel(x_axis)
# pyplot.ylabel("Obesity Levels")
# pyplot.show()