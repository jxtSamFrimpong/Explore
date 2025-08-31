import  numpy as np
import pandas as pd
import sklearn as sk
import matplotlib.pyplot as plt
import pickle
from matplotlib import style, pyplot

data = pd.read_csv('../student+performance/student/student-mat.csv', sep=';')

# print('data head\n', data.head())
#
# print(data.shape)

data = data[["studytime", "G1", "G2", "G3", "failures", "absences"][:]]
print(data.head())

predict_variables = ['G3']

x = np.array(data.drop(columns=predict_variables))
print(x[:5])
y = np.array(data[predict_variables])
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

saved_model = pickle.load(open("saved_model.pkl", "rb"))


print('Coefficient: \n', saved_model.coef_)
print('Intercept: \n', saved_model.intercept_)
acc = saved_model.score(x_test, y_test)
print('Model Accuracy: \n', acc)

predictions = saved_model.predict(x_test)

for x in range(len(predictions)):
    print('Predicted value: ', predictions[x], 'Input Value: ', x_test[x], 'Actual Value: ', y_test[x])


x_axis = "G1"
y_axis = predict_variables
style.use('ggplot')
pyplot.scatter(data[x_axis], data[y_axis])
pyplot.xlabel(x_axis)
pyplot.ylabel("Final Grade")
pyplot.show()