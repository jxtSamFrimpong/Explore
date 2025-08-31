import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn as sk
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('../Data/car.data', header=None)
df_columns = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "class"]
predict_column = ['class']
df.columns = df_columns

print(df.head())

le = LabelEncoder()
for col in df.select_dtypes(include=['object']).columns:
    df[col] = le.fit_transform(df[col])

print(df.head())

X = df[[i for i in df_columns if i != 'class']]
y = df['class']
#Or Any of the Methods below
# y = df[['class']].values.ravel()
#y = df[['class']].squeeze()

print(X.head())
print(y[:5])


x_train, x_test, y_train, y_test = sk.model_selection.train_test_split(X, y, test_size=0.2)
print('done with test split')

model = KNeighborsClassifier(n_neighbors=3)
print('about to fit the model')
model.fit(x_train, y_train)
print('done with fit')
acc = model.score(x_test, y_test)
print(acc)

predictions = model.predict(x_test)
pred_names = le.inverse_transform(predictions)
class_names = le.inverse_transform(sorted(set(predictions)))
# print(class_names)
# print(pred_names)


# #Bar chart
# x_test['predicted_class'] = predictions
# plt.figure(figsize=(6,4))
# sns.countplot(x=pred_names, order=class_names)
# plt.title("Distribution of Predicted Classes")
# plt.xlabel("Class")
# plt.ylabel("Count")
# plt.xticks(rotation=45)
# plt.show()
#
# # 1.ii. Pie chart for class proportions
# class_counts = pd.Series(pred_names).value_counts()
# print('which class count', class_counts, type(class_counts))
# plt.figure(figsize=(6,6))
# plt.pie(class_counts, labels=class_counts.index, autopct='%1.1f%%', startangle=90)
# plt.title("Class Proportions")
# plt.show()

# #
# if x_test['predicted_class'].dtype == 'object':
#     le = LabelEncoder()
#     x_test['pred_class_num'] = le.fit_transform(x_test['predicted_class'])
# else:
#     x_test['pred_class_num'] = x_test['predicted_class']
#
# numeric_columns = x_test.select_dtypes(include=['int64', 'float64']).columns
# x_all_columns = x_test.columns.tolist()

# for col in numeric_columns:
#     if col != 'pred_class_num':  # avoid plotting the class against itself
#         plt.figure(figsize=(6,4))
#         plt.scatter(x_test[col], range(len(x_test)),
#                     c=x_test['pred_class_num'], cmap='viridis', alpha=0.7)
#         plt.colorbar(label='Predicted Class')
#         plt.title(f"{col} vs Predicted Class")
#         plt.xlabel(col)
#         plt.ylabel("Sample Index")
#         plt.show()

x_test.reset_index(drop=True, inplace=True)
y_test.reset_index(drop=True, inplace=True)
for x in range(len(predictions)):
    print('Input Data\n', x_test.iloc[x])
    print('Predicted Value', class_names[predictions[x]])
    # print(y_test, predictions)
    print('Actual Value', class_names[y_test.iloc[x]])
    print('\n\n\n')