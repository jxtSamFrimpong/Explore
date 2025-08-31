import pandas as pd
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

df = pd.read_csv('~/Dev/ML-AI/Explore/ML-Concepts/Intro/KNN/Data/car.data', header=None)
df.columns = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "class"]

print(df.head())


#OPTION 1: transform only non numerical columns
print('transform only non numerical columns')
le = LabelEncoder()
for col in df.select_dtypes(include=['object']).columns:
    df[col] = le.fit_transform(df[col])

print(df.head())


#OPTION 2: transform only selected columns
print('transform only selected columns')
df = pd.read_csv('~/Dev/ML-AI/Explore/ML-Concepts/Intro/KNN/Data/car.data', header=None)
df.columns = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "class"]

#intentionally not included class from columns to include
cols_to_encode = ["buying", "maint", "doors", "persons", "lug_boot", "safety"]
for col in cols_to_encode:
    df[col] = le.fit_transform(df[col])
print(df.head())


#Option 3: Exclude specific columns
print('Exclude specific columns')
df = pd.read_csv('~/Dev/ML-AI/Explore/ML-Concepts/Intro/KNN/Data/car.data', header=None)
df.columns = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "class"]

exclude = ['class', 'safety']
for col in df.columns.difference(exclude):
    if df[col].dtype == 'object':
        df[col] = le.fit_transform(df[col])
print(df.head())


#Option 4: Use OrdinalEncoder for multiple columns at once
print('Use OrdinalEncoder for multiple columns at once')
df = pd.read_csv('~/Dev/ML-AI/Explore/ML-Concepts/Intro/KNN/Data/car.data', header=None)
df_columns = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "class"]
df.columns = df_columns

enc = OrdinalEncoder()
df[df_columns] = enc.fit_transform(df[df_columns])
print(df.head())

#Option 5: Automatically detect numeric columns and exlude them
print('Automatically detect numeric columns and exlude them')
df = pd.read_csv('~/Dev/ML-AI/Explore/ML-Concepts/Intro/KNN/Data/car.data', header=None)
df_columns = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "class"]
df.columns = df_columns


categorical_cols = df.select_dtypes(exclude=['number']).columns

# Apply LabelEncoder to each categorical column
le = LabelEncoder()
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])
print(df.head())