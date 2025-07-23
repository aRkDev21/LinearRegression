from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.pipeline import Pipeline, make_pipeline

import pandas as pd
import numpy as np


df = pd.read_csv("insurance.csv")
df["smoker"] = [1 if i == "yes" else 0 for i in df["smoker"].values]
df["sex"] = [1 if i == "male" else 0 for i in df["sex"].values]

X = df.drop("charges", axis=1)
y = np.log1p(df["charges"])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

num_attribs = list(X)
num_attribs.remove("region")
cat_attribs = ["region"]


preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(), cat_attribs),
    ('num', make_pipeline(
        PolynomialFeatures(degree=2, include_bias=False),
        StandardScaler()
    ), num_attribs)
])

pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('reg', LinearRegression())
])

pipe.fit(X_train, y_train)

y_pred = pipe.predict(X_test)
print(f"MSE: {mean_squared_error(y_test, y_pred):.2f}")
print(f"R²: {r2_score(y_test, y_pred):.2f}")