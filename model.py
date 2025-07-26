from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.pipeline import Pipeline, make_pipeline

import pandas as pd
import numpy as np


df = pd.read_csv("insurance.csv")

X = df.drop("charges", axis=1)
y = np.log1p(df["charges"])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

num_attribs = ["age", "bmi", "children"]
cat_attribs = ["region", "sex", "smoker"]


preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(), cat_attribs),
    ('num', StandardScaler(), num_attribs)
])

pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('reg', LinearRegression())
])

pipe.fit(X_train, y_train)

y_pred = pipe.predict(X_test)
scores = cross_val_score(pipe, X, y, cv=5, scoring='r2')
print(f"RMSE: {root_mean_squared_error(np.expm1(y_test), np.expm1(y_pred)):.2f}")
print(f"R²: {r2_score(y_test, y_pred):.2f}")
print(f"Cross validated R²: {scores.mean():.2f} and std deviation: {scores.std():.2f}")