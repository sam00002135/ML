from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 載入資料
data = fetch_california_housing()
X, y = data.data, data.target

# 分割資料
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# 建立回歸模型
reg = LinearRegression()
reg.fit(X_train, y_train)

# 預測與評估
y_pred = reg.predict(X_test)
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
