from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# 載入資料
digits = load_digits()
X, y = digits.data, digits.target

# 分割資料
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# 建立模型
clf = KNeighborsClassifier()
clf.fit(X_train, y_train)

# 預測與評估
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
