import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Генеруємо дані з шумом
np.random.seed(42)
data = 100 + np.random.randn(1000) * 5  # постійне значення з нормальним шумом

# Додаємо аномалії
data[::50] += np.random.randn(20) * 50  # великі аномалії


# Очищуємо дані від аномалій за допомогою IQR
def clean_data(data):
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    cleaned_data = data[(data >= lower_bound) & (data <= upper_bound)]
    return cleaned_data, lower_bound, upper_bound


cleaned_data, lower_bound, upper_bound = clean_data(data)

# Поліноміальна регресія (ступінь 2)
poly_features = PolynomialFeatures(degree=2)
X_poly = poly_features.fit_transform(np.arange(len(cleaned_data)).reshape(-1, 1))
model = LinearRegression().fit(X_poly, cleaned_data)


# Передбачення та екстраполяція
def extrapolate(model, poly_features, data, factor=0.5):
    max_index = len(data)
    new_index = int(max_index + max_index * factor)
    new_data = np.linspace(0, new_index, new_index).reshape(-1, 1)
    new_data_poly = poly_features.transform(new_data)
    return model.predict(new_data_poly)


predicted_data = extrapolate(model, poly_features, cleaned_data)

# Візуалізуємо результати поліноміальної регресії
plt.figure(figsize=(10, 6))
plt.plot(np.arange(len(cleaned_data)), cleaned_data, label="Очищені Дані")
plt.plot(predicted_data, label="Передбачені Дані")
plt.legend()
plt.show()
