import numpy as np
import matplotlib.pyplot as plt


# Функція рекурентного згладжування Alpha-Beta фільтром
def alpha_beta_filter(data, alpha=0.85, beta=0.005, initial_value=0, initial_trend=0):
    est = initial_value
    trend = initial_trend
    estimates = []
    for measurement in data:
        # Оновлення оцінки та тренду
        est_prev = est
        est = est + trend + alpha * (measurement - (est + trend))
        trend = trend + beta * (measurement - (est_prev + trend))
        estimates.append(est)
    return np.array(estimates)


# Генеруємо дані з шумом (властивості зазначені у практичній роботі)
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

# Застосовуємо рекурентне згладжування
filtered_data = alpha_beta_filter(cleaned_data)

# Візуалізуємо результати
plt.figure(figsize=(10, 6))
plt.plot(cleaned_data, label="Очищені Дані")
plt.plot(filtered_data, label="Згладжені Дані (Alpha-Beta фільтр)")
plt.axhline(y=lower_bound, color="r", linestyle="--", label="Нижня межа")
plt.axhline(y=upper_bound, color="g", linestyle="--", label="Верхня межа")
plt.legend()
plt.show()

# Виводимо результати
print(f"Початкова кількість значень: {len(data)}")
print(f"Кількість значень після очищення: {len(cleaned_data)}")
print(f"Нижня межа для виявлення аномалій: {lower_bound}")
print(f"Верхня межа для виявлення аномалій: {upper_bound}")
