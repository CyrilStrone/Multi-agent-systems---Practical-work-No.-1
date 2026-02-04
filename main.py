import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

import config


# Загрузка данных
df = pd.read_csv(config.DATA_PATH)

print("Первые 5 строк датасета:")
print(df.head())

print("\nИнформация о датасете:")
print(df.info())


# Матрица корреляции
correlation_matrix = df.corr(numeric_only=True)

print("\nМатрица корреляции:")
print(correlation_matrix)

# Сохраняем в файл
correlation_matrix.to_csv("correlation_matrix.csv")


# Визуализация корреляции (Рис. 1)
plt.figure(figsize=config.FIGURE_SIZE)
plt.matshow(correlation_matrix, fignum=1)
plt.colorbar()
plt.xticks(
    range(len(correlation_matrix.columns)),
    correlation_matrix.columns,
    rotation=90
)
plt.yticks(
    range(len(correlation_matrix.columns)),
    correlation_matrix.columns
)
plt.title("Correlation matrix", pad=20)
plt.show()


# Выбор наиболее коррелирующих РАЗНЫХ признаков
corr_pairs = (
    correlation_matrix
    .abs()
    .unstack()
    .reset_index()
)

corr_pairs.columns = ["feature_1", "feature_2", "correlation"]

# Убираем одинаковые признаки
corr_pairs = corr_pairs[corr_pairs["feature_1"] != corr_pairs["feature_2"]]

# Убираем дубли (A-B и B-A)
corr_pairs["pair"] = corr_pairs.apply(
    lambda row: "-".join(sorted([row["feature_1"], row["feature_2"]])),
    axis=1
)
corr_pairs = corr_pairs.drop_duplicates(subset="pair")

# Берём пару с максимальной корреляцией
best_pair = corr_pairs.sort_values("correlation", ascending=False).iloc[0]

X_column = best_pair["feature_1"]
y_column = best_pair["feature_2"]

print("\nВыбранные признаки:")
print(f"Независимая переменная (X): {X_column}")
print(f"Зависимая переменная (y): {y_column}")


# Визуализация данных (Рис. 2)
plt.figure()
plt.scatter(df[X_column], df[y_column])
plt.xlabel(X_column)
plt.ylabel(y_column)
plt.title("Data distribution")
plt.show()


# Подготовка данных
X = df[[X_column]]   # DataFrame!
y = df[y_column]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=config.TEST_SIZE,
    random_state=config.RANDOM_STATE
)


# Обучение модели
model = LinearRegression()
model.fit(X_train, y_train)


# Параметры модели ---
print("\nПараметры линейной регрессии:")
print("Коэффициент наклона (k):", model.coef_[0])
print("Точка пересечения (b):", model.intercept_)


# Предсказание и ошибки
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("\nОшибки модели:")
print("MSE:", mse)
print("MAE:", mae)


# Реальные и предсказанные значения
results = pd.DataFrame({
    "Real values": y_test.values,
    "Predicted values": y_pred
})

print("\nПример результатов:")
print(results.head())


# Линия линейной регрессии (Рис. 3)
plt.figure()
plt.scatter(X, y, label="Data")
plt.plot(X, model.predict(X), color="red", label="Regression line")
plt.xlabel(X_column)
plt.ylabel(y_column)
plt.legend()
plt.title("Linear Regression")
plt.show()


# Предсказание по вводу пользователя
value = float(input(f"\nВведите значение признака '{X_column}': "))
input_df = pd.DataFrame({X_column: [value]})
prediction = model.predict(input_df)

print("Предсказанное значение зависимой переменной:", prediction[0])
