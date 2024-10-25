# Importy niezbędnych bibliotek
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Wczytanie danych
data_path = 'data/CollegeDistance.csv'
df = pd.read_csv(data_path)

# Eksploracja danych
print(df.info())
print(df.describe())

# Sprawdzenie brakujących wartości
missing_values = df.isnull().sum()
print("Missing values per column:\n", missing_values)

# Usunięcie brakujących wartości (lub imputacja, jeśli potrzebna)
df = df.dropna()

# Kodowanie zmiennych kategorycznych na zmienne numeryczne
df = pd.get_dummies(df, drop_first=True)

# Podział zmiennych niezależnych i zależnych (zakładamy, że 'score' to nasza zmienna zależna)
X = df.drop('score', axis=1)
y = df['score']

# Standaryzacja danych
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Podział danych na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Trenowanie modelu - regresja liniowa
model = LinearRegression()
model.fit(X_train, y_train)

# Predykcja na zbiorze testowym
y_pred = model.predict(X_test)

# Ocena modelu
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae}")
print(f"MSE: {mse}")
print(f"R²: {r2}")

# Tworzenie wykresu dla wyników predykcji
plt.scatter(y_test, y_pred)
plt.xlabel('Rzeczywiste wartości')
plt.ylabel('Przewidywane wartości')
plt.title('Rzeczywiste vs Przewidywane wartości')
plt.savefig('predictions_plot.png')  # Zapis wykresu

# Generowanie dokumentacji (README) z kodowaniem utf-8
with open('README.md', 'w', encoding='utf-8') as f:
    f.write(f"""
# Analizator wyników
Model predykcyjny został zbudowany przy użyciu regresji liniowej.

### Wyniki:
- MAE: {mae}
- MSE: {mse}
- R²: {r2}

Wykres został zapisany w pliku `predictions_plot.png`.
""")