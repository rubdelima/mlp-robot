import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from ga import GeneticAlgorithm

df = pd.read_csv('simulated_data.csv')
df = df.dropna().reset_index(drop=True)

output_delay=2

X_raw = df[['axis0', 'axis1', 'axis2', 'axis3']].values  # 4D input
Y_raw = df[['x_final', 'y_final']].values  # 2D output

X = df[['axis0', 'axis1', 'axis2', 'axis3']].values  # 4D input
Y = df[['x_final', 'y_final']].values  # 2D output

X_new = []
Y_new = []
for t in range(output_delay, len(X_raw)):
    current_input = X_raw[t]
    delayed_outputs = Y_raw[t-output_delay:t].flatten()
    combined_input = np.concatenate([current_input, delayed_outputs])
    X_new.append(combined_input)
    Y_new.append(Y_raw[t])
    
X_new = np.array(X_new)
Y_new = np.array(Y_new)

X_train, X_test, Y_train, Y_test = train_test_split(
    X_new, Y_new, test_size=0.2, random_state=42
)

input_scaler = StandardScaler()
X_train_norm = input_scaler.fit_transform(X_train)
X_test_norm = input_scaler.transform(X_test)

output_scaler = StandardScaler()
Y_train_norm = output_scaler.fit_transform(Y_train)
Y_test_norm = output_scaler.transform(Y_test)

X_tensor = torch.tensor(X_train_norm, dtype=torch.float32)
Y_tensor = torch.tensor(Y_train_norm, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_norm, dtype=torch.float32)
Y_test_tensor = torch.tensor(Y_test_norm, dtype=torch.float32)

input_dim = X_raw.shape[1] + output_delay * Y_raw.shape[1]

ga = GeneticAlgorithm(population_size=100, mutation_rate=0.2, crossover_rate=0.5, input_dim=input_dim, X_tensor=X_tensor, Y_tensor=Y_tensor, tournament_size=3)
best_model = ga.run(generations=3000)

with torch.no_grad():
    predictions = best_model(X_test_tensor).numpy()

predictions_original = output_scaler.inverse_transform(predictions)
Y_test_original = output_scaler.inverse_transform(Y_test_norm)

mse = mean_squared_error(Y_test_original, predictions_original)
rmse = np.sqrt(mse)
mae = mean_absolute_error(Y_test_original, predictions_original)
r2 = r2_score(Y_test_original, predictions_original)

print("Evaluation Metrics on Test Data:")
print(f"MSE: {mse:.5f}")
print(f"RMSE: {rmse:.5f}")
print(f"MAE: {mae:.5f}")
print(f"R^2: {r2:.5f}")

step = 20
time_steps = np.arange(0, len(Y_test_original), step)
plt.figure(figsize=(10, 5))
plt.plot(time_steps, Y_test_original[::step, 0], label='True x_final', alpha=0.7)
plt.plot(time_steps, predictions_original[::step, 0], label='Predicted x_final', alpha=0.7)
plt.xlabel("Time Step")
plt.ylabel("x_final Value")
plt.title("Time Series Comparison for x_final (Subsampled)")
plt.legend()
plt.show()

time_steps = np.arange(0, len(Y_test_original), step)
plt.figure(figsize=(10, 5))
plt.plot(time_steps, Y_test_original[::step, 1], label='True y_final', alpha=0.7)
plt.plot(time_steps, predictions_original[::step, 1], label='Predicted y_final', alpha=0.7)
plt.title("Time Series Comparison for y_final")
plt.xlabel("Time Step")
plt.ylabel("y_final Value")
plt.legend()
plt.show()
