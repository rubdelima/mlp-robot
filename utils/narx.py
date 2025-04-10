import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from narxwithga import NARXModel 
import random
from ga import GeneticAlgorithm
import joblib

# Load and clean data
df = pd.read_csv('data/train_data.csv')
df = df.dropna().reset_index(drop=True)

# Set delays
output_delay = 1
input_delay = 1

# Prepare raw input and output
X_raw = df[['t0', 't1', 't2', 't3']].values 
Y_raw = df[['x', 'y']].values  

# Build NARX-style delayed inputs
X_new = []
Y_new = []

for t in range(max(input_delay, output_delay), len(X_raw)):
    delayed_inputs = X_raw[t - input_delay:t].flatten()
    delayed_outputs = Y_raw[t - output_delay:t].flatten()
    current_input = X_raw[t]
    combined_input = np.concatenate([delayed_inputs, delayed_outputs, current_input])
    X_new.append(combined_input)
    Y_new.append(Y_raw[t])

X_new = np.array(X_new)
Y_new = np.array(Y_new)

# Train/test split
X_train, X_test, Y_train, Y_test = train_test_split(X_new, Y_new, test_size=0.25, random_state=42)

# Normalize inputs/outputs
input_scaler = StandardScaler()
X_train_norm = input_scaler.fit_transform(X_train)
X_test_norm = input_scaler.transform(X_test)

output_scaler = StandardScaler()
Y_train_norm = output_scaler.fit_transform(Y_train)
Y_test_norm = output_scaler.transform(Y_test)

np.save('X_test_norm.npy', X_test_norm)
np.save('Y_test_norm.npy', Y_test_norm)
np.save('Y_test.npy', Y_test)
joblib.dump(output_scaler, 'output_scaler.pkl')

# Convert to PyTorch tensors
X_tensor = torch.tensor(X_train_norm, dtype=torch.float32)
Y_tensor = torch.tensor(Y_train_norm, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_norm, dtype=torch.float32)
Y_test_tensor = torch.tensor(Y_test_norm, dtype=torch.float32)

# Determine input dimension
input_dim = input_delay * X_raw.shape[1] + output_delay * Y_raw.shape[1] + X_raw.shape[1]

# Run Genetic Algorithm
ga = GeneticAlgorithm(
    population_size=100,
    mutation_rate=0.2,
    crossover_rate=0.5,
    input_dim=input_dim,
    X_tensor=X_tensor,
    Y_tensor=Y_tensor,
    tournament_size=3
)


best_model = ga.run(generations=1000)
torch.save(best_model.state_dict(), 'best_ga_model.pth')
print("Model saved to 'best_ga_model.pth'")

with torch.no_grad():
    # Test predictions
    predictions = best_model(X_test_tensor).numpy()
    predictions_original = output_scaler.inverse_transform(predictions)
    Y_test_original = output_scaler.inverse_transform(Y_test_norm)

    # Train predictions
    preds_train_norm = best_model(X_tensor).numpy()
    preds_train = output_scaler.inverse_transform(preds_train_norm)
    Y_train_true = output_scaler.inverse_transform(Y_tensor.numpy())


# Train metrics
train_mse  = mean_squared_error(Y_train_true, preds_train)
train_rmse = np.sqrt(train_mse)
train_mae  = mean_absolute_error(Y_train_true, preds_train)
train_r2   = r2_score(Y_train_true, preds_train)

print("\nTrain set performance:")
print(f"MSE : {train_mse:.5f}")
print(f"RMSE: {train_rmse:.5f}")
print(f"MAE : {train_mae:.5f}")
print(f"R^2 : {train_r2:.5f}")


# Test metrics
mse = mean_squared_error(Y_test_original, predictions_original)
rmse = np.sqrt(mse)
mae = mean_absolute_error(Y_test_original, predictions_original)
r2 = r2_score(Y_test_original, predictions_original)

print("Evaluation Metrics on Test Data:")
print(f"MSE : {mse:.5f}")
print(f"RMSE: {rmse:.5f}")
print(f"MAE : {mae:.5f}")
print(f"R^2 : {r2:.5f}")


# --- Plotting ---

step = 1
time_steps = np.arange(0, len(Y_test_original), step)

plt.figure(figsize=(10, 5))
plt.plot(time_steps, Y_test_original[::step, 0], label='True xf', alpha=0.7)
plt.plot(time_steps, predictions_original[::step, 0], label='Predicted xf', alpha=0.7)
plt.xlabel("Time Step")
plt.ylabel("xf Value")
plt.title("Time Series Comparison for xf (Subsampled)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(time_steps, Y_test_original[::step, 1], label='True yf', alpha=0.7)
plt.plot(time_steps, predictions_original[::step, 1], label='Predicted yf', alpha=0.7)
plt.xlabel("Time Step")
plt.ylabel("yf Value")
plt.title("Time Series Comparison for yf (Subsampled)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

X_test_norm = np.load("X_test_norm.npy")
Y_test_norm = np.load("Y_test_norm.npy")  # assuming this is still normalized
Y_test = np.load("Y_test.npy")  # assuming this is still normalized
output_scaler = joblib.load('output_scaler.pkl')
input_dim = X_test_norm.shape[1]

X_test_tensor = torch.tensor(X_test_norm, dtype=torch.float32)


model = NARXModel(input_dim=input_dim, output_dim=2)  # output_dim=2 for (x, y)
model.load_state_dict(torch.load("best_ga_model.pth"))
model.eval()


# ---- Predict ----
max_delay = max(input_delay, output_delay)
rand_index = random.randint(max_delay, len(X_test_norm) - 1)

# Prepare delayed input
delayed_inputs = X_raw[rand_index - input_delay:rand_index].flatten()
delayed_outputs = Y_raw[rand_index - output_delay:rand_index].flatten()
current_input = X_raw[rand_index]
combined_input = np.concatenate([delayed_inputs, delayed_outputs, current_input])

# Normalize and predict
rand_test_index = random.randint(0, len(X_test_norm) - 1)

input_tensor = torch.tensor(X_test_norm[rand_test_index:rand_test_index+1], dtype=torch.float32)
true_output = Y_test[rand_test_index] 

# Predict
model.eval()
with torch.no_grad():
    pred_scaled = model(input_tensor).numpy()
    pred_original = output_scaler.inverse_transform(pred_scaled)

# Output prediction and ground truth
print(f"Predicted (x, y): {pred_original[0]}")
print(f"True (x, y): {true_output}")

