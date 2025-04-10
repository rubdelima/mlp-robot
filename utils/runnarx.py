import torch
import numpy as np
import joblib
from utils.narxwithga import NARXModel
import pandas as pd

# === Load scalers and model ===
input_scaler = joblib.load("models/input_scaler.pkl")
output_scaler = joblib.load("models/output_scaler.pkl")
X_test_norm = np.load("data/X_test_norm.npy")
input_dim = 24  # explicitly use 24 instead of loading from test shape

model = NARXModel(input_dim=input_dim, output_dim=2)
model.load_state_dict(torch.load("models/best_ga_model.pth"))
model.eval()

# === Config ===
input_delay = 4
output_delay = 4
angle_dim = 4
coord_dim = 2

df = pd.read_csv("data/train_data.csv").dropna()

columns = ['0.30','0.50','0.75','0.90','0.95','f']

angles = df[['t0', 't1', 't2', 't3']].values
coords = df[['xc', 'yc']].values

# Use first `input_delay` and `output_delay` for delayed histories
delayed_inputs = angles[:input_delay]     # shape: (4, 4)
delayed_outputs = coords[:output_delay]   # shape: (4, 2)

def run_narx(input_angles):
    """
    Predicts (x, y) coordinates from a batch of joint angle predictions (Tensor).
    input_angles: Tensor of shape (batch_size, 4)
    Returns: Tensor of shape (batch_size, 2)
    """
    if input_angles.requires_grad:
        input_angles = input_angles.detach()

    input_angles = input_angles.cpu().numpy()
    
    preds = []
    for angles in input_angles:
        # Shift histories
        global delayed_inputs, delayed_outputs
        delayed_inputs = np.roll(delayed_inputs, shift=-1, axis=0)
        delayed_outputs = np.roll(delayed_outputs, shift=-1, axis=0)

        delayed_inputs[-1] = angles

        flat_outputs = delayed_outputs.flatten()   # 4 * 2 = 8
        flat_inputs = delayed_inputs.flatten()     # 4 * 4 = 16

        narx_input = np.concatenate([flat_outputs, flat_inputs])  # 8 + 16 = 24

        narx_input_scaled = input_scaler.transform([narx_input])
        input_tensor = torch.tensor(narx_input_scaled, dtype=torch.float32)

        with torch.no_grad():
            pred_scaled = model(input_tensor).detach().numpy()
            pred_original = output_scaler.inverse_transform(pred_scaled)

        delayed_outputs[-1] = pred_original[0]
        preds.append(pred_original[0])

    return torch.tensor(preds, dtype=torch.float32)
