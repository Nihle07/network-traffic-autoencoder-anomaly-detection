import torch

import random

SEED = 42
random.seed(SEED)

torch.manual_seed(SEED)

import torch.nn as nn


class Autoencoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(10, 6),
            nn.ReLU(),
            nn.Linear(6, 3)
        )

        self.decoder = nn.Sequential(
            nn.Linear(3, 6),
            nn.ReLU(),
            nn.Linear(6, 10)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


model = Autoencoder()

print(model)



import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Veriyi oku
df = pd.read_csv("normal_features.csv")
X = df.drop(columns=["window_start"])

# Train / Validation
X_train, X_val = train_test_split(
    X,
    test_size=0.20,
    random_state=42
)

# Ölçeklendirme
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# PyTorch tensor'e çevir
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)

# Hata fonksiyonu ve optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Eğitim
epochs = 100

for epoch in range(epochs):
    model.train()

    output = model(X_train_tensor)
    loss = criterion(output, X_train_tensor)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs} - Loss: {loss.item():.6f}")



torch.save(model.state_dict(), "autoencoder_model.pth")
print("Model kaydedildi: autoencoder_model.pth")


import numpy as np

# Validation moduna geç
model.eval()

with torch.no_grad():
    reconstructed = model(X_val_tensor)

# Her validation örneğinin reconstruction error'u
errors = torch.mean(
    (X_val_tensor - reconstructed) ** 2,
    dim=1
).numpy()

# Normal validation hatalarının %95 persentili
threshold = np.percentile(errors, 95)

print("\nValidation Reconstruction Errors:")
print(errors)

print("\nThreshold:", threshold)


import joblib
import json

# Scaler'ı kaydet
joblib.dump(scaler, "scaler.pkl")

# Threshold ve feature sırasını kaydet
config = {
    "threshold": float(threshold),
    "features": list(X.columns)
}

with open("model_config.json", "w") as f:
    json.dump(config, f, indent=4)

print("Scaler kaydedildi: scaler.pkl")
print("Threshold kaydedildi: model_config.json")
