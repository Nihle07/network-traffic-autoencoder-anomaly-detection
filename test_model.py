import torch
import torch.nn as nn
import pandas as pd
import joblib
import json
import matplotlib.pyplot as plt


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


# Kaydedilmiş scaler ve threshold
scaler = joblib.load("scaler.pkl")

with open("model_config.json", "r") as f:
    config = json.load(f)

threshold = config["threshold"]
features = config["features"]


# Eğitilmiş modeli yükle
model = Autoencoder()
model.load_state_dict(
    torch.load("autoencoder_model.pth")
)
model.eval()


def test_file(filename, real_class):

    df = pd.read_csv(filename)

    X = df[features]

    # Eğitimde kullanılan AYNI ölçeklendirme
    X_scaled = scaler.transform(X)

    X_tensor = torch.tensor(
        X_scaled,
        dtype=torch.float32
    )

    with torch.no_grad():
        reconstructed = model(X_tensor)

    errors = torch.mean(
        (X_tensor - reconstructed) ** 2,
        dim=1
    ).numpy()

    results = []

    for i, error in enumerate(errors):

        if error > threshold:
            prediction = "ANOMALI"
        else:
            prediction = "NORMAL"

        results.append({
            "pencere": i + 1,
            "gercek_sinif": real_class,
            "reconstruction_error": float(error),
            "threshold": threshold,
            "tahmin": prediction
        })

    return pd.DataFrame(results)


# NORMAL test
normal_results = test_file(
    "test_normal_features.csv",
    "NORMAL"
)

# ANOMALI test
anomaly_results = test_file(
    "test_anomaly_features.csv",
    "ANOMALI"
)

# Sonuçları birleştir
results = pd.concat(
    [normal_results, anomaly_results],
    ignore_index=True
)

# CSV olarak kaydet
results.to_csv(
    "test_results.csv",
    index=False
)

print("\nSONUCLAR")
print(results)

print("\nThreshold:", threshold)

print(
    "\nNormal doğru:",
    (
        (normal_results["tahmin"] == "NORMAL")
    ).sum()
)

print(
    "Normal yanlış alarm:",
    (
        (normal_results["tahmin"] == "ANOMALI")
    ).sum()
)

print(
    "Anomali doğru:",
    (
        (anomaly_results["tahmin"] == "ANOMALI")
    ).sum()
)


# --------------------------
# GRAFİK
# --------------------------

plt.figure(figsize=(12, 6))

plt.scatter(
    normal_results["pencere"],
    normal_results["reconstruction_error"],
    label="Normal Test"
)

# Anomaliyi normal pencerelerin sonrasına yerleştir
anomaly_x = (
    anomaly_results["pencere"]
    + len(normal_results)
)

plt.scatter(
    anomaly_x,
    anomaly_results["reconstruction_error"],
    label="Anomali Test"
)

# Threshold çizgisi
plt.axhline(
    y=threshold,
    linestyle="--",
    label=f"Threshold = {threshold:.4f}"
)

plt.xlabel("Test Penceresi")
plt.ylabel("Reconstruction Error")

plt.yscale("log")

plt.title(
    "Autoencoder Ağ Trafiği Anomali Tespiti"
)

plt.yscale("log")

plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    "reconstruction_error_graph_log.png",
    dpi=300
)

print("\nDosyalar oluşturuldu:")
print("test_results.csv")
print("reconstruction_error_graph.png")

plt.show()
