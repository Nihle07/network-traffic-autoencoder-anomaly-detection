import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("normal_features_final.csv")
X = df.drop(columns=["window_start"])

X_train, X_val = train_test_split(
    X,
    test_size=0.20,
    random_state=42
)

print("Train:", X_train.shape)
print("Validation:", X_val.shape)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

print("Scaled Train:", X_train_scaled.shape)
print("Scaled Validation:", X_val_scaled.shape)

print("\nİlk örnek:")
print(X_train_scaled[0])
