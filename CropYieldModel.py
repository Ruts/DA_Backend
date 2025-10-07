import os
import ast
import cv2
import torch
import shap
import pandas as pd
import numpy as np
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models import resnet18, ResNet18_Weights
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import requests
from io import BytesIO
from urllib.parse import urlparse
import joblib

# --- CONFIG ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = (128, 128)
CSV_PATH = "data/soil_data.csv"
MODEL_DIR = "saved_models"

# --- MODEL ---
class YieldPredictor(nn.Module):
    def __init__(self, soil_dim, img_dim=512):
        super().__init__()
        self.soil_net = nn.Sequential(
            nn.Linear(soil_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        self.fusion = nn.Sequential(
            nn.Linear(img_dim + 32, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, soil, img):
        soil_feat = self.soil_net(soil)
        combined = torch.cat((soil_feat, img), dim=1)
        return self.fusion(combined)

# --- IMAGE FEATURE EXTRACTION ---
def extract_image_features(image_paths):
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    model.fc = nn.Identity()
    model.eval().to(DEVICE)

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor()
    ])

    features = []
    for path in image_paths:
        # Check if path is a URL
        is_url = urlparse(path).scheme in ("http", "https")
        if is_url:
            try:
                response = requests.get(path, timeout=5)
                response.raise_for_status()
                img_array = np.asarray(bytearray(response.content), dtype=np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            except Exception as e:
                print(f"Failed to load image from URL {path}: {e}")
                continue
        else:
            img = cv2.imread(path)

        if img is None:
            print(f"Image not found or unreadable: {path}")
            continue

        img_tensor = transform(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            feat = model(img_tensor).cpu().numpy()
        features.append(feat[0])

    if not features:
        raise ValueError("No valid image features extracted.")
    return np.mean(features, axis=0)

# --- LOAD & PREPROCESS DATA ---
def load_dataset(csv_path):
    df = pd.read_csv(csv_path)
    print("df columns :\n", df.columns)
    df["image_paths"] = df["image_paths"].apply(ast.literal_eval)
    return df

def preprocess_soil(df):
    df.head()
    soil_features = df.drop(columns=["date", "crop_type", "yield_per_acre", "image_paths"])
    ct = ColumnTransformer([
        ("texture", OneHotEncoder(), ["soilTexture"]),
        ("num", StandardScaler(), soil_features.select_dtypes(include="number").columns.tolist())
    ])
    soil_array = ct.fit_transform(soil_features)
    return soil_array, ct

def train_models(df, soil_array, ct):
    os.makedirs(MODEL_DIR, exist_ok=True)
    models_by_crop = {}
    target_scalers = {}

    for crop in df["crop_type"].unique():
        crop_df = df[df["crop_type"] == crop].reset_index(drop=True)
        soil = torch.tensor(soil_array[crop_df.index], dtype=torch.float32).to(DEVICE)

        feats = np.array([extract_image_features(paths) for paths in crop_df["image_paths"]])
        img_feats = torch.from_numpy(feats).float().to(DEVICE)

        y_raw = crop_df["yield_per_acre"].values.reshape(-1, 1)
        y_scaler = StandardScaler()
        y_scaled = y_scaler.fit_transform(y_raw)
        y = torch.tensor(y_scaled, dtype=torch.float32).to(DEVICE)

        model = YieldPredictor(soil.shape[1]).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.MSELoss()

        for epoch in range(30):
            model.train()
            optimizer.zero_grad()
            pred = model(soil, img_feats)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()

        models_by_crop[crop] = model
        target_scalers[crop] = y_scaler
        joblib.dump(y_scaler, f"{MODEL_DIR}/{crop}_scaler.pkl")
        torch.save(model.state_dict(), f"{MODEL_DIR}/{crop}_model.pt")

    return models_by_crop, ct, target_scalers

def load_preprocessing(csv_path):
    df = pd.read_csv(csv_path)
    df.head()
    df["image_paths"] = df["image_paths"].apply(ast.literal_eval)
    soil_features = df.drop(columns=["date", "crop_type", "yield_per_acre", "image_paths"])
    ct = ColumnTransformer([
        ("texture", OneHotEncoder(), ["soilTexture"]),
        ("num", StandardScaler(), soil_features.select_dtypes(include="number").columns.tolist())
    ])
    soil_array = ct.fit_transform(soil_features)
    return ct, df, soil_array

# --- LOAD MODELS ---
def load_models(df, soil_array):
    models_by_crop = {}
    
    if hasattr(soil_array, "shape"):
        soil_dim = soil_array.shape[1]
    else:
        raise TypeError(f"Expected array-like input, got {type(soil_array)}: {soil_array}")

    for crop in df["crop_type"].unique():
        model = YieldPredictor(soil_dim).to(DEVICE)
        model.load_state_dict(torch.load(f"{MODEL_DIR}/{crop}_model.pt"))
        model.eval()
        models_by_crop[crop] = model
    return models_by_crop

def predict_yield(new_soil_dict, new_image_paths, crop_type, models_by_crop, ct, target_scalers):
    soil_df = pd.DataFrame([new_soil_dict])
    soil_array = ct.transform(soil_df)
    soil_tensor = torch.tensor(soil_array, dtype=torch.float32).to(DEVICE)

    img_feat = torch.tensor(extract_image_features(new_image_paths), dtype=torch.float32).unsqueeze(0).to(DEVICE)

    model = models_by_crop[crop_type]
    with torch.no_grad():
        pred_scaled = model(soil_tensor, img_feat).cpu().numpy()

    # Convert back to original yield scale
    yield_kg = target_scalers[crop_type].inverse_transform(pred_scaled)[0][0]
    return yield_kg

def recommend_soil_changes(model, new_soil_dict, ct):
    soil_df = pd.DataFrame([new_soil_dict])
    soil_array = ct.transform(soil_df)
    soil_np = np.array(soil_array)

    # Define wrapper that converts NumPy input to tensor
    def soil_model_wrapper(x_np):
        x_tensor = torch.tensor(x_np, dtype=torch.float32)
        with torch.no_grad():
            return model.soil_net(x_tensor).numpy()

    masker = shap.maskers.Independent(soil_np)
    explainer = shap.PermutationExplainer(soil_model_wrapper, masker)
    shap_values = explainer(soil_np)
    # shap.summary_plot(shap_values, soil_df)
    shap.summary_plot(shap_values[:, :-1], soil_df)


# --- MAIN EXECUTION ---
if __name__ == "__main__":
    df = load_dataset(CSV_PATH)
    soil_array, ct = preprocess_soil(df)

    # Train and save models if not already saved
    if not os.path.exists(f"{MODEL_DIR}/Maize_model.pt"):
        models_by_crop, ct, target_scalers = train_models(df, soil_array, ct)
    else:
        models_by_crop = load_models(df, soil_array)
        target_scalers = {}
        for crop in df["crop_type"].unique():
            scaler = joblib.load(f"{MODEL_DIR}/{crop}_scaler.pkl")
            target_scalers[crop] = scaler

    # Example prediction
    new_soil = {
        "pH": 6.5, "electricalConductivity": 1.75, "organicMatter": 4.6,
        "nitrogen": 71, "phosphorus": 21, "potassium": 43, "calcium": 2125,
        "magnesium": 145, "sulfur": 26, "zinc": 4.0, "iron": 5.7,
        "soilTexture": "Sandy Loam", "moistureContent": 27.1
    }
    new_images = ["data/images/maize_25_03_16_01.png", "data/images/maize_25_03_16_02.png"]
    predicted_yield = predict_yield(new_soil, new_images, "maize", models_by_crop, ct, target_scalers)
    print(f"Predicted Maize Yield: {predicted_yield:.2f} kg")

    # Optional: visualize soil impact
    # recommend_soil_changes(models_by_crop["maize"], new_soil, ct)
