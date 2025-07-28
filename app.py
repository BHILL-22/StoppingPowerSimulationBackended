from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
import pickle
import logging
import sys
import os
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from stopping_power_ml.stopping.stopping_power import compute_stopping_power
from stopping_power_ml.cell import Cell

# --- Logging setup ---
logging.basicConfig(level=logging.INFO)

# --- FastAPI app ---
app = FastAPI()

# --- CORS middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Ensure stopping_power_ml is importable ---
sys.path.append(os.path.abspath("."))

# --- Load model ---
model = Sequential([
    Input(shape=(22,)),
    Dense(22, activation='relu'),
    Dense(32, activation='relu'),
    Dense(24, activation='relu'),
    Dense(8, activation='relu'),
    Dense(4, activation='relu'),
    Dense(3, activation='relu'),
    Dense(1)
])
model.load_weights("model.h5")

# --- Load patched featurizer and cell ---
import joblib
featurizers = joblib.load("featurizers_patched.pkl")


cell = pickle.load(open("al_starting_frame.pkl", "rb"))

# --- Input schema for prediction ---
class StoppingPowerInput(BaseModel):
    start_pos: List[float]  # fractional coordinates in conventional cell
    vdir: List[float]       # contravariant velocity direction in conventional cell
    vmag: float = 1.0       # velocity magnitude

# --- Input schema for raw featurization ---
class FeaturizerInput(BaseModel):
    position: List[float]  # Cartesian position
    velocity: List[float]  # Cartesian velocity

# --- Prediction endpoint ---
@app.post("/predict")
def predict_stopping_power(data: StoppingPowerInput):
    try:
        # Convert fractional to Cartesian
        start_pos_cart = cell.conv_strc.lattice.get_cartesian_coords(np.array(data.start_pos))
        vdir_cart = cell.conv_strc.lattice.get_cartesian_coords(np.array(data.vdir))
        velocity_vec = data.vmag * vdir_cart / np.linalg.norm(vdir_cart)

        # Featurize
        features = featurizers.featurize(start_pos_cart, velocity_vec)

        # Predict
        prediction = model.predict(np.array([features]), verbose=0)
        stopping_power = float(prediction[0]) if prediction.shape != () else float(prediction)

        logging.info(f"Prediction successful: {stopping_power}")
        return {"stopping_power": stopping_power}

    except Exception as e:
        logging.error(f"Prediction failed: {e}")
        return {"error": str(e)}

# --- Raw featurization endpoint ---
@app.post("/featurize")
def featurize_raw(data: FeaturizerInput):
    try:
        features = featurizers.featurize(data.position, data.velocity)
        logging.info(f"Featurization successful: {features}")
        return {"features": features}

    except Exception as e:
        logging.error(f"Featurization failed: {e}")
        return {"error": str(e)}









