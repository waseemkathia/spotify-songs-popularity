# train_model.py

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib # Library for saving and loading models

print("Starting model training process...")

# Load the dataset
df = pd.read_csv('spotify_songs.csv')

# --- Data Preparation ---
# Create the 'popular' column (same logic as before)
df['popular'] = np.where(df['track_popularity'] >= 60, 1, 0)

# Define features (X) and target (y)
features = ['acousticness', 'danceability', 'duration_ms', 'energy', 
            'instrumentalness', 'liveness', 'loudness', 'speechiness', 'tempo', 'valence']
X = df[features]
y = df['popular']

# --- Model Training ---
# Initialize and train the RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
print("Training the model... (This might take a moment)")
model.fit(X, y)
print("Model training complete.")

# --- Save the Trained Model ---
# This is the most important step. We are saving the trained 'model' object to a file.
joblib.dump(model, 'spotify_model.joblib')
print("Model saved successfully as 'spotify_model.joblib'")