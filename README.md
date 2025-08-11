# 🎵 Spotify Songs Popularity Predictor

Predict a Spotify track’s **popularity** from its audio features using classic ML models — plus an interactive demo.

---

## 📂 What’s Inside

* 📓 **spotify-songs-popularity.ipynb** – EDA, model comparison (📈 Linear Regression, 🔗 Elastic Net, 📊 SVR, 🌳 Decision Tree, 🌲 Random Forest), clustering (🎯 K-Means, 🌀 BIRCH), evaluation visuals.
* 🛠 **train\_model.py** – Preprocesses data, trains models, saves the best as `spotify_model.joblib`.
* 💻 **App.py** – Script/UI for loading the trained model & making predictions.
* 📄 **spotify\_songs.csv** – Dataset.
* 💾 **spotify\_model.joblib** – Pre-trained model for instant inference.

---

## ⚡ Quick Start

```bash
git clone https://github.com/waseemkathia/spotify-songs-popularity.git
cd spotify-songs-popularity
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

* 📊 **Explore & analyze:**
  `jupyter notebook spotify-songs-popularity.ipynb`
* 🔄 **Retrain model:**
  `python train_model.py`
* 🎯 **Predict:**
  `python App.py`

---

## ✨ Features

* 🎼 Predict popularity from features like **danceability**, **energy**, **acousticness**, **tempo**, **valence**.
* 📈 Compare regression models & visualize performance.
* 🎯 Discover clusters of similar tracks.
* ⚙️ Use the trained model instantly for predictions.

---

## 👨‍💻 About the Creator

Created by **Muhammad Waseem Sabir** — a data science enthusiast passionate about turning raw data into actionable insights.

* 💼 Loves **machine learning, data visualization & analytics**
* 📬 Contact: [GitHub Profile](https://github.com/waseemkathia)

---

