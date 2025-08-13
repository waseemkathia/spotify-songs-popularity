# 🎵 Spotify Songs Popularity Predictor

Predict a Spotify track’s **popularity** from its audio features using classic ML models — plus an interactive demo.

Try it now:

### [🌐 Launch the interactive app](https://checkifspotifyhit.streamlit.app/)

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
## 👨‍💻 Connect with Me

I am passionate about leveraging AI to solve real-world problems in healthcare, climate change and other emerging challenges. Please feel free to connect, ask questions, or discuss potential collaborations.

<p align="center">
  <a href="https://www.linkedin.com/in/waseemkathia/" target="_blank">
    <img src="https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white" alt="LinkedIn">
  </a>
  &nbsp;
  <a href="https://github.com/waseemkathia" target="_blank">
    <img src="https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white" alt="GitHub">
  </a>
  &nbsp;
  <a href="https://waseemkathia.github.io/" target="_blank">
    <img src="https://img.shields.io/badge/Website-4A90E2?style=for-the-badge&logo=blogger&logoColor=white" alt="Website">
  </a>
</p>
---

