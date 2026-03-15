# ⚽ MatchPrediction_AI

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange)
![Streamlit](https://img.shields.io/badge/Web%20App-Streamlit-red)
![License](https://img.shields.io/badge/License-MIT-green)

A machine learning project that predicts football match outcomes using historical match data and team statistics.

---

# 📌 Project Overview

**MatchPrediction_AI** is a machine learning based football match prediction system that analyzes historical match data and team statistics to predict match outcomes.

The system uses a **Random Forest Classifier** trained on past match results to predict the probability of:

- Home Win  
- Draw  
- Away Win  

The project also includes a **Streamlit web application** where users can input match data and instantly receive predictions.

This project demonstrates a **complete machine learning workflow**, including:

- Data preprocessing  
- Feature engineering  
- Model training  
- Prediction system  
- Interactive web interface  

---

# 🚀 Features

### ⚽ Match Outcome Prediction
Predicts football match outcomes using machine learning.

### 🤖 Random Forest Model
Uses a Random Forest classifier trained on historical data.

### 📊 Probability Predictions
Provides probability scores for each possible match outcome.

### 🌐 Streamlit Web Application
Interactive interface to input match data and view predictions.

### 📈 Data Visualization
Visual analysis of team performance and trends.

### ⚡ Live Prediction Script
Quick command-line predictions using the trained model.

---

# 🧠 Machine Learning Model

The project uses a **Random Forest Classifier** from Scikit-learn.

### Model Workflow

1. Load historical match dataset  
2. Clean and preprocess data  
3. Perform feature engineering  
4. Train Random Forest model  
5. Save trained model  
6. Predict match outcomes

Random Forest was chosen because it:

- Handles tabular data well  
- Works effectively with non-linear relationships  
- Reduces overfitting using ensemble learning  

---

# 🧰 Tech Stack

| Component | Technology |
|-----------|------------|
| Programming Language | Python |
| Machine Learning | Scikit-learn |
| Data Processing | Pandas |
| Data Visualization | Matplotlib |
| Web Framework | Streamlit |
| API Handling | Requests |

---

# 📂 Project Structure

```
MatchPrediction_Ai/
│
├── data/                # Dataset used for training
├── models/              # Saved trained machine learning models
├── utils/               # Utility functions for preprocessing and prediction
│
├── main.py              # Model training pipeline
├── live_predictor.py    # Script for running predictions
├── webapp.py            # Streamlit web interface
│
├── requirements.txt     # Project dependencies
└── README.md            # Project documentation
```

---

# ⚙️ Installation & Setup

## 1️⃣ Clone the Repository

```bash
git clone https://github.com/NLZoro/MatchPrediction_Ai.git
cd MatchPrediction_Ai
```

---

## 2️⃣ Create Virtual Environment

```bash
python -m venv env
```

Activate the environment:

**Mac/Linux**
```bash
source env/bin/activate
```

**Windows**
```bash
env\Scripts\activate
```

---

## 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 🧠 Train the Model

Run the training pipeline:

```bash
python main.py
```

This will:

- Load and preprocess data  
- Train the Random Forest model  
- Save the trained model in the `models/` folder  

---

# 🌐 Run the Web Application

Launch the Streamlit app:

```bash
streamlit run webapp.py
```

Then open the provided local URL in your browser.

You can:

- Enter match data  
- View prediction probabilities  
- Analyze results  

---

# 🔮 Run Live Predictions

```bash
python live_predictor.py
```

---

# 📊 Example Prediction

```
Match: Team A vs Team B

Prediction:
Home Win Probability: 62%
Draw Probability: 21%
Away Win Probability: 17%
```

---

# 🎥 Demo

You can add screenshots or a demo GIF here later.

Example:

```
screenshots/
 ├── homepage.png
 ├── prediction.png
```

Then display them like this:

```markdown
![App Screenshot](screenshots/homepage.png)
```

---

# 🧪 Testing

Run tests using:

```bash
python -m unittest
```

---

# 🚀 Future Improvements

- Add advanced models (XGBoost, LightGBM)  
- Integrate live football APIs  
- Deploy using Docker and cloud platforms  
- Add multiple league support  
- Improve feature engineering  

---

# 👨‍💻 Author

**Ronit Gupta**

GitHub:  
https://github.com/NLZoro

---

# 📝 License

This project is licensed under the **MIT License**.
