## 📉 Customer Churn Prediction (Data Preprocessing & Neural Network)

A deep learning app that predicts telecom customer churn with a strong focus on data cleaning, feature engineering, and reproducible results.

⸻

## 🌐 Live Demo
	•	App not deployed yet, but Flask app code is provided for future deployment.

⸻

## 🚀 Features
	•	Reads and cleans customer data: removes missing values and fixes data types.
	•	Encodes categorical columns and scales numeric ones using MinMaxScaler.
	•	Engineers robust feature pipelines for model-ready data.
	•	Trains and tests a neural network achieving 79% test accuracy.
	•	Includes app code for easy data input and churn prediction visualization.

⸻

## 🧠 Tech Stack
	•	Programming Language: Python
	•	Libraries: Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn
	•	Deep Learning: TensorFlow, Keras (Sequential Neural Network)
	•	App Framework: Flask
	•	Development Tools: Jupyter Notebook, VS Code

⸻

## 📊 Dataset
	•	Source: Telco Customer Churn dataset (WA_Fn-UseC_-Telco-Customer-Churn.csv)
	•	Label: “Churn” (1 = Yes, 0 = No)
	•	Size: 7,000+ records with contract, charges, and usage details

⸻

## 🛠️ How It Works
	•	Data Loading & Preprocessing: Loads CSV, drops missing values, and corrects data types.
	•	Feature Engineering: Encodes binary columns (0/1), one-hot encodes service and contract details.
	•	Scaling: Scales tenure, MonthlyCharges, and TotalCharges using MinMaxScaler.
	•	Model: Splits data (80% train, 20% test), trains a neural network (2 hidden layers + dropout), and achieves 79% accuracy.
	•	App Interface: Flask app preprocesses user input like training data and outputs churn predictions.

⸻

## 📦 Files Included
	•	WA_Fn-UseC_-Telco-Customer-Churn.csv — Main dataset.
	•	test.csv — Sample test inputs.
	•	CustomerChurnPrediction.ipynb — Analysis, preprocessing, and modeling notebook.
	•	app.py — Flask web app ready for deployment.
	•	requirements.txt — Python dependencies.
