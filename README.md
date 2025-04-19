
![image](https://github.com/user-attachments/assets/b39964cf-c3ad-4e4e-bf89-3b2500030a28)
**Car Price Prediction** 
This is a full-stack machine learning web application that predicts the selling price of a used car based on various features such as year, body type, transmission, state, condition, odometer reading, and MMR (Manheim Market Report value).

🔍 Overview
The goal of this project is to build a predictive model using real-world car sales data and serve it through an interactive web interface. The system uses:

📊 XGBoost for powerful and efficient regression modeling

🧹 Scikit-learn pipelines for preprocessing (OneHotEncoding categorical data, etc.)

🧠 Flask to expose the ML model as an API

⚙️ Express.js (Node.js) as an API gateway

🌐 Next.js (React) frontend for a responsive user interface

📦 Features
Predicts car prices based on key features

Clean and intuitive UI for user input

Full-stack integration (frontend + backend + ML model)

Modular code structure

Input validation and error handling

📁 Tech Stack

Layer	Technology
Frontend	Next.js (React)
Backend API	Express.js (Node.js)
ML API	Flask + Scikit-learn + XGBoost
Model	Trained on CSV car price data
Data Prep	Pandas, Scikit-learn pipelines
🧪 Input Features
year: Manufacturing year of the vehicle

body: Body type (e.g., SUV, sedan, etc.)

transmission: Transmission type (e.g., automatic)

state: U.S. state abbreviation

condition: Numerical vehicle condition (e.g., 1–5 or 1–10 scale)

odometer: Odometer reading in miles

mmr: Manheim Market Report value of the car

🛠️ How It Works
The user enters vehicle details into the frontend.

The React frontend sends the data to the Node.js API.

Node.js forwards the request to the Flask backend.

The Flask API processes the input, runs the model, and returns a predicted price.

The prediction is displayed in the UI.


