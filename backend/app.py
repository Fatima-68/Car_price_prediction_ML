from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)
model = joblib.load("best_model.pkl")

# Expected columns in correct order
EXPECTED_COLUMNS = ['year', 'body', 'transmission', 'state', 'condition', 'odometer', 'mmr']

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        print(" Incoming data:", data)

        # Clean up and reorder input
        input_data = {}
        for col in EXPECTED_COLUMNS:
            value = data.get(col, "")
            if isinstance(value, str):
                value = value.strip()  # remove whitespace and tabs
            input_data[col] = value

        df = pd.DataFrame([input_data])
        print(" Cleaned DataFrame:\n", df)
        print(" Data types:\n", df.dtypes)

        prediction = model.predict(df)[0]
        return jsonify({'predicted_price': float(prediction)})

    except Exception as e:
        print(" Error during prediction:", e)
        return jsonify({'error': str(e)}), 500



if __name__ == '__main__':
    app.run(port=5001)
