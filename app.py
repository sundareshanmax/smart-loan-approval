
from flask import Flask, render_template, request, jsonify
import pickle

app = Flask(__name__)

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

feature_names = [
    "Gender", "Married", "Dependents", "Education",
    "Self_Employed", "ApplicantIncome", "CoapplicantIncome",
    "LoanAmount", "Loan_Amount_Term", "Credit_History", "Property_Area"
]

def explain_prediction(model, features):
    importances = model.feature_importances_
    impact = dict(zip(feature_names, importances))
    return sorted(impact.items(), key=lambda x: x[1], reverse=True)[:3]

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    probability = None
    explanation = None

    if request.method == "POST":
        features = [
            int(request.form["Gender"]),
            int(request.form["Married"]),
            int(request.form["Dependents"]),
            int(request.form["Education"]),
            int(request.form["Self_Employed"]),
            float(request.form["ApplicantIncome"]),
            float(request.form["CoapplicantIncome"]),
            float(request.form["LoanAmount"]),
            float(request.form["Loan_Amount_Term"]),
            float(request.form["Credit_History"]),
            int(request.form["Property_Area"])
        ]

        pred = model.predict([features])[0]
        prob = model.predict_proba([features])[0][1]

        prediction = "✅ Loan Approved" if pred == 1 else "❌ Loan Rejected"
        probability = round(prob * 100, 2)
        explanation = explain_prediction(model, features)

    return render_template("index.html",
                           prediction=prediction,
                           probability=probability,
                           explanation=explanation)

@app.route("/predict", methods=["POST"])
def predict_api():
    data = request.json
    features = [data[f] for f in feature_names]
    pred = int(model.predict([features])[0])
    prob = model.predict_proba([features])[0][1]
    return jsonify({
        "loan_status": pred,
        "approval_probability": round(prob * 100, 2)
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
