import requests
import json

# The URL where your FastAPI app is running
URL = "http://localhost:8000/predict"

# Fake customer data designed to trigger a churn prediction
# (High monthly charges, short tenure, month-to-month contract)
test_customer_features = {
    "gender": "Male",
    "tenure": 2,
    "MonthlyCharges": 95.50,
    "TotalCharges": 191.00,
    "InternetService": "Fiber optic",
    "Contract": "Month-to-month",
    "PaymentMethod": "Electronic check",
    "PaperlessBilling": "Yes",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "PhoneService": "Yes",
    "MultipleLines": "Yes",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "No"
}

print("Sending request to API...")
print("-" * 50)

# Send the POST request
response = requests.post(URL, json={"features": test_customer_features})

# Print the final result!
print("Status Code:", response.status_code)
print("Response Body:")
print(json.dumps(response.json(), indent=4))
