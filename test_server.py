import pytest
from fastapi.testclient import TestClient
from server import app

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"data": "Welcome to the Dyslexia Predictor API"}

def test_predict():
    data = {"data": [1.0, 7.0, 2.0, 2.0, 0.0, 2.0, 1.0, 0.0, 3.0, 3.0, 0.0, 3.0, 1.0, 8.0, 5.0, 3.0, 2.0, 3.0, 0.0, 6.0, 0.0, 4.0, 9.0, 1.0, 0.1, 0.1]}  # Example feature list; adjust according to your model's input
    response = client.post("/predict", json=data)
    assert response.status_code == 200
    json_response = response.json()
    assert "prediction" in json_response
    assert "message" in json_response
    assert json_response["prediction"] in [0, 1]  # Assuming binary classification
    assert json_response["message"] in ["Less likely to have dyslexia", "More likely to have dyslexia"]

if __name__ == "__main__":
    pytest.main()
