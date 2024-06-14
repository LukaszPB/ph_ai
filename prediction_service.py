from flask import Flask, request, jsonify
from flasgger import Swagger
import mlflow.pytorch
import torch

app = Flask(__name__)
swagger = Swagger(app)

model_name = "NeuralNetworkRegressionModel"
model_uri = "mlflow-artifacts:/499684513618114738/b27c7189585a4e82a176f12557870e2f/artifacts/model"
#model_uri = "mlflow-artifacts:/499684513618114738/c97e0910d14a4e0080cabcafc580f3ca/artifacts/model"
model = mlflow.pytorch.load_model(model_uri)

@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint to predict using Neural Network Regression Model.
    ---
    parameters:
      - name: body
        in: body
        required: true
        schema:
          type: array
          items:
            type: array
            items:
              type: number
    responses:
      200:
        description: Predictions from the model
    """
    input_data = request.json
    input_tensor = torch.tensor(input_data, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        predictions = model(input_tensor)
    return jsonify(predictions.numpy().tolist())

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)