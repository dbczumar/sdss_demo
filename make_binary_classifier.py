import argparse

import mlflow
import mlflow.pyfunc
from mlflow.pyfunc import PythonModel

parser = argparse.ArgumentParser()
parser.add_argument('--model-uri', '-m', type=str)
args = parser.parse_args()

class BinaryMNISTModel(PythonModel):

    def load_context(self, context):
        self.mnist_model = mlflow.pyfunc.load_pyfunc(
            context.artifacts["mnist-model"])

    def predict(self, context, model_input):
        import numpy as np
        digit_probabilities = self.mnist_model.predict(model_input)
        return [
            "even" if x % 2 == 0 else "odd"
            for x in np.argmax(digit_probabilities, axis=1)
        ]

with mlflow.start_run():
    mlflow.pyfunc.log_model(
        python_model=BinaryMNISTModel(),
        artifacts={
            "mnist-model": args.model_uri
        },
        artifact_path="binary-model",
    )
    run_id = mlflow.active_run().info.run_id

binary_classifier_uri = "runs:/{run_id}/binary-model".format(run_id=run_id)
print(binary_classifier_uri)

