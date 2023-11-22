import os

import numpy as np
import onnxruntime as ort
import rootutils
from PIL import Image
from pydantic import BaseModel, Field

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


class OnnxPredictor(BaseModel):
    onnx_model_path: str = Field(..., pattern=r".*\.onnx$", frozen=True)
    predict_image_path: str

    def _load_image(self, path: str):
        with Image.open(path) as img:
            img = img.resize((640, 640))
            img = np.array(img).astype(np.float32)
            img = img[:, :, :3]
            img /= 255.0
            img = img.transpose((2, 0, 1))[None, :]
            return img

    def predict(self):
        session = ort.InferenceSession(self.onnx_model_path)
        image = self._load_image(self.predict_image_path)
        inputs = {session.get_inputs()[0].name: image}
        outputs = session.run(None, inputs)
        return outputs


if __name__ == "__main__":
    # Init predictor
    predictor = OnnxPredictor(
        onnx_model_path="/home/ds906659/side-project/pose_detection/runs/pose/train/weights/best.onnx",
        predict_image_path="/home/ds906659/side-project/pose_detection/data/predict_test/test.png",
    )

    # Run prediction
    results = predictor.predict()
    print(results)
