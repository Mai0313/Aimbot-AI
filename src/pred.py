import os
from typing import Optional, Union

import rootutils
from omegaconf import OmegaConf
from pydantic import BaseModel, Field, model_validator
from rich.console import Console
from ultralytics import YOLO

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

console = Console()


class GetKeypoint(BaseModel):
    NOSE: int = 0
    LEFT_EYE: int = 1
    RIGHT_EYE: int = 2
    LEFT_EAR: int = 3
    RIGHT_EAR: int = 4
    LEFT_SHOULDER: int = 5
    RIGHT_SHOULDER: int = 6
    LEFT_ELBOW: int = 7
    RIGHT_ELBOW: int = 8
    LEFT_WRIST: int = 9
    RIGHT_WRIST: int = 10
    LEFT_HIP: int = 11
    RIGHT_HIP: int = 12
    LEFT_KNEE: int = 13
    RIGHT_KNEE: int = 14
    LEFT_ANKLE: int = 15
    RIGHT_ANKLE: int = 16


class PoseDetectionPredict(BaseModel):
    yolov8_model_weights: str = Field(..., pattern=r".*\.pt$", frozen=True)
    best_model_path: str = Field(..., pattern=r".*\.pt$", frozen=True)
    predict_image_path: Union[str, list]

    save_prediction: Optional[bool] = False

    @model_validator(mode="before")
    def get_predict_image_path(cls, values):
        if os.path.isdir(values["predict_image_path"]):
            predict_images = [f for f in os.listdir(values["predict_image_path"])]
            predict_images = [f"{values['predict_image_path']}/{f}" for f in predict_images]
            values["predict_image_path"] = predict_images
        return values

    def predict(self):
        console.log("Start prediction...")
        model = YOLO(self.yolov8_model_weights)
        model = YOLO(self.best_model_path)

        results = model.predict(
            self.predict_image_path, save=self.save_prediction, stream=True, conf=0.5
        )
        for result in results:
            # This part is not in used, but it is useful to know.
            boxes = result.boxes
            masks = result.masks
            keypoints = result.keypoints
            probs = result.probs
            skeleton = keypoints.xy
            skeleton = skeleton.cpu().numpy()
        return boxes, masks, probs, skeleton


if __name__ == "__main__":
    config = OmegaConf.load("./configs/experiments/md1.yaml")
    yolov8_model_weights = config.model.yolov8_model_weights

    best_model_path = config.model.yolov8_model_export
    predict_image_path = config.data.predict_image_path

    save_prediction = config.output_model.save_prediction

    pose_detection_eval = PoseDetectionPredict(
        yolov8_model_weights=yolov8_model_weights,
        best_model_path=best_model_path,
        predict_image_path=predict_image_path,
        save_prediction=save_prediction,
    )
    boxes, masks, probs, skeleton = pose_detection_eval.predict()
