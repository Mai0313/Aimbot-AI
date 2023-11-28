import os
from typing import Optional, Union

import rootutils
from omegaconf import OmegaConf
from PIL import Image
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
    yolov8_model_weights: str = Field(..., frozen=True)
    best_model_path: Union[str, None] = Field(..., frozen=True)
    predict_image_folder: str

    save_prediction: Optional[bool] = False

    def __loggers(self, results):
        human_detected, not_detected, total_pic = 0, 0, 0
        for result in results:
            boxes = result.boxes
            masks = result.masks
            keypoints = result.keypoints
            probs = result.probs
            skeleton = keypoints.xy
            skeleton = skeleton.cpu().numpy()
            total_pic += 1
            if len(boxes) >= 1:
                human_detected += 1
            elif len(boxes) == 0:
                not_detected += 1
        console.log(f"{human_detected} people detected")
        console.log(f"{not_detected} people not detected")
        console.log(f"{total_pic} total picture")
        console.log(f"{human_detected / total_pic * 100}% people detected")
        console.log(f"from {self.predict_image_folder.split('/')[-1]}")

        if os.path.isdir(self.predict_image_folder):
            with open(f"{self.predict_image_folder}/result.md", "w") as f:
                f.write(f"{human_detected / total_pic * 100}% people detected\n")
                f.write(f"{human_detected} people detected\n")
                f.write(f"{not_detected} people not detected\n")
                f.write(f"from {self.predict_image_folder.split('/')[-1]}")
        return boxes, masks, probs, skeleton

    def predict(self, loggers=True):
        console.log("Start prediction...")
        model = YOLO(self.yolov8_model_weights)
        if self.best_model_path:
            model = YOLO(self.best_model_path)

        results = model.predict(
            self.predict_image_folder, save=self.save_prediction, stream=True, conf=0.5
        )
        try:
            if loggers:
                boxes, masks, probs, skeleton = self.__loggers(results)
            return boxes, masks, probs, skeleton
        except Exception as e:
            return None, None, None, None


if __name__ == "__main__":
    yolov8_model_weights = "./pretrained/finetuned/csgo/yolov8m-csgo.pt"

    best_model_path = None  # This is finetune model
    predict_image_folder = "./datasets/test_images.png"  # config.data.predict_image_path

    save_prediction = True

    pose_detection_eval = PoseDetectionPredict(
        yolov8_model_weights=yolov8_model_weights,
        best_model_path=best_model_path,
        save_prediction=save_prediction,
        predict_image_folder=predict_image_folder,
    ).predict()
