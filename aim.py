import random
from typing import Optional

from omegaconf import OmegaConf
from pydantic import BaseModel, Field

# from src.click import MouseController, ScreenShot
from src.pred import PoseDetectionPredict


class BodyDetection(BaseModel):
    config_path: Optional[str] = Field(default="./configs/experiments/md1.yaml")

    def get_skeleton(self):
        config = OmegaConf.load(self.config_path)
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
        if len(boxes) != len(skeleton):
            raise ValueError("An error occurred in the number of people.")
        return skeleton


class AimBot(BaseModel):
    pass


if __name__ == "__main__":
    config_path = "./configs/experiments/md1.yaml"
    skeleton = BodyDetection(config_path=config_path).get_skeleton()

    user_setting = "./settings/user.yaml"
    user_setting = OmegaConf.load(user_setting)
    for human_no, human_skeleton in enumerate(skeleton):
        head = human_skeleton[:3]
        upper_body = human_skeleton[3:9]
        lower_body = human_skeleton[9:]
        aim_position = random.choice(head)
        print(aim_position)
