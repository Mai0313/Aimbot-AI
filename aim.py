import random
import shutil
from typing import Optional

from omegaconf import OmegaConf
from pydantic import BaseModel, Field
from rich.console import Console

from src.click import MouseController, ScreenShot
from src.pred import PoseDetectionPredict

console = Console()


class BodyDetection(BaseModel):
    def get_skeleton(self, predict_image_folder: str):
        yolov8_model_weights = "./pretrained/yolov8x-pose-p6.pt"

        best_model_path = None

        save_prediction = False

        pose_detection_eval = PoseDetectionPredict(
            yolov8_model_weights=yolov8_model_weights,
            best_model_path=best_model_path,
            save_prediction=save_prediction,
            predict_image_folder=predict_image_folder,
        )
        boxes, masks, probs, skeleton = pose_detection_eval.predict(loggers=False)
        if len(boxes) != len(skeleton):
            raise ValueError("An error occurred in the number of people.")
        return skeleton


class ScreenShotTaker(ScreenShot):
    def get_screenshots(self):
        capture_percent = 1.0
        ScreenShot(
            output_path_folder=self.output_path_folder, capture_percent=capture_percent
        ).take_screenshot()
        return self.output_path_folder


class AimBot(MouseController):
    def trigger(self):
        self.move_mouse()
        self.right_click()


if __name__ == "__main__":
    user_setting = "./settings/user.yaml"
    user_setting = OmegaConf.load(user_setting)

    predict_image_folder = "./datasets/test_screen.png"
    output_path_folder = "./outputs"
    output_path_folder = ScreenShotTaker(output_path_folder=output_path_folder).get_screenshots()
    skeleton = BodyDetection().get_skeleton(predict_image_folder=output_path_folder)
    shutil.rmtree(output_path_folder)

    for human_no, human_skeleton in enumerate(skeleton):
        head = human_skeleton[1:3]
        upper_body = human_skeleton[4:7]
        lower_body = human_skeleton[9:]
        aim_position = random.choice(upper_body)
        # console.log(aim_position)
        AimBot(position = aim_position).trigger()
        # # break
