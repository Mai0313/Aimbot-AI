import random
import shutil
from typing import Optional

from omegaconf import OmegaConf
from pydantic import BaseModel, Field
from pynput import mouse
from rich.console import Console

from src.click import MouseController, MouseListener, ScreenShot
from src.pred import PoseDetectionPredict

console = Console()


class BodyDetection(BaseModel):
    def get_skeleton(self, predict_image_folder: str):
        yolov8_model_weights = "./pretrained/yolov8x-pose-p6.pt"

        best_model_path = None

        save_prediction = True

        pose_detection_eval = PoseDetectionPredict(
            yolov8_model_weights=yolov8_model_weights,
            best_model_path=best_model_path,
            save_prediction=save_prediction,
            predict_image_folder=predict_image_folder,
        )
        boxes, masks, probs, skeleton = pose_detection_eval.predict(loggers=False)
        return skeleton


aimbot = BodyDetection()
console.log("Model has been loaded.")


class AimbotUtils(ScreenShot):
    def get_screenshots(self):
        capture_percent = 1.0
        ScreenShot(
            output_path_folder=self.output_path_folder, capture_percent=capture_percent
        ).take_screenshot()
        return self.output_path_folder

    def start_mouse_listener(self):
        def on_click(x, y, button, pressed):
            if button == mouse.Button.left and pressed:
                image_output_folder = self.get_screenshots()
                skeleton = aimbot.get_skeleton(predict_image_folder=image_output_folder)
                shutil.rmtree(output_path_folder)
                skeleton = [skeleton[0]]
                for human_no, human_skeleton in enumerate(skeleton):
                    upper_body = human_skeleton[4:7]
                    aim_position = random.choice(upper_body)
                    AimBot(position=aim_position).trigger()
                    return

        with mouse.Listener(on_click=on_click) as listener:
            listener.join()


class AimBot(MouseController):
    def trigger(self):
        self.move_mouse()
        self.click()


if __name__ == "__main__":
    user_setting = "./settings/user.yaml"
    user_setting = OmegaConf.load(user_setting)

    predict_image_folder = "./datasets/test_screen.png"
    output_path_folder = "./outputs"
    screenshot_taker = AimbotUtils(output_path_folder=output_path_folder)
    screenshot_taker.start_mouse_listener()
    # aimbot = BodyDetection()
    # skeleton = aimbot.get_skeleton(predict_image_folder=output_path_folder)
    # shutil.rmtree(output_path_folder)

    # for human_no, human_skeleton in enumerate(skeleton):
    #     head = human_skeleton[1:3]
    #     upper_body = human_skeleton[4:7]
    #     lower_body = human_skeleton[9:]
    #     aim_position = random.choice(upper_body)
    #     AimBot(position = aim_position).trigger()
