import datetime
import hashlib
import os
from typing import Optional, Union

import pyautogui
from pydantic import BaseModel, Field, model_validator

pyautogui.PAUSE = 1
pyautogui.FAILSAFE = False


class ScreenShot(BaseModel):
    output_path_folder: str = Field(..., frozen=True)
    capture_percent: Optional[float] = Field(1.0, ge=0, le=1.0)

    @model_validator(mode="before")
    def check_output_path_folder(cls, values):
        os.makedirs(values["output_path_folder"], exist_ok=True)
        return values

    def take_screenshot(self):
        now = datetime.datetime.now()
        filename = hashlib.md5(str(now).encode()).hexdigest()

        screen_width, screen_height = pyautogui.size()
        capture_width = screen_width * self.capture_percent
        capture_height = screen_height * self.capture_percent
        start_x = (screen_width - capture_width) / 2
        start_y = (screen_height - capture_height) / 2

        screenshot = pyautogui.screenshot(
            region=(int(start_x), int(start_y), int(capture_width), int(capture_height))
        )
        screenshot.save(f"{self.output_path_folder}/{filename}.png")


class MouseListener(BaseModel):
    screenshotter: ScreenShot

    def on_click(self, x, y, button, pressed):
        if button == mouse.Button.right and pressed:
            self.screenshotter.take_screenshot()

    def start_listening(self):
        with mouse.Listener(on_click=self.on_click) as listener:
            listener.join()


class MouseController(BaseModel):
    position: list[Union[int, float]] = Field(..., min_items=2, max_items=2)
    # orig_shape: Optional[list[int]] = Field(default = [1080, 810])
    x: Optional[Union[int, float]] = Field(None, ge=0)
    y: Optional[Union[int, float]] = Field(None, ge=0)

    @model_validator(mode="before")
    def get_position(cls, values):
        # orig_width, orig_height = values.get("orig_shape")
        # screen_width, screen_height = pyautogui.size()
        # scale_width = screen_width / orig_width
        # scale_height = screen_height / orig_height

        values["x"], values["y"] = values["position"]

        # # TODO: Should we resize the image?
        # values["x"] = values["x"] * scale_width
        # values["y"] = values["y"] * scale_height
        return values

    def move_mouse(self):
        pyautogui.moveTo(self.x, self.y)

    def click(self):
        pyautogui.click(self.x, self.y)

    def double_click(self):
        pyautogui.doubleClick(self.x, self.y)

    def right_click(self):
        pyautogui.rightClick(self.x, self.y)


if __name__ == "__main__":
    output_path_folder = "outputs"
    capture_percent = 0.5
    ScreenShot(
        output_path_folder=output_path_folder, capture_percent=capture_percent
    ).take_screenshot()
    orig_shape = [1080, 810]
    position = [718.4456, 470.7785]
    mouse = MouseController(orig_shape=orig_shape, position=position)
    mouse.move_mouse()
