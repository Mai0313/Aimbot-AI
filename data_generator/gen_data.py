import datetime
import hashlib
import os
import time
from typing import Optional, Union

import pyautogui
from pydantic import BaseModel, Field, model_validator
from rich.console import Console

pyautogui.PAUSE = 1
pyautogui.FAILSAFE = True

console = Console()


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


if __name__ == "__main__":
    while True:
        output_path_folder = "outputs"
        capture_percent = 1
        ScreenShot(
            output_path_folder=output_path_folder, capture_percent=capture_percent
        ).take_screenshot()
        console.log(
            f"Screenshot saved to {output_path_folder} at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        time.sleep(1)
