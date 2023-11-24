import datetime
import hashlib
import os
import time
from typing import Optional

import pyautogui
from rich.console import Console

pyautogui.PAUSE = 1
pyautogui.FAILSAFE = True

console = Console()


class ScreenShot:
    def __init__(self, output_path_folder, capture_percent):
        self.output_path_folder: str = output_path_folder
        self.capture_percent: Optional[float] = capture_percent
        os.makedirs(self.output_path_folder, exist_ok=True)

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
        capture_percent = 0.5
        ScreenShot(
            output_path_folder=output_path_folder, capture_percent=capture_percent
        ).take_screenshot()
        console.log(
            f"Screenshot saved to {output_path_folder} at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        time.sleep(0.5)
