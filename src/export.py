import os
from typing import Optional, Union

import rootutils
from omegaconf import OmegaConf
from pydantic import BaseModel, Field, model_validator
from rich.console import Console
from ultralytics import YOLO

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

console = Console()


class PoseDetectionExport(BaseModel):
    yolov8_model_weights: str = Field(..., pattern=r".*\.pt$", frozen=True)

    yolov8_model_export_format: list[str] = Field(
        ...,
        description="It can be a list of these: \n onnx, torchscript, openvino, engine, coreml, saved_model, pb, tflite, edgetpu, tfjs, paddle, ncnn",
    )

    @model_validator(mode="before")
    def check_model_format(cls, values):
        export_model_list = [
            "onnx",
            "torchscript",
            "openvino",
            "engine",
            "coreml",
            "saved_model",
            "pb",
            "tflite",
            "edgetpu",
            "tfjs",
            "paddle",
            "ncnn",
        ]
        if [f for f in values["yolov8_model_export_format"] if f not in export_model_list]:
            raise ValueError(
                f"yolov8_model_export_format must be one of these: {export_model_list}"
            )
        return values

    def export_model(self):
        for export_format in self.yolov8_model_export_format:
            console.log(f"Saving Model in {export_format}...")
            model = YOLO(self.yolov8_model_weights)
            model_output_dir = model.export(format=export_format)
            console.log(f"Model saved in {model_output_dir}")


if __name__ == "__main__":
    yolov8_model_weights = (
        "./pretrained/finetuned/csgo/yolov8s-csgo.pt"  # config.model.yolov8_model_weights
    )
    yolov8_model_export_format = [
        "onnx",
        "torchscript",
    ]  # config.output_model.yolov8_model_export_format

    pose_detection_export = PoseDetectionExport(
        yolov8_model_weights=yolov8_model_weights,
        yolov8_model_export_format=yolov8_model_export_format,
    )
    pose_detection_export.export_model()
