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
    yolov8_model_export: str = Field(..., pattern=r".*\.pt$", frozen=True)

    yolov8_model_export_format: str = Field(
        default="onnx",
        description="It can be one of these: \n onnx, torchscript, openvino, engine, coreml, saved_model, pb, tflite, edgetpu, tfjs, paddle, ncnn",
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
        if values["yolov8_model_export_format"] not in export_model_list:
            raise ValueError(
                f"yolov8_model_export_format must be one of these: {export_model_list}"
            )
        return values

    def export_model(self):
        console.log(f"Saving Model in {self.yolov8_model_export_format}...")
        model = YOLO(self.yolov8_model_weights)
        model = YOLO(self.yolov8_model_export)
        model_output_dir = model.export(format=self.yolov8_model_export_format)
        console.log(f"Model saved in {model_output_dir}")


if __name__ == "__main__":
    config = OmegaConf.load("./configs/experiments/md1.yaml")
    yolov8_model_weights = config.model.yolov8_model_weights
    yolov8_model_export = config.model.yolov8_model_export
    yolov8_model_export_format = config.output_model.yolov8_model_export_format

    pose_detection_export = PoseDetectionExport(
        yolov8_model_weights=yolov8_model_weights,
        yolov8_model_export=yolov8_model_export,
        yolov8_model_export_format=yolov8_model_export_format,
    )
    pose_detection_export.export_model()
