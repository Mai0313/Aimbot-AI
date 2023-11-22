import os
from typing import Optional, Union

import rootutils
from omegaconf import OmegaConf
from pydantic import BaseModel, Field, model_validator
from rich.console import Console
from ultralytics import YOLO

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

console = Console()


class PoseDetectionTrainer(BaseModel):
    yolov8_model_weights: str = Field(..., pattern=r".*\.pt$", frozen=True)

    yolov8_model_config: str = Field(..., pattern=r".*\.yaml$", frozen=True)
    yolov8_data_config: str = Field(..., pattern=r".*\.yaml$", frozen=True)

    project: Optional[str] = Field(default="project_exp")
    name: Optional[str] = Field(default="exp")
    optimizer: Optional[str] = Field(default="auto")
    dropout: Optional[float] = Field(default=0.0, ge=0.0, le=1.0)
    batch: Optional[int] = Field(default=16, ge=1)
    save_period: Optional[int] = Field(default=-1)
    device: Optional[list] = Field(default=["0"])
    epochs: Optional[int] = Field(default=10, ge=1)

    @model_validator(mode="before")
    def get_config(cls, values):
        YOLO(values["yolov8_model_config"])
        YOLO(values["yolov8_model_weights"])
        return values

    def train(self):
        console.log("Start training...")
        model = YOLO(self.yolov8_model_config).load(self.yolov8_model_weights)
        results = model.train(data=self.yolov8_data_config, epochs=self.epochs, imgsz=640)
        return f"./{results.save_dir}/weights/best.pt"


class PoseDetectionEval(PoseDetectionTrainer):
    best_model_path: str = Field(..., pattern=r".*\.pt$", frozen=True)

    def val(self):
        console.log(f"Loading Model from {self.best_model_path}...")
        console.log("Start evaluation...")
        model = YOLO(self.yolov8_model_weights)
        model = YOLO(self.best_model_path)
        metrics = model.val()
        metrics.box.map
        metrics.box.map50
        metrics.box.map75
        metrics.box.maps
        return metrics


if __name__ == "__main__":
    config = OmegaConf.load("./configs/experiments/md1.yaml")
    yolov8_model_weights = config.model.yolov8_model_weights

    yolov8_model_config = config.model.yolov8_model_config
    yolov8_data_config = config.model.yolov8_data_config

    project = config.trainer.project
    name = config.trainer.name
    optimizer = config.trainer.optimizer
    batch = config.trainer.batch
    save_period = config.trainer.save_period
    device = config.trainer.device
    epochs = config.trainer.epochs

    pose_detection_trainer = PoseDetectionTrainer(
        yolov8_model_weights=yolov8_model_weights,
        yolov8_model_config=yolov8_model_config,
        yolov8_data_config=yolov8_data_config,
        project=project,
        name=name,
        optimizer=optimizer,
        batch=batch,
        save_period=save_period,
        device=device,
        epochs=epochs,
    )
    best_model_path = pose_detection_trainer.train()

    pose_detection_eval = PoseDetectionEval(
        yolov8_model_weights=yolov8_model_weights,
        yolov8_model_config=yolov8_model_config,
        yolov8_data_config=yolov8_data_config,
        best_model_path=best_model_path,
    )
    pose_detection_eval.val()
