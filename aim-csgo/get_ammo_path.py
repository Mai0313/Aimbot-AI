import os

import pandas as pd
from pydantic import BaseModel, Field, model_validator


class SelectWeapon(BaseModel):
    selected_weapon: str = Field(..., description="Select weapon to export ammo path")
    weapons: list

    @model_validator(mode="before")
    def check_selected_weapon(cls, values):
        if values["selected_weapon"] not in weapons:
            raise ValueError(f"selected_weapon must be one of these: {weapons}")
        return values

    def get_ammo_path(self):
        weapon_path = f"{ammo_path}/{self.selected_weapon}.csv"
        ammo_data = pd.read_csv(weapon_path, header=None, names=["x", "y", "z"])
        return ammo_data


if __name__ == "__main__":
    ammo_path = "./aim-csgo/ammo_path"
    weapons = [f for f in os.listdir(ammo_path) if f.endswith(".csv")]
    weapons = [f.replace(".csv", "") for f in weapons]
    select_weapon = SelectWeapon(selected_weapon="ak47", weapons=weapons)
    data = select_weapon.get_ammo_path()
