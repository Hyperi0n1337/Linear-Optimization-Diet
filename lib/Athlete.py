import json
from typing import Dict, List, Optional
from rich import print

from pydantic import BaseModel

class Athlete(BaseModel):

    Weight_lbs: Optional[float]
    BF_Percent: Optional[float]
    LBM: Optional[float]

    Caloric_Multiplier: Optional[float]
    Extra_Caloric_Multiplier: Optional[float]

    total_calories: Optional[float]
    protein_minimum: Optional[float]
    fat_minimum: Optional[float]
    fat_maximum: Optional[float]
    net_carb_minimum: Optional[float]


def main():
    with open("./test_files/sample_athlete.json", "r") as samplefile:
        athlete = json.load(samplefile)
        try:
            a = Athlete(**athlete)
            print(a)
        except:
            print(f"couldn't parse valid athlete from {samplefile}")


if __name__ == "__main__":
    main()
