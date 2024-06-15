import json
import os
from typing import Optional
from pydantic import BaseModel

class Athlete(BaseModel):
    Name: Optional[str]
    Weight_lbs: Optional[float]
    BF_Percent: Optional[float]
    Caloric_Multiplier: Optional[float]
    Extra_Caloric_Multiplier: Optional[float]
    extra_PSMF_multiplier: Optional[float]

def create_athlete(
    name: str, 
    weight_lbs: float, 
    bf_percent: Optional[float] = None,
    caloric_multiplier: Optional[float] = None,
    extra_caloric_multiplier: Optional[float] = None,
    extra_psmf_multiplier: Optional[float] = None,
    file_path: str = './athletes.json'
) -> None:
    # Ensure the directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Load existing athletes
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            athletes = json.load(file)
    else:
        athletes = {}

    # Check if the athlete already exists
    if name in athletes:
        print(f"Overwriting existing entry for {name}.")
    else:
        print(f"Creating new entry for {name}.")

    # Create or update the athlete entry
    athlete = Athlete(
        Name=name,
        Weight_lbs=weight_lbs,
        BF_Percent=bf_percent,
        Caloric_Multiplier=caloric_multiplier,
        Extra_Caloric_Multiplier=extra_caloric_multiplier,
        extra_PSMF_multiplier=extra_psmf_multiplier
    )

    athletes[name] = athlete.dict()

    # Save back to the JSON file
    with open(file_path, 'w') as file:
        json.dump(athletes, file, indent=4)

    print(f"Athlete {name} saved successfully.")

def get_athlete(name: str, file_path: str = './athletes.json') -> Optional[Athlete]:
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            athletes = json.load(file)
        if name in athletes:
            return Athlete(**athletes[name])
    return None
