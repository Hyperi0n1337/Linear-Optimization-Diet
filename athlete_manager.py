import json
import os
from typing import Optional, Tuple
from pydantic import BaseModel, ValidationError

# NOTE: It seems the following can be rolled into Athlete by having
# Athlete contain optional other classes,like  Requirements/Preferences
class Requirements(BaseModel):
    protein_minimum: Optional[float] # = LBM * lbm_protein_multiplier_table_for_PSMF[BF_Percent] * athlete.extra_PSMF_multiplier
    fiber_range: Optional[Tuple[int, int]]
    fat_range: Optional[Tuple[int, int]]
    net_carb_maximum: Optional[int]
    protein_quality_minimum: Optional[int]

# If the Requirements are optional, the current athlete dict -> Athlete
# parse will still work, and Requirements can be constructed separately
class Athlete(BaseModel):
    Name: Optional[str]
    Weight_lbs: Optional[float]
    BF_Percent: Optional[float]
    Caloric_Multiplier: Optional[float]
    Extra_Caloric_Multiplier: Optional[float]
    extra_PSMF_multiplier: Optional[float]
    # requirements: Optional[Requirements]

# TODO: Pass this, and create_athlete(), an athlete object and a valid filepath:
# def update_athlete(athlete: Athlete, athletes: Path) -> None
# and do validation elsewhere, e.g. change [2] in the notebook to be
#
# attrs = ['Markos', 194, 0.17, 15.1, 1, 1] ## NOTE: you could use a dict if you want to see labels
# athlete = Athlete(*attrs)
#
# csv_path = whatever is the right way to do this
#
# create_athlete(athlete, csv_path)
#
# Then the checks can be removed from these fns; pydantic will error if you create
# an athlete with invalid attrs immediately. You could also just do a bare
# get_athlete("markos"), since there's no need to create your own entry each time
def update_athlete(
    name: str,
    weight_lbs: Optional[float],
    bf_percent: Optional[float] = None,
    caloric_multiplier: Optional[float] = None,
    extra_caloric_multiplier: Optional[float] = None,
    extra_psmf_multiplier: Optional[float] = None,
    file_path: str = './athletes.json'
) -> Optional[Athlete]:

    try:
        try:
            with open(file_path, 'r') as file:
                athletes = json.load(file)
        except json.JSONDecodeError as e:
            print(f"Couldn't decode:\n {e}")
    except FileNotFoundError as e:
        print(f"No athlete files exist here: {e}")

    # Err if not present
    assert name in athletes, f"{name} has no entries, I only found:\n{athletes.keys()}"

    # NOTE: Code golf idiocy, does this even work?
    # Take passed in vars, except the last, and make a dict out of them
    supplied_values = {k:v for k,v in locals()[:-1].items() if v is not None}

    # Make a new Athlete out of a new dict, made by right-hand-side preferred merge
    updated_athlete = Athlete({**athletes[name], **supplied_values})

    # Mutate athletes dict at key Athlete.Name
    athletes[update_athlete.Name] = updated_athlete.dict()

    # Save back to the JSON file
    try:
        with open(file_path, 'w') as file:
            json.dump(athletes, file, indent=4)
    except FileNotFoundError:
        "I shouldn't be possible"


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
    # NOTE: I think this will kill the file if it already exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    supplied_values = {k:v for k,v in locals()[:-1].items() if v is not None}

    new_athlete = Athlete(**supplied_values)

    # Load existing athletes
    try:
        try:
            with open(file_path, 'r') as file:
                athletes = json.load(file)
        except json.JSONDecodeError as e:
            print(f"Couldn't decode:\n {e}")
    except FileNotFoundError as e:
        print(f"No athletes exist here:\n {e}")

    assert new_athlete.Name not in athletes.keys(), f"{new_athlete.Name} already exists: {Athlete(**athletes.name)}"

    athletes = athletes | new_athlete.dict()

    try:
        with open(file_path, 'w') as file:
            json.dump(athletes, file, indent=4)
    except FileNotFoundError:
        "I shouldn't be possible"


def get_athlete(name: str, file_path: str = './athletes.json') -> Optional[Athlete]:
    try:
        with open(file_path, 'r') as file:
            athletes = json.load(file)
    except FileNotFoundError as e:
        print(f"No athlete files exist here:\n {e}")

    # Error if either does not exist, or data is malformed
    try:
        return Athlete(**athletes[name])
    except ValidationError as e:
        print(f"I couldn't turn {athletes[name]} into an Athlete!")
