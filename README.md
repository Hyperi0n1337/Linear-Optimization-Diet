# Install Lib

## Build

Build jupyter
``` sh
Whatever is valid instructions
```

Build Lib

exit from existing venv, probably, then:
``` sh
cd lib
python3 -m venv venv && source venv/bin/activate
pip install poetry
poetry install --no-root
```

## Use

``` python3
import lib.Athlete
import json
import os 
from rich import print

with open("./lib/test_files/sample_athlete.json", "r") as samplefile:
    athlete = json.load(samplefile)
    try:
        a = Athlete(**athlete)
        print(f"Athlete's attributes are: {a})
    except:
        print(f"couldn't parse valid athlete from {samplefile}")
```

## Make jupyter into raw python file

` jupyter nbconvert --to script Linear\ Optimization.ipynb`
