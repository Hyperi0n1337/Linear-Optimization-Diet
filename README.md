# Install Lib

## Build

Build jupyter
For the moment, from project folder directory, run
``` sh
python -m venv venv
```
then to activate virtual environment:
``` sh
venv\Scripts\activate
```
To install the necessary dependencies to the environment, run
``` sh
pip install -r requirements.txt --no-cache-dir
```
Then open jupyter in editor of choice, run the cells, they should work.
///

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
