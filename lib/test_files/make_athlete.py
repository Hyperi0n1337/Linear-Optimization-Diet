import json
import os

def main():

    Weight_lbs = 194
    BF_Percent = 0.17
    LBM = Weight_lbs-(Weight_lbs*BF_Percent)

    #For Maintenance
    Caloric_Multiplier = 15.1 #Calculated based on pre-cut maintenance; use a value 16-18 typically
    Extra_Caloric_Multiplier = 1 #Percentage, use for extra 'fudge' factor tbh?

    total_calories = Weight_lbs*Caloric_Multiplier*Extra_Caloric_Multiplier
    protein_minimum = 1.5*LBM # 1.1 - 1.4 g / lbm protein for athletes, 1.4 - 1.5 g / lbm protein when dieting, (I'm just using the top of the range 1.5 g / lbm cuz im awesome)
    fat_minimum = 0.25*total_calories/9
    fat_maximum = 0.35*total_calories/9 #not really necessary, but why not, carbs provide more energy anyway
    net_carb_minimum = total_calories-(protein_minimum*4+fat_maximum*9)

    athlete = dict()
    athlete["Weight_lbs"] = 194
    athlete["BF_Percent"] = 0.17
    athlete["LBM"] = Weight_lbs-(Weight_lbs*BF_Percent)
    athlete["Caloric_Multiplier"] = 15.1 #Calculated based on pre-cut maintenance; use a value 16-18 typically
    athlete["Extra_Caloric_Multiplier"] = 1 #Percentage, use for extra 'fudge' factor tbh?
    athlete["total_calories"] = Weight_lbs*Caloric_Multiplier*Extra_Caloric_Multiplier
    athlete["protein_minimum"] = 1.5*LBM # 1.1 - 1.4 g / lbm protein for athletes, 1.4 - 1.5 g / lbm protein when dieting, (I'm just using the top of the range 1.5 g / lbm cuz im awesome)
    athlete["fat_minimum"] = 0.25*total_calories/9
    athlete["fat_maximum"] = 0.35*total_calories/9 #not really necessary, but why not, carbs provide more energy anyway
    athlete["net_carb_minimum"] = total_calories-(protein_minimum*4+fat_maximum*9)

    print(athlete)
    with open("sample_athlete.json", "w") as samplefile:
        json.dump(athlete, samplefile, indent=4)

if __name__ == "__main__":
    main()
