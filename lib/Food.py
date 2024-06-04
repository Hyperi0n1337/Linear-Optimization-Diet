import json
from typing import Dict, List, Optional
from rich import print
from rich.traceback import install

from pydantic import BaseModel, Field

class Food(BaseModel):

    # I don't see any reason to parse the units at the moment
    # The Field bit is to fix the fucked up keys that make
    # illegal dict values in python, class fields should
    # be lower case anyway. I did this with a vim macro

    id: Optional[int] = Field(alias="ID")
    name: Optional[str]
    category: Optional[str] = Field(alias="Category") # TODO: Category class
    calories: Optional[float] = Field(alias="Calories")
    fat: Optional[float] = Field(alias="Fat (g)")
    protein: Optional[float] = Field(alias="Protein (g)")
    carbohydrate: Optional[float] = Field(alias="Carbohydrate (g)")
    sugars: Optional[float] = Field(alias="Sugars (g)")
    fiber: Optional[float] = Field(alias="Fiber (g)")
    cholesterol: Optional[float] = Field(alias="Cholesterol (mg)")
    saturated: Optional[float] = Field(alias="Saturated Fats (g)")
    calcium: Optional[float] = Field(alias="Calcium (mg)")
    iron: Optional[float] = Field(alias="Iron, Fe (mg)")
    potassium: Optional[float] = Field(alias="Potassium, K (mg)")
    magnesium: Optional[float] = Field(alias="Magnesium (mg)")
    vitamin_a: Optional[float] = Field(alias="Vitamin A, IU (IU)")
    vitamin_a: Optional[float] = Field(alias="Vitamin A, RAE (mcg)")
    vitamin_c: Optional[float] = Field(alias="Vitamin C (mg)")
    vitamin_b12: Optional[float] = Field(alias="Vitamin B-12 (mcg)")
    vitamin_d: Optional[float] = Field(alias="Vitamin D (mcg)")
    vitamin_e: Optional[float] = Field(alias="Vitamin E (Alpha-Tocopherol) (mg)")
    net_carbs: Optional[float] = Field(alias="Net-Carbs (g)")
    water: Optional[float] = Field(alias="Water (g)")
    omega3: Optional[float] = Field(alias="Omega 3s (mg)")
    omega6: Optional[float] = Field(alias="Omega 6s (mg)")
    pral_score: Optional[float] = Field(alias="PRAL score")
    trans_fatty_acids: Optional[float] = Field(alias="Trans Fatty Acids (g)")
    sucrose: Optional[float] = Field(alias="Sucrose (g)")
    glucose: Optional[float] = Field(alias="Glucose (Dextrose) (g)")
    fructose: Optional[float] = Field(alias="Fructose (g)")
    lactose: Optional[float] = Field(alias="Lactose (g)")
    maltose: Optional[float] = Field(alias="Maltose (g)")
    galactose: Optional[float] = Field(alias="Galactose (g)")
    starch: Optional[float] = Field(alias="Starch (g)")
    sugar_alcohols: Optional[float] = Field(alias="Total sugar alcohols (g)")
    phosphorus: Optional[float] = Field(alias="Phosphorus, P (mg)")
    sodium: Optional[float] = Field(alias="Sodium (mg)")
    zinc: Optional[float] = Field(alias="Zinc, Zn (mg)")
    copper: Optional[float] = Field(alias="Copper, Cu (mg)")
    manganese: Optional[float] = Field(alias="Manganese (mg)")
    selenium: Optional[float] = Field(alias="Selenium, Se (mcg)")
    fluoride: Optional[float] = Field(alias="Fluoride, F (mcg)")
    molybdenum: Optional[float] = Field(alias="Molybdenum (mcg)")
    chlorine: Optional[float] = Field(alias="Chlorine (mg)")
    thiamin: Optional[float] = Field(alias="Thiamin (B1) (mg)")
    riboflavin: Optional[float] = Field(alias="Riboflavin (B2) (mg)")
    niacin: Optional[float] = Field(alias="Niacin (B3) (mg)")
    pantothenic_acid: Optional[float] = Field(alias="Pantothenic acid (B5) (mg)")
    vitamin_b6: Optional[float] = Field(alias="Vitamin B6 (mg)")
    biotin: Optional[float] = Field(alias="Biotin (B7) (mcg)")
    folate: Optional[float] = Field(alias="Folate (B9) (mcg)")
    folic_acid: Optional[float] = Field(alias="Folic acid (mcg)")
    food_folate: Optional[float] = Field(alias="Food Folate (mcg)")
    folate_dfe: Optional[float] = Field(alias="Folate DFE (mcg)")
    choline: Optional[float] = Field(alias="Choline (mg)")
    betaine: Optional[float] = Field(alias="Betaine (mg)")
    retinol: Optional[float] = Field(alias="Retinol (mcg)")
    beta_carotene: Optional[float] = Field(alias="Carotene, beta (mcg)")
    alpha_carotene: Optional[float] = Field(alias="Carotene, alpha (mcg)")
    lycopene: Optional[float] = Field(alias="Lycopene (mcg)")
    lutein_zeaxanthin: Optional[float] = Field(alias="Lutein + Zeaxanthin (mcg)")
    vitamin_d2: Optional[float] = Field(alias="Vitamin D2 (ergocalciferol) (mcg)")
    vitamin_d3: Optional[float] = Field(alias="Vitamin D3 (cholecalciferol) (mcg)")
    vitamin_d: Optional[float] = Field(alias="Vitamin D (IU) (IU)")
    vitamin_k: Optional[float] = Field(alias="Vitamin K (mcg)")
    dihydrophylloquinone: Optional[float] = Field(alias="Dihydrophylloquinone (mcg)")
    menaquinone_4: Optional[float] = Field(alias="Menaquinone-4 (mcg)")
    mono_fat: Optional[float] = Field(alias="Fatty acids, total monounsaturated (mg)")
    poly_fat: Optional[float] = Field(alias="Fatty acids, total polyunsaturated (mg)")
    ala: Optional[float] = Field(alias="18:3 n-3 c,c,c (ALA) (mg)")
    epa: Optional[float] = Field(alias="20:5 n-3 (EPA) (mg)")
    dpa: Optional[float] = Field(alias="22:5 n-3 (DPA) (mg)")
    dha: Optional[float] = Field(alias="22:6 n-3 (DHA) (mg)")
    tryptophan: Optional[float] = Field(alias="Tryptophan (mg)")
    threonine: Optional[float] = Field(alias="Threonine (mg)")
    isoleucine: Optional[float] = Field(alias="Isoleucine (mg)")
    leucine: Optional[float] = Field(alias="Leucine (mg)")
    lysine: Optional[float] = Field(alias="Lysine (mg)")
    methionine: Optional[float] = Field(alias="Methionine (mg)")
    cystine: Optional[float] = Field(alias="Cystine (mg)")
    phenylalanine: Optional[float] = Field(alias="Phenylalanine (mg)")
    tyrosine: Optional[float] = Field(alias="Tyrosine (mg)")
    valine: Optional[float] = Field(alias="Valine (mg)")
    arginine: Optional[float] = Field(alias="Arginine (mg)")
    histidine: Optional[float] = Field(alias="Histidine (mg)")
    alanine: Optional[float] = Field(alias="Alanine (mg)")
    aspartic_acid: Optional[float] = Field(alias="Aspartic acid (mg)")
    glutamic_acid: Optional[float] = Field(alias="Glutamic acid (mg)")
    glycine: Optional[float] = Field(alias="Glycine (mg)")
    proline: Optional[float] = Field(alias="Proline (mg)")
    serine: Optional[float] = Field(alias="Serine (mg)")
    hydroxyproline: Optional[float] = Field(alias="Hydroxyproline (mg)")
    alcohol: Optional[float] = Field(alias="Alcohol (g)")
    net_carbs: Optional[float] = Field(alias="Net Carbs (g)")
    caffeine: Optional[float] = Field(alias="Caffeine (mg)")
    theobromine: Optional[float] = Field(alias="Theobromine (mg)")
    calorie_weight_200: Optional[float] = Field(alias="200 Calorie Weight (g)")
    sugar: Optional[float] = Field(alias="Sugar (g)")
    price_per_kg: Optional[float] = Field(alias="Price / kg")
    protein_quality: Optional[float] = Field(alias="Protein Quality")
    product_name: Optional[str] = Field(alias="Product Name")
    last_updated: Optional[str] = Field(alias="Last Updated") # TODO: Figure out appropriate datetime
    source: Optional[str] = Field(alias="Source")
    comments: Optional[str] = Field(alias="Comments")
    is_pill: Optional[bool] = Field(alias="is Pill")


def main():
    install(show_locals=True)
    with open("./test_files/FoodJson.json", "r") as foods:
        foods = json.load(foods)
        fl = list()
        # Leaving this out of a try block since pydantic will blow
        # up more helpfully if it can just print to rich terminal
        for food in foods:
            fl += Food(**food)
        print(f"couldn't parse valid foods from foods")

    print(len(fl))
    print(fl[5])


if __name__ == "__main__":
    main()
