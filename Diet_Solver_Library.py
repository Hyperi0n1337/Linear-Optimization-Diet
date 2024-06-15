
import pandas as pd
import numpy as np
import scipy as sp
import pulp
from scipy.optimize import linprog
from pulp import LpMaximize, LpProblem, LpStatus, lpSum, LpVariable, LpMinimize
import matplotlib.pyplot as plt
import seaborn as sns
from pydantic import BaseModel
import sys
import os
# Now you can import the create_athlete function
from athlete_manager import create_athlete, get_athlete



# Function to set up and solve the maintenance model for a given athlete
def solve_PSMF_model(athlete_name, file_path='FoodDatabase.csv'):
    # Retrieve athlete data
    athlete = get_athlete(athlete_name)
    if not athlete:
        raise ValueError(f"Athlete '{athlete_name}' not found.")
    
        # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path)

    # Filter rows with nonzero 'Price / kg'
    df_filtered = df[df['Price / kg'] != 0]

    # Replace NaNs with 0s
    df_filtered = df_filtered.fillna(0)

    # Add 'Cost ($)' column with initial value of zero at the start
    df_filtered.insert(0, 'Cost ($)', 0)

    # Add Quantities' (g)' column with initial value of zero as the second column
    df_filtered.insert(1, 'Quantities (g)', 0)

    #We want to minimize c_1 * x_1 + c_2 * x_2 ... c_n * x_n , where c is the cost per unit mass of a given food, and x is the mass of the given food (aka the objective function)

    # The constraints are the macros and micros of a diet, and other conditions we set and must be met. 

    #Excel was using Simplex LP method
    
    # Define base class for constraints
    class Constraint:
        def __init__(self, model, variables, coefficients, name):
            self.model = model
            self.variables = variables
            self.coefficients = coefficients
            self.name = name

        def get_sum(self):
            return lpSum(self.variables[i] * self.coefficients[i-1] for i in range(1, len(self.variables) + 1))

    class RangeConstraint(Constraint):
        def __init__(self, model, variables, coefficients, name, lower_bound, upper_bound):
            super().__init__(model, variables, coefficients, name)
            self.lower_bound = lower_bound
            self.upper_bound = upper_bound

        def add_constraints(self):
            sum_expr = self.get_sum()
            self.model += (sum_expr >= self.lower_bound, f"{self.name} Lower Bound Constraint")
            self.model += (sum_expr <= self.upper_bound, f"{self.name} Upper Bound Constraint")

    class MinimumConstraint(Constraint):
        def __init__(self, model, variables, coefficients, name, minimum):
            super().__init__(model, variables, coefficients, name)
            self.minimum = minimum

        def add_constraint(self):
            sum_expr = self.get_sum()
            self.model += (sum_expr >= self.minimum, f"{self.name} Minimum Constraint")

    class EqualityConstraint(Constraint):
        def __init__(self, model, variables, coefficients, name, value):
            super().__init__(model, variables, coefficients, name)
            self.value = value

        def add_constraint(self):
            sum_expr = self.get_sum()
            self.model += (sum_expr == self.value, f"{self.name} Equality Constraint")

    class MaximumConstraint(Constraint):
        def __init__(self, model, variables, coefficients, name, maximum):
            super().__init__(model, variables, coefficients, name)
            self.maximum = maximum

        def add_constraint(self):
            sum_expr = self.get_sum()
            self.model += (sum_expr <= self.maximum, f"{self.name} Maximum Constraint")

    # Define the model (for maintenance macro + micros)
    maintenance_model = LpProblem(name="Maintenance_Diet_Minimum_Cost", sense=LpMinimize)

    # Define the decision variables (what we are solving for; quantities of each food)
    x = {i: LpVariable(name=f"x_{i}", lowBound=0, upBound=10000) for i in range(1, len(df_filtered)+1)}

    # Function to extract column array from a DataFrame with optional scaling
    def extract_column_array(df, column, scale_factor=1):
        values = df.iloc[:, column].values
        if np.issubdtype(values.dtype, np.number):
            return values / scale_factor
        return values

    # Index dictionary for human-readable names
    index = {
        "dollars_per_g": (100, 1000),
        "name_per_food": (102, 1),
        "db_name_per_food": (3, 1),
        "category_per_food": (4, 1),
        "is_pill_per_food": (106, 1), # is or is not a pill (supplement)
        "calories_per_g": (5, 100), # scale factor in these is 100 to convert 100g into g
        "protein_per_g": (7, 100),
        "net_carbs_per_g": (95, 100),
        "fiber_per_g": (10, 100),
        "fat_per_g": (6, 100),
        "protein_quality": (101, 1),
        "sugar_per_g": (99, 100),
        "satfat_per_g": (12, 100),
        "calcium_per_g": (13, 100),
        "iron_per_g": (14, 100),
        "potassium_per_g": (15, 100),
        "magnesium_per_g": (16, 100),
        "sodium_per_g": (38, 100),
        "zinc_per_g": (39, 100),
        "copper_per_g": (40, 100),
        "manganese_per_g": (41, 100),
        "selenium_per_g": (42, 100),
        "vitaminC_per_g": (19, 100),
        "niacin_per_g": (48, 100),
        "EPA_per_g": (72, 100),
        "DHA_per_g": (74, 100),
        "beta_carotene_per_g": (59, 100),
        "alpha_carotene_per_g": (60, 100),
        "lycopene_per_g": (61, 100),
        "lut_zeaxanthin_per_g": (62, 100),
        "vitaminE_per_g": (22, 100),
    }

    # Function to get the array for a given nutrient type
    def get_nutrient_array(nutrient_type: str):
        if nutrient_type in index:
            col, scale = index[nutrient_type]
            return extract_column_array(df_filtered, col, scale)
        else:
            raise ValueError(f"Nutrient type '{nutrient_type}' is not recognized")

    dollars_per_g = get_nutrient_array("dollars_per_g")
    name_per_food = get_nutrient_array("name_per_food")
    db_name_per_food = get_nutrient_array("db_name_per_food")
    category_per_food = get_nutrient_array("category_per_food")
    is_pill_per_food = get_nutrient_array("is_pill_per_food")

    # Macros arrays
    calories_per_g = get_nutrient_array("calories_per_g")
    protein_per_g = get_nutrient_array("protein_per_g")
    net_carbs_per_g = get_nutrient_array("net_carbs_per_g")
    fiber_per_g = get_nutrient_array("fiber_per_g")
    fat_per_g = get_nutrient_array("fat_per_g")
    protein_quality = get_nutrient_array("protein_quality")
    sugar_per_g = get_nutrient_array("sugar_per_g")
    satfat_per_g = get_nutrient_array("satfat_per_g")

    # Micros arrays
    calcium_per_g = get_nutrient_array("calcium_per_g")
    iron_per_g = get_nutrient_array("iron_per_g")
    potassium_per_g = get_nutrient_array("potassium_per_g")
    magnesium_per_g = get_nutrient_array("magnesium_per_g")
    sodium_per_g = get_nutrient_array("sodium_per_g")
    zinc_per_g = get_nutrient_array("zinc_per_g")
    copper_per_g = get_nutrient_array("copper_per_g")
    manganese_per_g = get_nutrient_array("manganese_per_g")
    selenium_per_g = get_nutrient_array("selenium_per_g")
    vitaminC_per_g = get_nutrient_array("vitaminC_per_g")
    niacin_per_g = get_nutrient_array("niacin_per_g")
    DHA_per_g = get_nutrient_array("EPA_per_g")
    EPA_per_g = get_nutrient_array("DHA_per_g")
    FishOils_per_g = EPA_per_g + DHA_per_g
    beta_carotene_per_g = get_nutrient_array("beta_carotene_per_g")
    alpha_carotene_per_g = get_nutrient_array("alpha_carotene_per_g")
    lycopene_per_g = get_nutrient_array("lycopene_per_g")
    lut_zeaxanthin_per_g = get_nutrient_array("lut_zeaxanthin_per_g")
    vitaminE_per_g = get_nutrient_array("vitaminE_per_g")

    #weird carotenoids not included
    #fucoxanthin (found in seaweed?), astaxanthin (found in crab, lobster?)

    # Objective Function
    maintenance_model += lpSum(x[i] * dollars_per_g[i-1] for i in range(1, len(df_filtered)+1))

    # Defining Constraints
    # TODO: PULL THESE FROM SOMEWHERE ELSE, MAYBE READ CSV THAT HAS BODYWEIGHT AND OTHER STATS AND COMPUTE THE MAINTENANCE AND OTHER NEEDS BASED ON 
    #Constants (Determinants of variables)
    Weight_lbs = athlete.Weight_lbs
    BF_Percent = athlete.BF_Percent
    LBM = Weight_lbs-(Weight_lbs*BF_Percent)

    #For PSMF

    extra_PSMF_multiplier = 1 # extra adjustment factor (in %), normally leave 1
    lbm_protein_multiplier_table_for_PSMF = {
        0.01: 2.0,
        0.02: 2.0,
        0.03: 2.0,
        0.04: 2.0,
        0.05: 2.0,
        0.06: 2.0,
        0.07: 2.0,
        0.08: 2.0,
        0.09: 2.0,
        0.1: 2.0,
        0.11: 1.94,
        0.12: 1.88,
        0.13: 1.81,
        0.14: 1.75,
        0.15: 1.71,
        0.16: 1.66,
        0.17: 1.61,
        0.18: 1.56,
        0.19: 1.51,
        0.2: 1.46,
        0.21: 1.41,
        0.22: 1.36,
        0.23: 1.31,
        0.24: 1.26,
        0.25: 1.21,
        0.26: 1.16,
        0.27: 1.11,
        0.28: 1.06,
        0.29: 1.01,
        0.3: 1.0,
        0.31: 1.0,
        0.32: 1.0,
        0.33: 1.0,
        0.34: 1.0,
        0.35: 1.0
    }


    protein_minimum = LBM * lbm_protein_multiplier_table_for_PSMF[BF_Percent] * athlete.extra_PSMF_multiplier
    fiber_lbound = 10  # lower bound for fiber
    fat_ubound = 50 #not really necessary, but why not, carbs provide more energy anyway
    net_carb_maximum = 30  # set net carb minimum
    protein_quality_minimum = 0.4  # this is a fraction ie 0.4 = 40%

    # Macro constraints using classes
    macro_constraints = [
        MinimumConstraint(maintenance_model, x, protein_per_g, "Protein", protein_minimum),
        MaximumConstraint(maintenance_model, x, net_carbs_per_g, "Net Carb", net_carb_maximum),
        MinimumConstraint(maintenance_model, x, fiber_per_g, "Fiber", fiber_lbound),
        MaximumConstraint(maintenance_model, x, fat_per_g, "Fat", fat_ubound),
        MinimumConstraint(maintenance_model, x, [protein_quality[i-1] * protein_per_g[i-1] for i in range(1, len(x)+1)], "Protein Quality", protein_quality_minimum * lpSum(x[i] * protein_per_g[i-1] for i in range(1, len(x)+1))), # Remember, divisions need to be reformulated in the inequality as multiplications to solve properly.
    ]

    # Add macro constraints to the model
    for constraint in macro_constraints:
        constraint.add_constraint() if isinstance(constraint, (MinimumConstraint, EqualityConstraint, MaximumConstraint)) else constraint.add_constraints()

    Fish_Oils_minimum = 3000 # we want 3g EPA+DHA daily


    # Define constraints
    micro_constraints = [
        MinimumConstraint(maintenance_model, x, FishOils_per_g, "Fish Oils", Fish_Oils_minimum),
    ]

    # Add constraints to the model
    for constraint in micro_constraints:
        constraint.add_constraints() if isinstance(constraint, RangeConstraint) else constraint.add_constraint()

    # Display the full model info (comment out when not needed)
    #print(maintenance_model)
    
    # Solve the optimization problem
    status = maintenance_model.solve()

    # Create an empty DataFrame to store the results
    results_df = pd.DataFrame(columns=['Food Name', 'Quantity (g)', 'Category', 'is Pill'])

    # Collect the data for the DataFrame
    results_list = [{'Food Name': name_per_food[i-1], 'Category': category_per_food[i-1],'is Pill': is_pill_per_food[i-1], 'Quantity (g)': int(round(var.value(), 0))} for i, var in x.items() if var.value() > 0]

    # Convert the list to a DataFrame and sort
    results_df = pd.DataFrame(results_list).sort_values(by='Quantity (g)', ascending=False)


    # Calculate the cost for each food type and add it as a new column
    results_df['Cost ($)'] = 0.0  # Initialize the column

    for index, row in results_df.iterrows():
        product_name = row['Food Name']
        quantity = row['Quantity (g)']
        is_pill = row['is Pill']
        
        # Find the corresponding row in df_filtered
        matched_row = df_filtered[df_filtered['Product Name'] == product_name]
        
        if not matched_row.empty:
            price_per_unit = matched_row['Price / kg'].values[0]
            
            if is_pill == 1:
                # Calculate cost for pills
                cost = price_per_unit * (round(quantity / 100,0))  # quantity / 100 as it's in caps already
            else:
                # Calculate cost for non-pill items
                cost = (price_per_unit * quantity) / 1000  # quantity in grams
            
            results_df.at[index, 'Cost ($)'] = round(cost, 2)

    # Move 'Supplements' category to the end
    supplements_df = results_df[results_df['Category'] == 'Supplements']
    other_categories_df = results_df[results_df['Category'] != 'Supplements']

    # Concatenate both parts, ensuring 'Supplements' is at the end
    results_df = pd.concat([other_categories_df, supplements_df])

    # Get the results and format them
    print(f"status: {maintenance_model.status}, {LpStatus[maintenance_model.status]}")
    objective_value = round(results_df['Cost ($)'].sum(), 2)
    monthly_cost = round(results_df['Cost ($)'].sum() * 30, 2)
    print(f"TOTAL COST OF DAILY DIET CALCULATED:\t${objective_value}")
    print(f"TOTAL COST OF MONTHLY DIET CALCULATED:\t${monthly_cost}")

    # Display the results DataFrame
    print("\nCONSISTING OF THE FOLLOWING FOODS:")
    for index, row in results_df.iterrows():
        food_name = row['Food Name']
        if row['is Pill'] == 1:
            quantity = f"{int(round(row['Quantity (g)'] / 100, 0))} caps"
        else:
            quantity = f"{int(row['Quantity (g)'])} g"
        cost = f"${row['Cost ($)']:.2f}"
        category = row['Category']
        print(f"{food_name.ljust(95)}: {quantity.ljust(20)}{cost.ljust(10)}{category}")

    #Visualization

    # Ensure the pie chart is in the correct order
    results_df = results_df.sort_values(by='Cost ($)', ascending=False)

    # Calculate rounded total
    rounded_total = round(results_df['Cost ($)'].sum(), 2)

    # Plotting
    plt.figure(figsize=(10, 10))

    # Generate colors
    colors = sns.color_palette('husl', len(results_df))

    # Create pie chart
    wedges, texts, autotexts = plt.pie(results_df['Cost ($)'], labels=results_df['Food Name'], startangle=90, colors=colors,
                                    autopct=lambda pct: f"${results_df['Cost ($)'].sum() * pct / 100:.2f}\n({pct:.1f}%)",
                                    textprops=dict(color="w"))

    # Equal aspect ratio ensures the pie chart is a circle
    plt.axis('equal')  
    plt.title(f'Cost Distribution of Daily Diet by Food Item - Total: ${rounded_total:.2f}')

    # Adding legend with text and cost
    legend_labels = [f"{results_df.iloc[i]['Food Name']}: ${results_df.iloc[i]['Cost ($)']:.2f}" for i in range(len(results_df))]
    plt.legend(wedges, legend_labels, loc='lower left', bbox_to_anchor=(-1, 0))

    plt.show()
    
    
    # Print errors per constraint equation
    print("\nERROR PER CONSTRAINT EQUATION:")
    for name, constraint in maintenance_model.constraints.items():
        # Replace underscores with spaces
        name_with_spaces = name.replace('_', ' ')

        # Determine the operator and RHS value
        if "Lower" in name:
            operator = '>='
            rhs_value = -constraint.constant
        elif "Upper" in name:
            operator = '<='
            rhs_value = -constraint.constant
        elif "Minimum" in name:
            operator = '>='
            rhs_value = -constraint.constant
        elif "Equality" in name:
            operator = '=='
            rhs_value = -constraint.constant
        elif "Maximum" in name:
            operator = '<='
            rhs_value = -constraint.constant

        # Calculate the left-hand side value
        lhs_value = sum(var.value() * coeff for var, coeff in constraint.items())

        # Round the lhs_value and rhs_value to the nearest integer
        lhs_value_rounded = round(lhs_value)
        rhs_value_rounded = round(rhs_value)

        # Calculate the error and round to three significant figures
        constraint_value_rounded = round(abs(lhs_value - rhs_value), 2)

        # Format the output
        if constraint_value_rounded >= 0.1:
            print(f"{name_with_spaces.ljust(55)}\t{lhs_value_rounded}\t{operator}\t{rhs_value_rounded},\t{constraint_value_rounded}")
        else:
            print(f"{name_with_spaces.ljust(55)}\t{lhs_value_rounded}\t{operator}\t{rhs_value_rounded}")

    
# Function to set up and solve the maintenance model for a given athlete
def solve_maintenance_model(athlete_name, file_path='FoodDatabase.csv'):
    # Retrieve athlete data
    athlete = get_athlete(athlete_name)
    if not athlete:
        raise ValueError(f"Athlete '{athlete_name}' not found.")
    
        # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path)

    # Filter rows with nonzero 'Price / kg'
    df_filtered = df[df['Price / kg'] != 0]

    # Replace NaNs with 0s
    df_filtered = df_filtered.fillna(0)

    # Add 'Cost ($)' column with initial value of zero at the start
    df_filtered.insert(0, 'Cost ($)', 0)

    # Add Quantities' (g)' column with initial value of zero as the second column
    df_filtered.insert(1, 'Quantities (g)', 0)

    #We want to minimize c_1 * x_1 + c_2 * x_2 ... c_n * x_n , where c is the cost per unit mass of a given food, and x is the mass of the given food (aka the objective function)

    # The constraints are the macros and micros of a diet, and other conditions we set and must be met. 

    #Excel was using Simplex LP method
    
    # Define base class for constraints
    class Constraint:
        def __init__(self, model, variables, coefficients, name):
            self.model = model
            self.variables = variables
            self.coefficients = coefficients
            self.name = name

        def get_sum(self):
            return lpSum(self.variables[i] * self.coefficients[i-1] for i in range(1, len(self.variables) + 1))

    class RangeConstraint(Constraint):
        def __init__(self, model, variables, coefficients, name, lower_bound, upper_bound):
            super().__init__(model, variables, coefficients, name)
            self.lower_bound = lower_bound
            self.upper_bound = upper_bound

        def add_constraints(self):
            sum_expr = self.get_sum()
            self.model += (sum_expr >= self.lower_bound, f"{self.name} Lower Bound Constraint")
            self.model += (sum_expr <= self.upper_bound, f"{self.name} Upper Bound Constraint")

    class MinimumConstraint(Constraint):
        def __init__(self, model, variables, coefficients, name, minimum):
            super().__init__(model, variables, coefficients, name)
            self.minimum = minimum

        def add_constraint(self):
            sum_expr = self.get_sum()
            self.model += (sum_expr >= self.minimum, f"{self.name} Minimum Constraint")

    class EqualityConstraint(Constraint):
        def __init__(self, model, variables, coefficients, name, value):
            super().__init__(model, variables, coefficients, name)
            self.value = value

        def add_constraint(self):
            sum_expr = self.get_sum()
            self.model += (sum_expr == self.value, f"{self.name} Equality Constraint")

    class MaximumConstraint(Constraint):
        def __init__(self, model, variables, coefficients, name, maximum):
            super().__init__(model, variables, coefficients, name)
            self.maximum = maximum

        def add_constraint(self):
            sum_expr = self.get_sum()
            self.model += (sum_expr <= self.maximum, f"{self.name} Maximum Constraint")

    # Define the model (for maintenance macro + micros)
    maintenance_model = LpProblem(name="Maintenance_Diet_Minimum_Cost", sense=LpMinimize)

    # Define the decision variables (what we are solving for; quantities of each food)
    x = {i: LpVariable(name=f"x_{i}", lowBound=0, upBound=10000) for i in range(1, len(df_filtered)+1)}

    # Function to extract column array from a DataFrame with optional scaling
    def extract_column_array(df, column, scale_factor=1):
        values = df.iloc[:, column].values
        if np.issubdtype(values.dtype, np.number):
            return values / scale_factor
        return values

    # Index dictionary for human-readable names
    index = {
        "dollars_per_g": (100, 1000),
        "name_per_food": (102, 1),
        "db_name_per_food": (3, 1),
        "category_per_food": (4, 1),
        "is_pill_per_food": (106, 1), # is or is not a pill (supplement)
        "calories_per_g": (5, 100), # scale factor in these is 100 to convert 100g into g
        "protein_per_g": (7, 100),
        "net_carbs_per_g": (95, 100),
        "fiber_per_g": (10, 100),
        "fat_per_g": (6, 100),
        "protein_quality": (101, 1),
        "sugar_per_g": (99, 100),
        "satfat_per_g": (12, 100),
        "calcium_per_g": (13, 100),
        "iron_per_g": (14, 100),
        "potassium_per_g": (15, 100),
        "magnesium_per_g": (16, 100),
        "sodium_per_g": (38, 100),
        "zinc_per_g": (39, 100),
        "copper_per_g": (40, 100),
        "manganese_per_g": (41, 100),
        "selenium_per_g": (42, 100),
        "vitaminC_per_g": (19, 100),
        "niacin_per_g": (48, 100),
        "EPA_per_g": (72, 100),
        "DHA_per_g": (74, 100),
        "beta_carotene_per_g": (59, 100),
        "alpha_carotene_per_g": (60, 100),
        "lycopene_per_g": (61, 100),
        "lut_zeaxanthin_per_g": (62, 100),
        "vitaminE_per_g": (22, 100),
    }

    # Function to get the array for a given nutrient type
    def get_nutrient_array(nutrient_type: str):
        if nutrient_type in index:
            col, scale = index[nutrient_type]
            return extract_column_array(df_filtered, col, scale)
        else:
            raise ValueError(f"Nutrient type '{nutrient_type}' is not recognized")

    dollars_per_g = get_nutrient_array("dollars_per_g")
    name_per_food = get_nutrient_array("name_per_food")
    db_name_per_food = get_nutrient_array("db_name_per_food")
    category_per_food = get_nutrient_array("category_per_food")
    is_pill_per_food = get_nutrient_array("is_pill_per_food")

    # Macros arrays
    calories_per_g = get_nutrient_array("calories_per_g")
    protein_per_g = get_nutrient_array("protein_per_g")
    net_carbs_per_g = get_nutrient_array("net_carbs_per_g")
    fiber_per_g = get_nutrient_array("fiber_per_g")
    fat_per_g = get_nutrient_array("fat_per_g")
    protein_quality = get_nutrient_array("protein_quality")
    sugar_per_g = get_nutrient_array("sugar_per_g")
    satfat_per_g = get_nutrient_array("satfat_per_g")

    # Micros arrays
    calcium_per_g = get_nutrient_array("calcium_per_g")
    iron_per_g = get_nutrient_array("iron_per_g")
    potassium_per_g = get_nutrient_array("potassium_per_g")
    magnesium_per_g = get_nutrient_array("magnesium_per_g")
    sodium_per_g = get_nutrient_array("sodium_per_g")
    zinc_per_g = get_nutrient_array("zinc_per_g")
    copper_per_g = get_nutrient_array("copper_per_g")
    manganese_per_g = get_nutrient_array("manganese_per_g")
    selenium_per_g = get_nutrient_array("selenium_per_g")
    vitaminC_per_g = get_nutrient_array("vitaminC_per_g")
    niacin_per_g = get_nutrient_array("niacin_per_g")
    DHA_per_g = get_nutrient_array("EPA_per_g")
    EPA_per_g = get_nutrient_array("DHA_per_g")
    FishOils_per_g = EPA_per_g + DHA_per_g
    beta_carotene_per_g = get_nutrient_array("beta_carotene_per_g")
    alpha_carotene_per_g = get_nutrient_array("alpha_carotene_per_g")
    lycopene_per_g = get_nutrient_array("lycopene_per_g")
    lut_zeaxanthin_per_g = get_nutrient_array("lut_zeaxanthin_per_g")
    vitaminE_per_g = get_nutrient_array("vitaminE_per_g")

    #weird carotenoids not included
    #fucoxanthin (found in seaweed?), astaxanthin (found in crab, lobster?)

    # Objective Function
    maintenance_model += lpSum(x[i] * dollars_per_g[i-1] for i in range(1, len(df_filtered)+1))

    # Defining Constraints
    # TODO: PULL THESE FROM SOMEWHERE ELSE, MAYBE READ CSV THAT HAS BODYWEIGHT AND OTHER STATS AND COMPUTE THE MAINTENANCE AND OTHER NEEDS BASED ON 
    #Constants (Determinants of variables)
    Weight_lbs = athlete.Weight_lbs
    BF_Percent = athlete.BF_Percent
    LBM = Weight_lbs-(Weight_lbs*BF_Percent)

    #For Maintenance
    Caloric_Multiplier = athlete.Caloric_Multiplier #Calculated based on pre-cut maintenance; use a value 16-18 typically
    Extra_Caloric_Multiplier = athlete.Extra_Caloric_Multiplier #Percentage, use for extra 'fudge' factor tbh?
    #####################################################

    # Macros
    maintenance_calories = Weight_lbs*Caloric_Multiplier*Extra_Caloric_Multiplier  # set maintenance calories
    protein_minimum = 1.5*LBM # 1.1 - 1.4 g / lbm protein for athletes, 1.4 - 1.5 g / lbm protein when dieting, (I'm just using the top of the range 1.5 g / lbm cuz im awesome)
    fiber_lbound = 30  # lower bound for fiber
    fiber_ubound = 60  # upper bound for fiber
    fat_lbound = 0.25*maintenance_calories/9
    fat_ubound = 0.35*maintenance_calories/9 #not really necessary, but why not, carbs provide more energy anyway
    net_carb_minimum = (maintenance_calories-(protein_minimum*4+fat_ubound*9))/4  # set net carb minimum
    protein_quality_minimum = 0.4  # this is a fraction ie 0.4 = 40%
    sugar_maximum = 75
    sat_fat_maximum = 0.5  # this is a fraction

    # Macro constraints using classes
    macro_constraints = [
        EqualityConstraint(maintenance_model, x, calories_per_g, "Maintenance Calories", maintenance_calories),
        MinimumConstraint(maintenance_model, x, protein_per_g, "Protein", protein_minimum),
        MinimumConstraint(maintenance_model, x, net_carbs_per_g, "Net Carb", net_carb_minimum),
        RangeConstraint(maintenance_model, x, fiber_per_g, "Fiber", fiber_lbound, fiber_ubound),
        RangeConstraint(maintenance_model, x, fat_per_g, "Fat", fat_lbound,fat_ubound),
        MinimumConstraint(maintenance_model, x, [protein_quality[i-1] * protein_per_g[i-1] for i in range(1, len(x)+1)], "Protein Quality", protein_quality_minimum * lpSum(x[i] * protein_per_g[i-1] for i in range(1, len(x)+1))), # Remember, divisions need to be reformulated in the inequality as multiplications to solve properly.
        MaximumConstraint(maintenance_model, x, sugar_per_g, "Sugar", sugar_maximum),
        MaximumConstraint(maintenance_model, x, [satfat_per_g[i-1] for i in range(1, len(x)+1)], "SatFat", sat_fat_maximum * lpSum(x[i] * fat_per_g[i-1] for i in range(1, len(x)+1))) # Currently Fast and loose with fat rules. Sat fats no more than 50% of total fats. May change this in the future.
    ]

    # Add macro constraints to the model
    for constraint in macro_constraints:
        constraint.add_constraint() if isinstance(constraint, (MinimumConstraint, EqualityConstraint, MaximumConstraint)) else constraint.add_constraints()

    # Micros (mg unless otherwise stated)
    calcium_lbound = 1500
    calcium_ubound = 2000
    iron_lbound = 30
    iron_ubound = 40
    potassium_maximum = 10000
    magnesium_lbound = 600  # Maybe play around with the calcium to magnesium ratios; apparently should be within 1.70 - 2.60 , but high calcium recommendations and maximum magnesium daily intake of 420 mg makes that difficult to attain. Let's stretch it to 1.70 - 3.15 (this probably doesn't matter if you get enough of everything)
    magnesium_ubound = 700
    sodium_maximum = 5000  # because potassium maximum is 10k, you want potassium to sodium ratio to be 2:1 or higher; but if you sweat a lot, probably can exceed it. just make sure potassium is at least 1:1 with sodium.
    zinc_lbound = 35
    zinc_ubound = 40
    copper_maximum = 10
    manganese_lbound = 8 # Probably more than ample
    manganese_ubound = 20
    selenium_maximum = 350  # micrograms
    vitaminC_lbound = 120 # spread between 3 meals? 
    vitaminC_ubound = 180
    niacin_minimum = 25
    Fish_Oils_minimum = 3000 # we want 3g EPA+DHA daily
    #Below are just 'educated' guesses based on papers:
    beta_carotene_minimum = 16000 # in mcg, 1 mg = 1000 mcg
    alpha_carotene_minimum = 4000 # random guess
    lycopene_minimum = 11000 #
    lut_zeaxanthin_minimum = 12000 # 
    vitaminE_minimum = 10 # this number ruins everything if it's just a bit high tbh

    # Define constraints
    micro_constraints = [
        RangeConstraint(maintenance_model, x, calcium_per_g, "Calcium", calcium_lbound, calcium_ubound),
        RangeConstraint(maintenance_model, x, iron_per_g, "Iron", iron_lbound, iron_ubound),
        MaximumConstraint(maintenance_model, x, potassium_per_g, "Potassium", potassium_maximum),
        RangeConstraint(maintenance_model, x, magnesium_per_g, "Magnesium", magnesium_lbound, magnesium_ubound),
        MaximumConstraint(maintenance_model, x, sodium_per_g, "Sodium", sodium_maximum),
        RangeConstraint(maintenance_model, x, zinc_per_g, "Zinc", zinc_lbound, zinc_ubound),
        MaximumConstraint(maintenance_model, x, copper_per_g, "Copper", copper_maximum),
        RangeConstraint(maintenance_model, x, manganese_per_g, "Manganese", manganese_lbound, manganese_ubound),
        MaximumConstraint(maintenance_model, x, selenium_per_g, "Selenium", selenium_maximum),
        RangeConstraint(maintenance_model, x, vitaminC_per_g, "Vitamin C", vitaminC_lbound, vitaminC_ubound),
        MinimumConstraint(maintenance_model, x, niacin_per_g, "Niacin", niacin_minimum),
        MinimumConstraint(maintenance_model, x, FishOils_per_g, "Fish Oils", Fish_Oils_minimum),
        MinimumConstraint(maintenance_model, x, beta_carotene_per_g, "Beta Carotene", beta_carotene_minimum),
        MinimumConstraint(maintenance_model, x, alpha_carotene_per_g, "Alpha Carotene", alpha_carotene_minimum),
        MinimumConstraint(maintenance_model, x, lycopene_per_g, "Lycopene", lycopene_minimum),
        MinimumConstraint(maintenance_model, x, lut_zeaxanthin_per_g, "Lutein / Zeaxanthin", lut_zeaxanthin_minimum),
        MinimumConstraint(maintenance_model, x, vitaminE_per_g, "Vitamin E", vitaminE_minimum),
    ]

    # Add constraints to the model
    for constraint in micro_constraints:
        constraint.add_constraints() if isinstance(constraint, RangeConstraint) else constraint.add_constraint()

    # Display the full model info (comment out when not needed)
    #print(maintenance_model)
    
    # Solve the optimization problem
    status = maintenance_model.solve()

    # Create an empty DataFrame to store the results
    results_df = pd.DataFrame(columns=['Food Name', 'Quantity (g)', 'Category', 'is Pill'])

    # Collect the data for the DataFrame
    results_list = [{'Food Name': name_per_food[i-1], 'Category': category_per_food[i-1],'is Pill': is_pill_per_food[i-1], 'Quantity (g)': int(round(var.value(), 0))} for i, var in x.items() if var.value() > 0]

    # Convert the list to a DataFrame and sort
    results_df = pd.DataFrame(results_list).sort_values(by='Quantity (g)', ascending=False)


    # Calculate the cost for each food type and add it as a new column
    results_df['Cost ($)'] = 0.0  # Initialize the column

    for index, row in results_df.iterrows():
        product_name = row['Food Name']
        quantity = row['Quantity (g)']
        is_pill = row['is Pill']
        
        # Find the corresponding row in df_filtered
        matched_row = df_filtered[df_filtered['Product Name'] == product_name]
        
        if not matched_row.empty:
            price_per_unit = matched_row['Price / kg'].values[0]
            
            if is_pill == 1:
                # Calculate cost for pills
                cost = price_per_unit * (round(quantity / 100,0))  # quantity / 100 as it's in caps already
            else:
                # Calculate cost for non-pill items
                cost = (price_per_unit * quantity) / 1000  # quantity in grams
            
            results_df.at[index, 'Cost ($)'] = round(cost, 2)

    # Move 'Supplements' category to the end
    supplements_df = results_df[results_df['Category'] == 'Supplements']
    other_categories_df = results_df[results_df['Category'] != 'Supplements']

    # Concatenate both parts, ensuring 'Supplements' is at the end
    results_df = pd.concat([other_categories_df, supplements_df])

    # Get the results and format them
    print(f"status: {maintenance_model.status}, {LpStatus[maintenance_model.status]}")
    objective_value = round(results_df['Cost ($)'].sum(), 2)
    monthly_cost = round(results_df['Cost ($)'].sum() * 30, 2)
    print(f"TOTAL COST OF DAILY DIET CALCULATED:\t${objective_value}")
    print(f"TOTAL COST OF MONTHLY DIET CALCULATED:\t${monthly_cost}")

    # Display the results DataFrame
    print("\nCONSISTING OF THE FOLLOWING FOODS:")
    for index, row in results_df.iterrows():
        food_name = row['Food Name']
        if row['is Pill'] == 1:
            quantity = f"{int(round(row['Quantity (g)'] / 100, 0))} caps"
        else:
            quantity = f"{int(row['Quantity (g)'])} g"
        cost = f"${row['Cost ($)']:.2f}"
        category = row['Category']
        print(f"{food_name.ljust(95)}: {quantity.ljust(20)}{cost.ljust(10)}{category}")

    #Visualization

    # Ensure the pie chart is in the correct order
    results_df = results_df.sort_values(by='Cost ($)', ascending=False)

    # Calculate rounded total
    rounded_total = round(results_df['Cost ($)'].sum(), 2)

    # Plotting
    plt.figure(figsize=(10, 10))

    # Generate colors
    colors = sns.color_palette('husl', len(results_df))

    # Create pie chart
    wedges, texts, autotexts = plt.pie(results_df['Cost ($)'], labels=results_df['Food Name'], startangle=90, colors=colors,
                                    autopct=lambda pct: f"${results_df['Cost ($)'].sum() * pct / 100:.2f}\n({pct:.1f}%)",
                                    textprops=dict(color="w"))

    # Equal aspect ratio ensures the pie chart is a circle
    plt.axis('equal')  
    plt.title(f'Cost Distribution of Daily Diet by Food Item - Total: ${rounded_total:.2f}')

    # Adding legend with text and cost
    legend_labels = [f"{results_df.iloc[i]['Food Name']}: ${results_df.iloc[i]['Cost ($)']:.2f}" for i in range(len(results_df))]
    plt.legend(wedges, legend_labels, loc='lower left', bbox_to_anchor=(-1, 0))

    plt.show()
    
        
    # Print errors per constraint equation
    print("\nERROR PER CONSTRAINT EQUATION:")
    for name, constraint in maintenance_model.constraints.items():
        # Replace underscores with spaces
        name_with_spaces = name.replace('_', ' ')

        # Determine the operator and RHS value
        if "Lower" in name:
            operator = '>='
            rhs_value = -constraint.constant
        elif "Upper" in name:
            operator = '<='
            rhs_value = -constraint.constant
        elif "Minimum" in name:
            operator = '>='
            rhs_value = -constraint.constant
        elif "Equality" in name:
            operator = '=='
            rhs_value = -constraint.constant
        elif "Maximum" in name:
            operator = '<='
            rhs_value = -constraint.constant

        # Calculate the left-hand side value
        lhs_value = sum(var.value() * coeff for var, coeff in constraint.items())

        # Round the lhs_value and rhs_value to the nearest integer
        lhs_value_rounded = round(lhs_value)
        rhs_value_rounded = round(rhs_value)

        # Calculate the error and round to three significant figures
        constraint_value_rounded = round(abs(lhs_value - rhs_value), 2)

        # Format the output
        if constraint_value_rounded >= 0.1:
            print(f"{name_with_spaces.ljust(55)}\t{lhs_value_rounded}\t{operator}\t{rhs_value_rounded},\t{constraint_value_rounded}")
        else:
            print(f"{name_with_spaces.ljust(55)}\t{lhs_value_rounded}\t{operator}\t{rhs_value_rounded}")


    