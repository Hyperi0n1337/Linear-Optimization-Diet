#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import scipy as sp
import pulp
from scipy.optimize import linprog
from pulp import LpMaximize, LpProblem, LpStatus, lpSum, LpVariable, LpMinimize


# In[2]:


# Assuming FoodDatabase.csv is in the same directory as your Jupyter Notebook
file_path = 'FoodDatabase.csv'
#All numbers (macros and micros) in the table are a serving of 100g

# Read the CSV file into a DataFrame
df = pd.read_csv(file_path)

# Filter rows with nonzero 'Price / kg'
df_filtered = df[df['Price / kg'] != 0]

# Add 'Cost ($)' column with initial value of zero at the start
df_filtered.insert(0, 'Cost ($)', 0)

# Add Quantities' (g)' column with initial value of zero as the second column
df_filtered.insert(1, 'Quantities (g)', 0)

# Display the first few rows of the DataFrame to verify it was read correctly
df_filtered.info()
df_filtered.head()


#We want to minimize c_1 * x_1 + c_2 * x_2 ... c_n * x_n , where c is the cost per unit mass of a given food, and x is the mass of the given food (aka the objective function)

# The constraints are the macros and micros of a diet, and other conditions we set and must be met. 

#Excel was using Simplex LP method


# In[3]:


# Define classes for constraints
class RangeConstraint:
    def __init__(self, model, variables, coefficients, name, lower_bound, upper_bound):
        self.model = model
        self.variables = variables
        self.coefficients = coefficients
        self.name = name
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def add_constraints(self):
        self.model += (lpSum(self.variables[i] * self.coefficients[i-1] for i in range(1, len(self.variables)+1)) >= self.lower_bound, f"{self.name} Lower Bound Constraint")
        self.model += (lpSum(self.variables[i] * self.coefficients[i-1] for i in range(1, len(self.variables)+1)) <= self.upper_bound, f"{self.name} Upper Bound Constraint")

class MinimumConstraint:
    def __init__(self, model, variables, coefficients, name, minimum):
        self.model = model
        self.variables = variables
        self.coefficients = coefficients
        self.name = name
        self.minimum = minimum

    def add_constraint(self):
        self.model += (lpSum(self.variables[i] * self.coefficients[i-1] for i in range(1, len(self.variables)+1)) >= self.minimum, f"{self.name} Minimum Constraint")

class EqualityConstraint:
    def __init__(self, model, variables, coefficients, name, value):
        self.model = model
        self.variables = variables
        self.coefficients = coefficients
        self.name = name
        self.value = value

    def add_constraint(self):
        self.model += (lpSum(self.variables[i] * self.coefficients[i-1] for i in range(1, len(self.variables)+1)) == self.value, f"{self.name} Equality Constraint")

class MaximumConstraint:
    def __init__(self, model, variables, coefficients, name, maximum):
        self.model = model
        self.variables = variables
        self.coefficients = coefficients
        self.name = name
        self.maximum = maximum

    def add_constraint(self):
        self.model += (lpSum(self.variables[i] * self.coefficients[i-1] for i in range(1, len(self.variables)+1)) <= self.maximum, f"{self.name} Maximum Constraint")

# Define the model (for maintenance macro + micros)
maintenance_model = LpProblem(name="Maintenance_Diet_Minimum_Cost", sense=LpMinimize)

# Define the decision variables (what we are solving for; quantities of each food)
x = {i: LpVariable(name=f"x_{i}", lowBound=0, upBound=10000) for i in range(1, len(df_filtered)+1)}

def extract_column_array(df, column, scale_factor=1):
    values = df.iloc[:, column].values
    if np.issubdtype(values.dtype, np.number):
        return values / scale_factor
    return values

# Arrays per column extracted from table
dollars_per_g = extract_column_array(df_filtered, 100, 1000)  # compute array of dollars per g for each food
name_per_food = extract_column_array(df_filtered, 102)  # get array of the names of each food (not database name but the specific item from the store)
calories_per_g = extract_column_array(df_filtered, 5, 100)  # get array of caloric values for each food
protein_per_g = extract_column_array(df_filtered, 7, 100)  # get array of amount of protein for each food
net_carbs_per_g = extract_column_array(df_filtered, 95, 100)  # get array of net carbs for each food (this is a computed column I had made; carbs -= fiber = net carbs, because we will use fiber as a separate constraint later)
fiber_per_g = extract_column_array(df_filtered, 10, 100)
fat_per_g = extract_column_array(df_filtered, 6, 100)
protein_quality = extract_column_array(df_filtered, 101)
sugar_per_g = extract_column_array(df_filtered, 99, 100)  # uses the more accurate computed sugar column on column # 99 (ie doesn't count glucose)
satfat_per_g = extract_column_array(df_filtered, 12, 100)

# Micros arrays
calcium_per_g = extract_column_array(df_filtered, 13, 100)
iron_per_g = extract_column_array(df_filtered, 14, 100)
potassium_per_g = extract_column_array(df_filtered, 15, 100)
magnesium_per_g = extract_column_array(df_filtered, 16, 100)
sodium_per_g = extract_column_array(df_filtered, 38, 100)
zinc_per_g = extract_column_array(df_filtered, 39, 100)
copper_per_g = extract_column_array(df_filtered, 40, 100)
manganese_per_g = extract_column_array(df_filtered, 41, 100)
selenium_per_g = extract_column_array(df_filtered, 42, 100)


# Objective Function
maintenance_model += lpSum(x[i] * dollars_per_g[i-1] for i in range(1, len(df_filtered)+1))

# Defining Constraints
# TODO: PULL THESE FROM SOMEWHERE ELSE, MAYBE READ CSV THAT HAS BODYWEIGHT AND OTHER STATS AND COMPUTE THE MAINTENANCE AND OTHER NEEDS BASED ON 
# Macros
maintenance_calories = 2937  # set maintenance calories
protein_minimum = 242  # set protein minimum
net_carb_minimum = 236  # set net carb minimum
fiber_lbound = 30  # lower bound for fiber
fiber_ubound = 60  # upper bound for fiber
fat_minimum = 82
protein_quality_minimum = 0.4  # this is a fraction ie 0.4 = 40%
sugar_maximum = 75
sat_fat_maximum = 0.5  # this is a fraction

# Macro constraints using classes
macro_constraints = [
    EqualityConstraint(maintenance_model, x, calories_per_g, "Maintenance Calories", maintenance_calories),
    MinimumConstraint(maintenance_model, x, protein_per_g, "Protein", protein_minimum),
    MinimumConstraint(maintenance_model, x, net_carbs_per_g, "Net Carb", net_carb_minimum),
    RangeConstraint(maintenance_model, x, fiber_per_g, "Fiber", fiber_lbound, fiber_ubound),
    MinimumConstraint(maintenance_model, x, fat_per_g, "Fat", fat_minimum),
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
manganese_lbound = 15 #NOT USING THIS RIGHT NOW BECAUSE IT WON'T SOLVE; NEED MORE LEAFY FOODS I GUESS?
manganese_ubound = 20
selenium_maximum = 350  # micrograms

# Define constraints
micro_constraints = [
    RangeConstraint(maintenance_model, x, calcium_per_g, "Calcium", calcium_lbound, calcium_ubound),
    RangeConstraint(maintenance_model, x, iron_per_g, "Iron", iron_lbound, iron_ubound),
    MaximumConstraint(maintenance_model, x, potassium_per_g, "Potassium", potassium_maximum),
    RangeConstraint(maintenance_model, x, magnesium_per_g, "Magnesium", magnesium_lbound, magnesium_ubound),
    MaximumConstraint(maintenance_model, x, sodium_per_g, "Sodium", sodium_maximum),
    RangeConstraint(maintenance_model, x, zinc_per_g, "Zinc", zinc_lbound, zinc_ubound),
    MaximumConstraint(maintenance_model, x, copper_per_g, "Copper", copper_maximum),
    MaximumConstraint(maintenance_model, x, selenium_per_g, "Selenium", selenium_maximum)
]

# Add constraints to the model
for constraint in micro_constraints:
    constraint.add_constraints() if isinstance(constraint, RangeConstraint) else constraint.add_constraint()

# Solve the optimization problem
status = maintenance_model.solve()

# Get the results and format them
print(f"status: {maintenance_model.status}, {LpStatus[maintenance_model.status]}")
monthly_cost = round(maintenance_model.objective.value() * 30, 2)
objective_value = round(maintenance_model.objective.value(), 2)
print(f"TOTAL COST OF DAILY DIET CALCULATED: ${objective_value}")
print(f"TOTAL COST OF MONTHLY DIET CALCULATED: ${monthly_cost}")

# Create an empty DataFrame to store the results
results_df = pd.DataFrame(columns=['Food Name', 'Quantity (g)'])

# Collect the data for the DataFrame
results_list = [{'Food Name': name_per_food[i-1], 'Quantity (g)': f"{int(round(var.value(), 0))} g"} for i, var in x.items() if var.value() > 0]

# Convert the list to a DataFrame and sort
results_df = pd.DataFrame(results_list).sort_values(by='Quantity (g)', ascending=False)

# Display the results DataFrame
print("\nCONSISTING OF THE FOLLOWING FOODS:")
for index, row in results_df.iterrows():
    food_name = row['Food Name']
    quantity = row['Quantity (g)']
    print(f"{food_name.ljust(90)}: {quantity}")



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



print(maintenance_model)



#Debugger cell
protein_quality = df_filtered.iloc[:, 101].values

print (protein_quality)


