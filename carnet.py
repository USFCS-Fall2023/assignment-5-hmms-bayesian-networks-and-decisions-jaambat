"""
James Ambat
CS-462
Assignment 5
"""

from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination

car_model = BayesianNetwork(
    [
        ("Battery", "Radio"),
        ("Battery", "Ignition"),
        ("Ignition", "Starts"),
        ("Gas", "Starts"),
        ("Starts", "Moves")
    ]
)

# Defining the parameters using CPT
from pgmpy.factors.discrete import TabularCPD

cpd_battery = TabularCPD(
    variable="Battery", variable_card=2, values=[[0.70], [0.30]],
    state_names={"Battery": ["Works", "Doesn't work"]},
)

cpd_gas = TabularCPD(
    variable="Gas", variable_card=2, values=[[0.40], [0.60]],
    state_names={"Gas": ["Full", "Empty"]},
)

cpd_radio = TabularCPD(
    variable="Radio", variable_card=2,
    values=[[0.75, 0.01], [0.25, 0.99]],
    evidence=["Battery"],
    evidence_card=[2],
    state_names={"Radio": ["turns on", "Doesn't turn on"],
                 "Battery": ["Works", "Doesn't work"]}
)

cpd_ignition = TabularCPD(
    variable="Ignition", variable_card=2,
    values=[[0.75, 0.01], [0.25, 0.99]],
    evidence=["Battery"],
    evidence_card=[2],
    state_names={"Ignition": ["Works", "Doesn't work"],
                 "Battery": ["Works", "Doesn't work"]}
)

cpd_starts = TabularCPD(
    variable="Starts",
    variable_card=2,
    values=[[0.95, 0.05, 0.05, 0.001], [0.05, 0.95, 0.95, 0.9999]],
    evidence=["Ignition", "Gas"],
    evidence_card=[2, 2],
    state_names={"Starts": ["yes", "no"], "Ignition": ["Works", "Doesn't work"], "Gas": ["Full", "Empty"]},
)

cpd_moves = TabularCPD(
    variable="Moves", variable_card=2,
    values=[[0.8, 0.01], [0.2, 0.99]],
    evidence=["Starts"],
    evidence_card=[2],
    state_names={"Moves": ["yes", "no"],
                 "Starts": ["yes", 'no']}
)


# Associating the parameters with the model structure
car_model.add_cpds(cpd_starts, cpd_ignition, cpd_gas, cpd_radio, cpd_battery, cpd_moves)

car_infer = VariableElimination(car_model)

print(car_infer.query(variables=["Moves"], evidence={"Radio": "turns on", "Starts": "yes"}))

print("\n"
      "--------------------------------------------------------\n"
      "PART 3.2 - Given that the car will not move, what is    \n"
      "the probability that the battery is not working?        \n"
      "--------------------------------------------------------\n")
print(car_infer.query(variables=["Battery"], evidence={"Moves": "no"}))
print("\n Probability the battery is not working: 0.3590")


print("\n"
      "--------------------------------------------------------\n"
      "PART 3.2 - Given that the radio is not working, what is \n"
      "the probability that the car will not start?            \n"
      "--------------------------------------------------------\n")
print(car_infer.query(variables=["Starts"], evidence={"Radio": "Doesn't turn on"}))
print("\n Probability the car will not start: 0.8687")

print("\n"
      "--------------------------------------------------------\n"
      "PART 3.2 - Given that the battery is working, does the  \n"
      "probability of the radio working change if we discover  \n"
      "that the car has gas in it?                             \n"
      "--------------------------------------------------------\n")
print("WITHOUT KNOWLEDGE OF GAS:  ")
print(car_infer.query(variables=["Radio"], evidence={"Battery": "Works"}))

print("\nDISCOVERED CAR HAS GAS:  ")
print(car_infer.query(variables=["Radio"], evidence={"Battery": "Works", "Gas": "Full"}))

print("** No. The Probability of the Radio working DOES NOT change if we discover there is gas in the car. **")

print("\n"
      "--------------------------------------------------------\n"
      "PART 3.2 - Given that the car doesn't move, how does the\n"
      "probability of the ignition failing change if we observe\n"
      "that the car does not have gas in it?                   \n"
      "--------------------------------------------------------\n")
print("WITHOUT OBSERVING THAT THE CAR DOES NOT GAS:  ")
print(car_infer.query(variables=["Ignition"], evidence={"Moves": "no"}))

print("\nOBSERVING THAT THE CAR DOES NOT GAS:  ")
print(car_infer.query(variables=["Ignition"], evidence={"Moves": "no", "Gas": "Empty"}))

print("** The probability of Ignition failing goes down from 0.5666 to 0.4822 **")

print("\n"
      "--------------------------------------------------------\n"
      "PART 3.2 - What is the probability that the car starts  \n"
      "if the radio works and it has gas in it?                \n"
      "--------------------------------------------------------\n")
print(car_infer.query(variables=["Starts"], evidence={"Radio": "turns on", "Gas": "Full"}))
print("** The probability that the car starts: 0.7212 **")

# Part 3.3
# Remaking the Bayesian Network so that previous probabilities are not affected.
car_model = BayesianNetwork(
    [
        ("Battery", "Radio"),
        ("Battery", "Ignition"),
        ("Ignition", "Starts"),
        ("Gas", "Starts"),
        ("Starts", "Moves"),
        ("KeyPresent", "Starts")  # Added the KeyPresent to Starts Edge
    ]
)

# Added the KeyPresent Node
cpd_key_present = TabularCPD(
    variable="KeyPresent", variable_card=2, values=[[0.70], [0.30]],
    state_names={"KeyPresent": ["yes", "no"]},
)

cpd_battery = TabularCPD(
    variable="Battery", variable_card=2, values=[[0.70], [0.30]],
    state_names={"Battery": ["Works", "Doesn't work"]},
)

cpd_gas = TabularCPD(
    variable="Gas", variable_card=2, values=[[0.40], [0.60]],
    state_names={"Gas": ["Full", "Empty"]},
)

cpd_radio = TabularCPD(
    variable="Radio", variable_card=2,
    values=[[0.75, 0.01], [0.25, 0.99]],
    evidence=["Battery"],
    evidence_card=[2],
    state_names={"Radio": ["turns on", "Doesn't turn on"],
                 "Battery": ["Works", "Doesn't work"]}
)

cpd_ignition = TabularCPD(
    variable="Ignition", variable_card=2,
    values=[[0.75, 0.01], [0.25, 0.99]],
    evidence=["Battery"],
    evidence_card=[2],
    state_names={"Ignition": ["Works", "Doesn't work"],
                 "Battery": ["Works", "Doesn't work"]}
)

# Updated Starts Node
cpd_starts = TabularCPD(
    variable="Starts",
    variable_card=2,
    values=[[0.99, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01], [0.01, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99]],
    evidence=["Ignition", "Gas", "KeyPresent"],
    evidence_card=[2, 2, 2],
    state_names={"Starts": ["yes", "no"], "Ignition": ["Works", "Doesn't work"], "Gas": ["Full", "Empty"],
                 "KeyPresent": ["yes", "no"]},
)

cpd_moves = TabularCPD(
    variable="Moves", variable_card=2,
    values=[[0.8, 0.01], [0.2, 0.99]],
    evidence=["Starts"],
    evidence_card=[2],
    state_names={"Moves": ["yes", "no"],
                 "Starts": ["yes", 'no']}
)

car_model.add_cpds(cpd_starts, cpd_ignition, cpd_gas, cpd_radio, cpd_battery, cpd_moves, cpd_key_present)
car_infer = VariableElimination(car_model)

print("\n"
      "-----------------------------------------\n"
      "P(starts | gas, ignition, keyPresent)    \n"
      "-----------------------------------------\n")
print(car_infer.query(variables=["Starts"], evidence={"Gas": "Full", "Ignition": "Works", "KeyPresent": "yes"}))

print("\n"
      "-----------------------------------------\n"
      "P(starts | gas, !ignition, keyPresent)    \n"
      "-----------------------------------------\n")
print(car_infer.query(variables=["Starts"], evidence={"Gas": "Full", "Ignition": "Doesn't work", "KeyPresent": "yes"}))

print("\n"
      "-----------------------------------------\n"
      "P(starts | !gas, ignition, keyPresent)    \n"
      "-----------------------------------------\n")
print(car_infer.query(variables=["Starts"], evidence={"Gas": "Empty", "Ignition": "Works", "KeyPresent": "yes"}))

print("\n"
      "-----------------------------------------\n"
      "P(starts | gas, ignition, !keyPresent)    \n"
      "-----------------------------------------\n")
print(car_infer.query(variables=["Starts"], evidence={"Gas": "Full", "Ignition": "Works", "KeyPresent": "no"}))

print("\n"
      "-----------------------------------------\n"
      "P(starts | !gas, !ignition, keyPresent)    \n"
      "-----------------------------------------\n")
print(car_infer.query(variables=["Starts"], evidence={"Gas": "Empty", "Ignition": "Doesn't work", "KeyPresent": "yes"}))

print("\n"
      "-----------------------------------------\n"
      "P(starts | !gas, ignition, !keyPresent)  \n"
      "-----------------------------------------\n")
print(car_infer.query(variables=["Starts"], evidence={"Gas": "Empty", "Ignition": "Works", "KeyPresent": "no"}))

print("\n"
      "-----------------------------------------\n"
      "P(starts | gas, !ignition, !keyPresent)  \n"
      "-----------------------------------------\n")
print(car_infer.query(variables=["Starts"], evidence={"Gas": "Full", "Ignition": "Doesn't work", "KeyPresent": "no"}))

print("\n"
      "-----------------------------------------\n"
      "P(starts | !gas, !ignition, !keyPresent)  \n"
      "-----------------------------------------\n")
print(car_infer.query(variables=["Starts"], evidence={"Gas": "Empty", "Ignition": "Doesn't work", "KeyPresent": "no"}))

