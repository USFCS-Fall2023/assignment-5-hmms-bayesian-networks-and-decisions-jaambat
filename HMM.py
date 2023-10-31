"""
James Ambat
CS-462
Assignment 5
"""

import random
import argparse
import codecs
import os
import numpy


# observations
class Observation:
    def __init__(self, stateseq, outputseq):
        self.stateseq = stateseq  # sequence of states
        self.outputseq = outputseq  # sequence of outputs

    def __str__(self):
        return ' '.join(self.stateseq) + '\n' + ' '.join(self.outputseq) + '\n'

    def __repr__(self):
        return self.__str__()

    def __len__(self):
        return len(self.outputseq)


# hmm model
class HMM:
    def __init__(self, transitions={}, emissions={}):
        """creates a model from transition and emission probabilities"""
        # Both of these are dictionaries of dictionaries. e.g. :
        # {'#': {'C': 0.814506898514, 'V': 0.185493101486},
        #  'C': {'C': 0.625840873591, 'V': 0.374159126409},
        #  'V': {'C': 0.603126993184, 'V': 0.396873006816}}

        self.transitions = transitions
        self.emissions = emissions

    # part 1 - you do this.
    def load(self, basename: str):
        """
        reads HMM structure from transition (basename.trans), and emission (basename.emit) files, as well as the
        probabilities.
        :param basename: str basename of the transition AND emission file to read from.
        """
        transition_file = None
        emission_file = None
        try:
            transition_txt = basename + ".trans"
            emission_txt = basename + ".emit"

            # Read in the transmission file
            transition_file = open(transition_txt, "r")
            transition_lines = transition_file.readlines()
            transition_lines = [item.strip() for item in transition_lines]

            # Read in the emission file
            emission_file = open(emission_txt, "r")
            emission_lines = emission_file.readlines()
            emission_lines = [item.strip() for item in emission_lines]

            # Process the transmission data into map
            transition_map = {}
            for line in transition_lines:
                tokenized_line = line.split(" ")

                # Each line will have three items, initial state, transmission state, probability
                initial_state = tokenized_line[0]
                transmission_state = tokenized_line[1]
                probability = float(tokenized_line[2])

                if transition_map.get(initial_state) is None:
                    transition_map[initial_state] = {}

                transition_map[initial_state][transmission_state] = probability

            # Save the transmission map
            self.transitions = transition_map

            # Process the emission data into map
            emission_map = {}
            for line in emission_lines:
                tokenized_line = line.split(" ")
                # There are 3 items in the tokenized line: observation, conditional, probability
                observation = tokenized_line[0]
                conditional = tokenized_line[1]
                probability = float(tokenized_line[2])

                if emission_map.get(observation) is None:
                    emission_map[observation] = {}

                emission_map[observation][conditional] = probability

            self.emissions = emission_map
        except FileNotFoundError:
            raise FileNotFoundError("The specified transmission and emission files could not be found.")
        except OSError:
            raise OSError("There was an issue with reading the transmission and emission files.")
        finally:
            transition_file.close()
            emission_file.close()

    # you do this.
    def generate(self, n: int) -> list[str]:
        """
        return an n-length observation by randomly sampling from this HMM.
        :param n: int length of the number of observations.
        """
        states = [None for _ in range(n + 1)]

        # Acquire a list of state. The first state must be a starting state
        starting_state_map = self.transitions["#"]
        random_choice = random.choice(list(starting_state_map))
        states[0] = random_choice

        for i in range(1, len(states)):
            previous_state = states[i - 1]

            state_map = self.transitions[previous_state]
            random_choice = random.choice(list(state_map))
            states[i] = random_choice

        # For each of the states get select a random observation from the emissions
        observations = [None for _ in range(n)]

        for i in range(len(states) - 1):
            curr_state = states[i]

            emission = self.emissions[curr_state]
            observation = random.choice(list(emission))
            observations[i] = observation

        return observations

    def forward(self, observation_sequence: list[str]) -> str:
        """
        Forward algorithm determines the most likely final state from a sequence of observations.
        :param observation_sequence: list[str] of observations from which to determine the final state.
        :return: str of the final state
        """

        # Acquire starting probabilities
        starting_probabilities_map = None
        for item in self.transitions.items():
            if item[0] == "#":
                starting_probabilities_map = item[1]
                break

        # Initialize forward matrix, each row will be labeled by the state name as a key
        column_names = ["-"]
        column_names.extend(observation_sequence)
        forward_matrix = {}
        for item in starting_probabilities_map.items():
            if forward_matrix.get(item[0]) is None:
                # Initialize each row in the matrix and add column for starting probabilities
                forward_matrix[item[0]] = [None for _ in range(len(observation_sequence) + 1)]
                forward_matrix[item[0]][0] = item[1]

        # Loop through each column and fill in probabilities in forward_matrix[i, j]
        for j in range(1, len(column_names)):
            curr_observation = column_names[j]

            p_curr_state_given_curr_observation = 0

            # Calculate P(curr_state | e_n, .... e1) for each of the states
            for item in forward_matrix.items():
                curr_state = item[0]

                if j == 1:
                    p_curr_observation_given_curr_state = self.emissions[curr_state][curr_observation]

                    matrix_row = item[1]
                    p_curr_state = matrix_row[0]

                    # Apply Bayes rule and record the entry into the forward_matrix
                    p_curr_state_given_curr_observation = p_curr_observation_given_curr_state * p_curr_state
                    forward_matrix[curr_state][j] = p_curr_state_given_curr_observation
                    continue

                # For all possible states, acquire P(curr_state | e_n, .... e1) from the emissions and transitions
                for state in forward_matrix.items():
                    state_i = state[0]

                    p_curr_observation_given_state_i = self.emissions[state_i][curr_observation]
                    p_curr_state_given_state_i = self.transitions[state_i][curr_state]
                    p_curr_state_prev_column = forward_matrix[state_i][j - 1]

                    # Apply Bayes rule for each state we loop through.
                    p_curr_state_given_curr_observation += (p_curr_observation_given_state_i *
                                                            p_curr_state_given_state_i *
                                                            p_curr_state_prev_column)

                # Update the forward matrix AFTER applying the emissions and transitions for every state
                forward_matrix[curr_state][j] = p_curr_state_given_curr_observation

        # Inspect the last column of the forward_matrix to see which of the states has the highest probability
        last_column_index = len(column_names) - 1
        last_column = [(item[0], item[1][last_column_index]) for item in forward_matrix.items()]

        entry_name = "None"
        entry_value = 0
        for entry in last_column:
            if entry[1] > entry_value:
                entry_value = entry[1]
                entry_name = entry[0]

        return entry_name

    # you do this: Implement the Viterbi alborithm. Given an Observation (a list of outputs or emissions)
    # determine the most likely sequence of states.

    def viterbi(self, observation):
        """given an observation,
        find and return the state sequence that generated
        the output sequence, using the Viterbi algorithm.
        """


print("----------------------------------------------------\n"
      "PART 1 - implement load()                           \n"
      "----------------------------------------------------\n")
hidden_markov = HMM()
hidden_markov.load("two_english")
print("-[x] Transition map: ")
print(hidden_markov.transitions)

print("\n-[x] Emission map first 100 chars for preview: ")
print(str(hidden_markov.emissions)[:100] + "...")

print("\n"
      "----------------------------------------------------\n"
      "PART 1 - implement generate() n observations        \n"
      "----------------------------------------------------\n")
observation_list = hidden_markov.generate(15)
print(observation_list)

print("\n"
      "----------------------------------------------------\n"
      "PART 1 - implement forward()                        \n"
      "----------------------------------------------------\n")
most_likely_final_state = hidden_markov.forward(observation_list)
print("Most likely final state:  " + most_likely_final_state)