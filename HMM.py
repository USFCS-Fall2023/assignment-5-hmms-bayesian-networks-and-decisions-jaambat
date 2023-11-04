"""
James Ambat
CS-462
Assignment 5
"""

import random
import argparse
import codecs
import os
import typing

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
    def generate(self, n: int) -> typing.Tuple[list[str], list[str]]:
        """
        return an n-length observation by randomly sampling from this HMM and using the probabilities as weights
        from which to select random samples.
        :param n: int length of the number of observations.
        :return: two lists list[str] and list[str] of the states and the actual observations.
        """
        # Acquire a list of states by randomly selecting the states while using the probabilities to make the selection.
        starting_state_map = self.transitions["#"]
        possible_states = list(starting_state_map)
        weights_per_state = [float(0) for _ in range(len(possible_states))]

        for i in range(len(weights_per_state)):
            curr_state = possible_states[i]
            weights_per_state[i] = starting_state_map[curr_state]

        states = random.choices(population=possible_states, weights=weights_per_state, k=n)

        # For each state, get a random observation from the emissions while using probabilities to make the selection
        observations = ["" for _ in range(n)]

        for i in range(len(states)):
            curr_state = states[i]

            emission_map = self.emissions[curr_state]
            possible_emissions = list(emission_map)

            weights_per_emission = [float(0) for _ in range(len(possible_emissions))]
            for j in range(len(possible_emissions)):
                curr_emission = possible_emissions[j]
                weights_per_emission[j] = emission_map[curr_emission]

            observation = random.choices(population=possible_emissions, weights=weights_per_emission, k=1)[0]
            observations[i] = observation

        return states, observations

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
        column_names.extend(observation_sequence)  # column names: ["-", "observation_1", "observation_2", ... ]
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
      "PART 1 - implement load().                          \n"
      "NOT COMMAND LINE ARG                                \n"        
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
      "NOT COMMAND LINE ARG                                \n"        
      "----------------------------------------------------\n")
_, observation_list = hidden_markov.generate(15)
print(observation_list)

print("\n"
      "----------------------------------------------------\n"
      "PART 1 - implement forward()                        \n"
      "NOT COMMAND LINE ARG                                \n"        
      "----------------------------------------------------\n")
most_likely_final_state = hidden_markov.forward(observation_list)
print("Most likely final state:  " + most_likely_final_state)


# Supporting command line argument calls
parser = argparse.ArgumentParser(description="Processes routines the evaluate properties of hidden markov models "
                                             "for a given filename and argument flags")
parser.add_argument("file", type=str, help="Path of the file to process subroutines from.")

# Add argument flag for --generate and only allow this flag in a command line prompt with mutually exclusive
generate_flag = parser.add_mutually_exclusive_group()
generate_flag.add_argument("--generate",
                           type=int,
                           help="Generates n number of random observations as specified in the command line argument.")

cli_args = parser.parse_args()

if __name__ == "__main__":
    print("\n"
          "----------------------------------------------------\n"
          "PART 1 - COMMAND LINE ARG OUTPUT                    \n"
          "----------------------------------------------------\n")

    print(cli_args)

    if cli_args.generate is not None and cli_args.file is not None:
        file = cli_args.file
        num_observations = cli_args.generate
        hidden_markov = HMM()
        hidden_markov.load(file)
        state_list, observation_list = hidden_markov.generate(num_observations)
        print("The observation list from python3 hmm.py two_english --generate [  n  ]")

        print(*state_list)
        print(*observation_list)
