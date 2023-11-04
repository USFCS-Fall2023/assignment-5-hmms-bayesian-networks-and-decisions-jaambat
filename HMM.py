"""
James Ambat
CS-462
Assignment 5
"""

import random
import argparse
import codecs
import os
import sys
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

            # Calculate P(curr_state | e_n, .... e1) for each of the states
            for item in forward_matrix.items():
                curr_state = item[0]
                p_curr_state_given_curr_observation = 0

                if j == 1:
                    p_curr_observation_given_curr_state = None
                    try:
                        p_curr_observation_given_curr_state = self.emissions[curr_state][curr_observation]
                    except KeyError:
                        # Case where there is no such state for the current observation.
                        # "i" can never be ADV based on the supplied emissions
                        p_curr_observation_given_curr_state = 0

                    matrix_row = item[1]
                    p_curr_state = matrix_row[0]

                    # Apply Bayes rule and record the entry into the forward_matrix
                    p_curr_state_given_curr_observation = p_curr_observation_given_curr_state * p_curr_state
                    forward_matrix[curr_state][j] = p_curr_state_given_curr_observation
                    continue

                # For all possible states, acquire P(curr_state | e_n, .... e1) from the emissions and transitions
                for state in forward_matrix.items():
                    state_i = state[0]

                    p_curr_observation_given_curr_state = None
                    try:
                        p_curr_observation_given_curr_state = self.emissions[curr_state][curr_observation]
                    except KeyError:
                        # Case where there is no such observation for the given state
                        p_curr_observation_given_curr_state = 0

                    p_curr_state_given_state_i = self.transitions[state_i][curr_state]
                    p_state_i_prev_column = forward_matrix[state_i][j - 1]

                    # Apply Bayes rule for each state we loop through.
                    curr_bayes_rule = (p_curr_observation_given_curr_state
                                       * p_curr_state_given_state_i
                                       * p_state_i_prev_column)

                    p_curr_state_given_curr_observation += curr_bayes_rule

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

    # you do this: Implement the Viterbi algorithm. Given an Observation (a list of outputs or emissions)
    # determine the most likely sequence of states.

    def viterbi(self, observations: list[str]) -> list[str]:
        """
        Determines the most likely sequence of states given a list of observations.
        :param observations: list[str] of a sequence of observations
        :return: list[str]
        """

        # Acquire starting probabilities, and prepared indexes for the starting state back pointers
        starting_probabilities_map = None
        for item in self.transitions.items():
            if item[0] == "#":
                starting_probabilities_map = item[1]
                break

        state_index_map = {}
        i = 1
        for item in list(starting_probabilities_map):
            state_index_map[item] = i
            i += 1

        # Reference the state names from indexes when needed in back_pointer traversal
        index_state_map = {}
        for key, value in state_index_map.items():
            index_state_map[value] = key

        # Initialize viterbi matrix and back pointer matrix, each row will be labeled by the state name as a key
        column_names = ["-"]
        column_names.extend(observations)  # column names: ["-", "observation_1", "observation_2", ... ]
        viterbi_matrix = {}
        back_pointer_matrix = {}
        for item in starting_probabilities_map.items():
            if viterbi_matrix.get(item[0]) is None:
                # Initialize each row in the viterbi matrix and add column for starting probabilities.
                viterbi_matrix[item[0]] = [None for _ in range(len(observations) + 1)]
                viterbi_matrix[item[0]][0] = item[1]

                # Initialize each row in back pointer matrix and leave the first column empty.
                back_pointer_matrix[item[0]] = [None for _ in range(len(observations) + 1)]
                back_pointer_matrix[item[0]][0] = None

        # ACQUIRE MAX OF BAYES RULE, loop through each column and fill in viterbi_matrix[i, j] and back_pointer_matrix
        for j in range(1, len(column_names)):
            curr_observation = column_names[j]

            # Calculate P(curr_state | e_n, .... e1) for each of the states
            for item in viterbi_matrix.items():
                curr_state = item[0]
                p_curr_state_given_curr_observation = 0

                if j == 1:
                    p_curr_observation_given_curr_state = None
                    try:
                        p_curr_observation_given_curr_state = self.emissions[curr_state][curr_observation]
                    except KeyError:
                        # Case where there is no such state for the current observation.
                        # "i" can never be ADV based on the supplied emissions
                        p_curr_observation_given_curr_state = 0

                    matrix_row = item[1]
                    p_curr_state = matrix_row[0]

                    # Apply Bayes rule and record the entry into the viterbi_matrix
                    p_curr_state_given_curr_observation = p_curr_observation_given_curr_state * p_curr_state
                    viterbi_matrix[curr_state][j] = p_curr_state_given_curr_observation

                    # Initialize entry in the first column of back pointer matrix with 0
                    back_pointer_matrix[curr_state][j] = 0
                    continue

                # For all possible states, acquire P(curr_state | e_n, .... e1) from the emissions and transitions
                bayes_rule_per_state = []
                for state in viterbi_matrix.items():
                    state_i = state[0]

                    p_curr_observation_given_curr_state = None
                    try:
                        p_curr_observation_given_curr_state = self.emissions[curr_state][curr_observation]
                    except KeyError:
                        # Case where there is no such observation for the given state
                        p_curr_observation_given_curr_state = 0

                    p_curr_state_given_state_i = self.transitions[state_i][curr_state]
                    p_state_i_prev_column = viterbi_matrix[state_i][j - 1]

                    # Apply Bayes rule for each state we loop through.
                    curr_bayes_rule = (p_curr_observation_given_curr_state
                                       * p_curr_state_given_state_i
                                       * p_state_i_prev_column)

                    # Save the Bayes Rule output for getting the max AFTER all states have been processed.
                    bayes_rule_per_state.append({state_i: curr_bayes_rule})

                # Acquire the maximum Bayes Rule output
                max_state_name = "None"
                max_state_bayes_rule_value = 0
                for entry in bayes_rule_per_state:
                    state_name = list(entry.keys())[0]
                    curr_bayes_rule_value = entry[state_name]

                    if curr_bayes_rule_value > max_state_bayes_rule_value:
                        max_state_name = state_name
                        max_state_bayes_rule_value = curr_bayes_rule_value

                # Update viterbi matrix
                p_curr_state_given_curr_observation = max_state_bayes_rule_value
                viterbi_matrix[curr_state][j] = p_curr_state_given_curr_observation

                # Update back_pointer matrix
                back_pointer_matrix[curr_state][j] = state_index_map[max_state_name]

        # viterbi and back_pointer matrix filled, walk back from back to front of back_pointer
        # Inspect last column and get state with the highest probability
        last_column_index = len(column_names) - 1
        last_column = [(item[0], item[1][last_column_index]) for item in viterbi_matrix.items()]

        entry_name = "None"
        entry_value = 0
        for entry in last_column:
            if entry[1] > entry_value:
                entry_value = entry[1]
                entry_name = entry[0]

        # Reference the state with the highest probability and traverse the back_pointer matrix
        predicted_state_index = back_pointer_matrix[entry_name][last_column_index]
        predicted_sequence_indexes = [None for _ in range(len(column_names))]
        predicted_sequence_indexes[last_column_index] = predicted_state_index

        predicted_sequence_states = [None for _ in range(len(column_names))]
        predicted_sequence_states[last_column_index] = index_state_map[predicted_state_index]

        # Traverse the remaining columns
        for j in range(last_column_index - 1, 0, -1):  # Stop before index 0, nothing in first column
            predicted_state_index = predicted_sequence_indexes[j + 1]  # Pull from the previous column

            predicted_state_name = index_state_map[predicted_state_index]
            predicted_state_index = back_pointer_matrix[predicted_state_name][j]

            if predicted_state_index == 0:
                # If 0, just use the previous state_name as the indicator of state at time 1.
                predicted_sequence_states[j] = predicted_state_name
            else:
                predicted_sequence_indexes[j] = predicted_state_index
                predicted_sequence_states[j] = index_state_map[predicted_state_index]

        predicted_sequence_states = [item for item in predicted_sequence_states
                                     if item is not None]

        return predicted_sequence_states

    def read_in_observations(self, observation_file_name: str):
        """
        Reads-in the observations from a file and produces sequences of observations for each line.
        These sequence will be used by the hidden markov routines in forward and viterbi.
        :param observation_file_name: file name to read in sequences of observations.
        :return: list[list[str]] where each row is a list[str]. The list[str] is a sequence of observations.
        """
        observation_file = None
        observation_lines = None

        try:
            observation_file = open(observation_file_name, "r")
            observation_lines = observation_file.readlines()
            observation_lines = [item.strip() for item in observation_lines  # Remove the white space
                                 if len(item.strip()) > 0]  # Skip adding blank lines to the list
        except FileNotFoundError:
            raise FileNotFoundError("The specified observation file could not be found.")
        except OSError:
            raise OSError("There was an issue with reading the observation file.")
        finally:
            observation_file.close()

        return [item.split(" ") for item in observation_lines]


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
_, observations_list = hidden_markov.generate(15)
print(observations_list)

print("\n"
      "----------------------------------------------------\n"
      "PART 1 - implement forward()                        \n"
      "NOT COMMAND LINE ARG                                \n"        
      "----------------------------------------------------\n")
most_likely_final_state = hidden_markov.forward(observations_list)
print("Most likely final state:  " + most_likely_final_state)

print("\n"
      "----------------------------------------------------\n"
      "PART 1 - Testing my implementation of               \n"
      "read_in_observations()                              \n"
      "NOT COMMAND LINE ARG                                \n"        
      "----------------------------------------------------\n")
observation_list_from_file = hidden_markov.read_in_observations(observation_file_name="ambiguous_sents.obs")
print(observation_list_from_file)

# Supporting command line argument calls
parser = argparse.ArgumentParser(description="Processes routines the evaluate properties of hidden markov models "
                                             "for a given filename and argument flags")
parser.add_argument("file", type=str, help="Path of the file to process subroutines from.")

# Add argument flag for --generate and only allow this flag in a command line prompt with mutually exclusive
generate_flag = parser.add_mutually_exclusive_group()
generate_flag.add_argument("--generate",
                           type=int,
                           help="Generates n number of random observations as specified in the command line argument.")

# Add argument flag for --forward and only allow this flag with mutually exclusive
forward_flag = parser.add_mutually_exclusive_group()
generate_flag.add_argument("--forward",
                           type=str,
                           help="Runs the forward algorithm on a file of observations and for each observation, "
                                "the algorithm will provide the most likely final state.")

# Add argument flag for --viterbi and only allow this flag with mutually exclusive
viterbi_flag = parser.add_mutually_exclusive_group()
generate_flag.add_argument("--viterbi",
                           type=str,
                           help="Runs the viterbi algorithm on a file of observations and for each observation, "
                                "the algorithm will determine the most likely sequence of states.")
cli_args = None

try:
    cli_args = parser.parse_args()
except IndexError:
    print("Command line arguments were not provided.")
    sys.exit()

if __name__ == "__main__":
    print("\n"
          "----------------------------------------------------\n"
          "PART 1 - COMMAND LINE ARG OUTPUT                    \n"
          "----------------------------------------------------\n")

    print(cli_args)
    file = None
    if cli_args.file is not None:
        file = cli_args.file
        print(file)

    hidden_markov = HMM()
    hidden_markov.load(file)

    if cli_args.generate is not None:
        # Supports generate() in the command line
        num_observations = cli_args.generate
        state_list, observations_list = hidden_markov.generate(num_observations)
        print("The observation list from python3 hmm.py two_english --generate [  n  ]")

        print(*state_list)
        print(*observations_list)

    if cli_args.forward is not None and cli_args.file is not None:
        # Supports forward() in the command line
        observation_file_name = cli_args.forward
        observations_list = hidden_markov.read_in_observations(observation_file_name=observation_file_name)
        print("** Observations from file [  %s  ] **\n%s\n" % (observation_file_name, observations_list))

        for observations in observations_list:
            # Take the observation sequences from each row and run forward on them
            most_likely_final_state = hidden_markov.forward(observation_sequence=observations)
            print(most_likely_final_state)

    if cli_args.viterbi is not None and cli_args.file is not None:
        # Supports viterbi() in the command line
        observation_file_name = cli_args.viterbi
        observations_list = hidden_markov.read_in_observations(observation_file_name=observation_file_name)

        for observations in observations_list:
            # Take the observation sequences from each row and run viterbi on them
            predicted_states = hidden_markov.viterbi(observations=observations)
            print(*predicted_states)

