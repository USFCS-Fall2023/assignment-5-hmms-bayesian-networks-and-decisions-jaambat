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
        :param basename: str basename of the transmission AND emission file to read from.
        """
        transmission_file = None
        emission_file = None
        try:
            transmission_txt = basename + ".trans"
            emission_txt = basename + ".emit"

            # Read in the transmission file
            transmission_file = open(transmission_txt, "r")
            transmission_lines = transmission_file.readlines()
            transmission_lines = [item.strip() for item in transmission_lines]

            # Read in the emission file
            emission_file = open(emission_txt, "r")
            emission_lines = emission_file.readlines()
            emission_lines = [item.strip() for item in emission_lines]

            # Process the transmission data into map
            transmission_map = {}
            for line in transmission_lines:
                tokenized_line = line.split(" ")

                # Each line will have three items, initial state, transmission state, probability
                initial_state = tokenized_line[0]
                transmission_state = tokenized_line[1]
                probability = float(tokenized_line[2])

                if transmission_map.get(initial_state) is None:
                    transmission_map[initial_state] = {}

                transmission_map[initial_state][transmission_state] = probability

            # Save the transmission map
            self.transitions = transmission_map

            # Process the transmission data into map
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
            raise OSError("There was an issue with reading the tranmission and emission files.")
        finally:
            transmission_file.close()
            emission_file.close()

        print()

    # you do this.
    def generate(self, n):
        """return an n-length observation by randomly sampling from this HMM."""

    # you do this: Implement the Viterbi alborithm. Given an Observation (a list of outputs or emissions)
    # determine the most likely sequence of states.

    def viterbi(self, observation):
        """given an observation,
        find and return the state sequence that generated
        the output sequence, using the Viterbi algorithm.
        """


hidden_markov = HMM()
hidden_markov.load("two_english")