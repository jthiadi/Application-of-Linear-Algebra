!pip install nltk
!pip install numpy

import re
from nltk import ngrams
import numpy as np
from typing import List, Tuple, Union, Dict
from itertools import product

def words_to_index(words: List[str], base: int, unique_words: Dict[str, int]) -> int:
    """
        Convert a list of words to an index representation.

        Args:
            words (List[str]): The list of words to be converted.
            base (int): The base value used for conversion.
            unique_words (Dict[str, int]): A dictionary mapping unique words to their corresponding indices.

        Returns:
            - int: The index representation of the given list of words.

        Example:
            >>> words_to_index(["the", "quick", "brown", "fox"], 10, {"the": 0, "quick": 1, "brown": 2, "fox": 3})
            123
    """
    length = len(words)
    numbers = [unique_words[word] for word in words]
    return sum([num * (base ** (length - 1 - idx)) for idx, num in enumerate(numbers)])

def index_to_words(index: int, base: int, length: int, unique_words: Dict[str, int]):
    """
        Converts an index to a list of words based on a given base and unique words.

        Args:
            index (int): The index to convert.
            base (int): The base used for conversion.
            length (int): The length of the resulting list.
            unique_words (Dict[str, int]): A dictionary mapping unique words to their corresponding indices.

        Returns:
            List[str]: A list of words corresponding to the given index.

        Examples:
            >>> unique_words = {'apple': 0, 'banana': 1, 'cherry': 2}
            >>> index_to_words(5, 3, 2, unique_words)
            ['banana', 'cherry'] (Because 5 = 1 * 3^1 + 2 * 2^0)

            >>> unique_words = {'red': 0, 'green': 1, 'blue': 2}
            >>> index_to_words(2, 3, 1, unique_words)
            ['blue'] (Because 2 = 2 * 3^0)
    """
    numbers = []

    unique_words_list = list(unique_words.keys())
    for pow in range(length - 1, -1, -1):
        numbers.append(index // (base ** pow))
        index -= numbers[-1] * (base ** pow)

    return [unique_words_list[num] for num in numbers]

    def remove_unnecessary_characters(text: str):
    """
        Removes unnecessary characters from the given text and converts it to lowercase.

        Args:
            text (str): The input text to be processed.
        Returns:
            str: The processed text with unnecessary characters removed and converted to lowercase.
    """
    # Remove unnecessary characters
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)

    # Convert to lowercase
    text = text.lower()

    return text

    def generate_n_grams(text: str, n: int):
    """
        Generate n-grams from the given text.

        Args:
            text (str): The input text from which n-grams will be generated.
            n (int): The number of consecutive words in each n-gram.
        Returns:
            List[Tuple[str]]: A list of tuples representing the generated n-grams.
    """
    # [TODO] Generate n-grams, using the ngrams function from the nltk library
    n_grams = ngrams(text.split(), n)

    # Convert to list of tuples
    n_grams = list(n_grams)

    return n_grams

    def compute_transition_matrix(n_grams: List[Tuple[str]]):
    """
        Computes the transition matrix and unique states for a given list of n-grams.

        Args:
            n_grams (List[Tuple[str]]): A list of n-grams, where each n-gram is a tuple of strings.
        Returns:
            np.ndarray: transition matrix.
            Dict[str, int]: A dictionary mapping unique words to their corresponding indices.
    """

    # Get the value of n
    n = len(n_grams[0])

    ## Step 3.1: Collect all possible words (label words with indices)

    unique_words = {}
    unique_words_count = 0

    # A helper function to add a word to the unique_words dictionary, if it is not already present
    def _add_to_unique_word(word: str):
        # We need to increase the counter outside this function
        nonlocal unique_words_count

        # Check if the state is already in the unique_states dictionary
        if unique_words.get(word, None) is None:
            unique_words[word] = unique_words_count
            unique_words_count += 1

    # Iterate over all n-grams to collect all possible words
    for n_gram in n_grams:
        for word in n_gram:
            # [TODO] Add the word to the unique_words dictionary
            # hint: use the _add_to_unique_word function
            _add_to_unique_word(word)
            continue


    ## Step 3.2: Compute the transition matrix

    # In this part, we use `words_to_index` and `index_to_words` functions to convert between state name (words) and indices

    # Example:
    # If we have 3 unique words: 0 for "apple", 1 for "banana", 2 for "cherry", and n = 3,
    # then the state names are "apple, apple", "apple, banana", "apple, cherry", "banana, apple", "banana, banana", and so on.
    # In this case, the mapping from states to indices would be:
    # "(apple, apple)" <-> 0, "(apple, banana)" <-> 1, "(apple, cherry)" <-> 2, "(banana, apple)" <-> 3, and so on.

    # [TODO] Compute the number of states in Markov chain. Number of states is the number of unique words to the power of n-1
    unique_states_count = unique_words_count ** (n - 1)

    # [TODO] Create transition matrix, filled with zeros (the size of the transition matrix is # states x # states)
    transition_matrix = np.zeros((unique_states_count, unique_states_count))

    # Count the number of transitions from each state to another state
    for n_gram in n_grams:
        # [TODO] Find the indices of the states (from and to) (i.e. the first n-1 words and the last n-1 words)
        # hint: use words_to_index function
        state_from_index = words_to_index(n_gram[:-1], unique_words_count, unique_words)
        state_to_index = words_to_index(n_gram[1:], unique_words_count, unique_words)

        # Increment the count of the transition from state_from to state_to
        transition_matrix[state_from_index][state_to_index] += 1

    # Special case:
    # if the sum of number of transition of a state is zero, which means it has no information about the next state,
    # we set the probabilities of transitioning to all possible next states to 1 / # words

    # Check if any row sums to zero
    row_sums_is_zero = transition_matrix.sum(axis=1) == 0

    # Iterate over rows to find zero-sum rows
    first_word = list(unique_words.keys())[0]
    for idx, is_zero in enumerate(row_sums_is_zero):
        # If the row sum is zero
        if is_zero:
            # Convert the index to state representation
            state_name = index_to_words(idx, unique_words_count, n-1, unique_words)

            # Calculate the range of indices for the next posiible states
            possible_states_start_index = words_to_index(state_name[1:] + [first_word, ], unique_words_count, unique_words)
            possible_states_end_index = possible_states_start_index + unique_words_count

            # Set the number of transitions to 1 for the range
            transition_matrix[idx, possible_states_start_index : possible_states_end_index] = 1

    ## Step 3.3: Normalize the transition matrix

    # The following steps is to normalize transition matrix so that the sum of each row is 1
    # (The reason is: By the definition of a Markov chain, the sum of each row should be 1)

    # [TODO] Compute the sum of each row
    # hint: use np.sum function and check if the axis parameter is set correctly
    row_sums = np.sum(transition_matrix, axis=1, keepdims=True)

    # [TODO] Divide the transition matrix by the sum of each row to make the sum of each row equals to 1
    transition_matrix = np.divide(transition_matrix, row_sums, where=row_sums != 0)

    return transition_matrix, unique_words

    def generate(unique_words: Dict[str, int], transition_matrix: np.ndarray, start_from: Union[str, List[str]], n: int, length: int=10):
    """
        Generate text using a Markov chain model.

        Args:
            unique_states (Dict[str, int]): A dictionary mapping unique words to their corresponding indices.
            transition_matrix (np.ndarray): A numpy array representing the transition probabilities between states.
            start_from (Union[str, List[str]]): The starting state(s) for text generation.
            n (int): The size of the grams.
            length (int, optional): The length of the generated text. Defaults to 10.

        Returns:
            The generated text.
    """
    # Generate text
    generated_words = start_from.copy() if type(start_from) is list else [start_from]

    # Assert if the number of start words does not equal to n-1
    assert len(generated_words) >= n-1, "The number of start words should be greater than or equals to n-1 ({})".format(n-1)

    # Get the number of unique words
    unique_words_count = len(unique_words)

    # [TODO] Get the number of unique states
    # hint: check step 3.2 in the compute_transition_matrix function
    unique_states_count = unique_words_count ** (n - 1)

    # Generate the next words
    for _ in range(length):
        # [TODO] Get index of current states
        # hint: The current states (current words) is the last n-1 words in the generated text
        # hint: use words_to_index function
        current_words_index = words_to_index(generated_words[-(n-1):], unique_words_count, unique_words)

        # [TODO] Get probability distribution for next state, using the information in the transition matrix
        probabilities = transition_matrix[current_words_index]

        # Select next word based on probabilities, using np.random.choice function
        next_words_index = np.random.choice(unique_states_count, p=probabilities)

        # [TODO] Decode the index and get the last word
        # hint: use index_to_words function
        next_word = index_to_words(next_words_index, unique_words_count, n-1, unique_words)[-1]

        # Add next word to generated text
        generated_words.append(next_word)

    # return generated string
    return ' '.join(generated_words)

    # [TODO] Change the text below and try different values of n
text = "Linear Algebra presents a real challenge. While I anticipated the complexity of the topics, I didn’t expect it to be this difficult to grasp. The concepts, though powerful and foundational, often feel overwhelming. Despite understanding their importance, translating them into something intuitive seems much harder than I imagined."
n = 4

# Process the text and generate the transition matrix
text = remove_unnecessary_characters(text)
n_grams = generate_n_grams(text, n)
transition_matrix, unique_words = compute_transition_matrix(n_grams)

# Print the transition matrix and unique states for obsevation
print("===== The indices for unique states are: =====")
unique_words_count = len(unique_words)
for word_name in list(product(*[unique_words for _ in range(n-1)]))[:20]:
    print(f"{','.join(word_name):10s}: {words_to_index(word_name, unique_words_count, unique_words)}")
print("...", end="\n\n")

print("===== The transition matrix is (Shape of trasition matrix: {}): =====".format(transition_matrix.shape))
print(transition_matrix)
print()

# [TODO] Write down 3 or more initial words and length of generated text to start the text generation
experiments = [
    # Each entry is (Initial words, the length of the generated text)
    ('a real challenge', 15), ('linear algebra presents a', 25), ('while i anticipated', 40)
]

for idx, (start_from, length) in enumerate(experiments, 1):
    start_from = start_from.split(" ")

    # Generate text using the transition matrix
    generated_text = generate(unique_words, transition_matrix, start_from, n, length=length)

    # Print out the generated text
    print("#{} (length={}): {}".format(idx, length, generated_text))

    def compute_stationary_distribution(transition_matrix: np.ndarray):
    """
        Compute the stationary distribution of a Markov chain.

        Parameters:
            transition_matrix (np.ndarray): A numpy array representing the transition probabilities between states.
        Returns:
            The stationary distribution of the Markov chain.
    """
    # [TODO] Implement the function to compute the stationary distribution of a Markov chain, using the method suggested by the LLM

# Initialize a random probability distribution (normalized vector)
    num_states = transition_matrix.shape[0]
    stationary_distribution = np.ones(num_states) / num_states  # Start with uniform distribution

    # [TODO] Implement the power method to compute the stationary distribution
    tol = 1e-6  # tolerance for convergence
    max_iter = 1000  # maximum number of iterations

    for i in range(max_iter):
        next_distribution = np.dot(stationary_distribution, transition_matrix)

        # Check for convergence
        if np.linalg.norm(next_distribution - stationary_distribution) < tol:
            break

        # Update the current distribution
        stationary_distribution = next_distribution

    # Normalize the stationary distribution so it sums to 1
    stationary_distribution = stationary_distribution / stationary_distribution.sum()
    return stationary_distribution

    # Compute the stationary distribution for the transition matrix obtained from the previous problem
stationary_distribution = compute_stationary_distribution(transition_matrix)

# Convert the stationary distribution to float16 to prevent something like -1.2345678e-16 but it is actually 0
print(stationary_distribution.astype(np.float16))