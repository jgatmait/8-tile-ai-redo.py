import numpy as np
import heapq
import time
import random
from statistics import mean, stdev
import tracemalloc


class PuzzleSolver:
    def __init__(self):
        # Initialize the solver with the goal state, initial state, and grid size.
        # The goal state is predefined for the 8-puzzle.
        self.goal_state = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 0]])
        #self.initial_state = np.array(initial_state)
        self.size = 3
        self.initial_state = None

    def set_initial_state_from_user_input(self):
        # Prompt the user to choose between entering the state manually or generating a random state.
        user_choice = input(
            "Enter '1' to input the initial state manually, or '2' to generate a random solvable state: ")

        if user_choice == '1':
            # User opts to enter the state manually.
            print("Please enter the initial state of the puzzle row by row, with numbers separated by space:")
            input_state = []
            for i in range(self.size):
                while True:
                    try:
                        row_input = input(f"Enter row {i + 1} (e.g., 1 2 3 for the first row): ")
                        row = [int(x) for x in row_input.split()]
                        if len(row) != self.size:
                            raise ValueError("Row does not have the correct number of elements.")
                        input_state.append(row)
                        break
                    except ValueError as e:
                        print(f"Invalid input: {e}. Please try again.")

            if self.set_initial_state(input_state):
                print("Initial state set successfully.")
            else:
                print("Failed to set initial state.")
        elif user_choice == '2':
            # User opts to generate a random solvable state.
            generated_state = self.generate_state()
            if self.set_initial_state(generated_state):
                print("Random solvable state generated and set successfully.")
                self.pretty_print(generated_state)
            else:
                print("Failed to generate or set a random solvable state.")
        else:
            print("Invalid choice. Please enter '1' or '2'.")

    def set_initial_state(self, state):
        # Set or update the initial state of the puzzle.
        try:
            if state is not None and len(state) == self.size and all(len(row) == self.size for row in state):
                self.initial_state = np.array(state)
                if not self.check_solvability(self.initial_state):
                    print("The provided state is not solvable. Please provide a solvable state.")
                    return False
                return True
            else:
                print("Invalid state provided. Please ensure it's a 3x3 list of lists.")
                return False
        except:
            print("An error occurred while setting the initial state.")
            return False

    def generate_state(self):
        # Generate a solvable random start state
        while True:
            temp_state = list(range(9))
            random.shuffle(temp_state)
            generated_state = np.array(temp_state).reshape((self.size, self.size))
            if self.check_solvability(generated_state):
                return generated_state

    def check_solvability(self, state):
        # Check if a given state is solvable
        inversions = 0
        state_list = state.flatten()
        for i in range(len(state_list)):
            for j in range(i + 1, len(state_list)):
                if state_list[i] != 0 and state_list[j] != 0 and state_list[i] > state_list[j]:
                    inversions += 1
        return inversions % 2 == 0

    def find_empty_tile(self, state):
        # Find and return the coordinates of the empty tile (represented by 0) in the given state.
        for i in range(self.size):
            for j in range(self.size):
                if state[i][j] == 0:
                    return i, j

    def move(self, state, direction):
        # Attempt to move a tile into the empty space in the given direction (up, down, left, right).
        # Returns the new state after the move, or None if the move is invalid.
        i, j = self.find_empty_tile(state)
        if direction == 'up' and i > 0:
            # Swap the empty tile with the tile above it
            new_state = state.copy()
            new_state[i][j], new_state[i - 1][j] = new_state[i - 1][j], new_state[i][j]
            return new_state
        elif direction == 'down' and i < self.size - 1:
            new_state = state.copy()
            new_state[i][j], new_state[i + 1][j] = new_state[i + 1][j], new_state[i][j]
            return new_state
        elif direction == 'left' and j > 0:
            new_state = state.copy()
            new_state[i][j], new_state[i][j - 1] = new_state[i][j - 1], new_state[i][j]
            return new_state
        elif direction == 'right' and j < self.size - 1:
            new_state = state.copy()
            new_state[i][j], new_state[i][j + 1] = new_state[i][j + 1], new_state[i][j]
            return new_state
        return None


    def is_goal(self, state):
        # Check if the given state is the goal state.
        return np.array_equal(state, self.goal_state)

    def pretty_print(self, state):
        # Print the current state in a readable format.
        for row in state:
            print(" ".join(map(str, row)))
        print()

    def a_star_search(self, heuristic):
        # Initialize an open list (priority queue) and a closed list (set of visited states)
        open_list = []
        closed_list = set()

        # Push the initial state with its heuristic value, cost, state bytes, and an empty path into the open list
        heapq.heappush(open_list, (
            heuristic(self.initial_state), 0, self.initial_state.tobytes(),
            []))  # Note the empty list [] for the path

        # Record the start time for measuring execution time
        start_time = time.time()
        expanded_nodes = 0  # Counter to track the number of expanded nodes

        # A* search algorithm loop
        while open_list:
            _, cost, current_state_bytes, path = heapq.heappop(open_list)  # Extract the path
            current_state = np.frombuffer(current_state_bytes, dtype=int).reshape((self.size, self.size))

            # Check if the current state is the goal state
            if self.is_goal(current_state):
                end_time = time.time()
                execution_time = end_time - start_time
                print(f"Solved in {execution_time:.6f} seconds.")
                print(f"Moves taken: {path}")  # Print the solution path
                print("Nodes expanded: ", expanded_nodes)
                return path, cost, expanded_nodes  # Return the solution path along with cost and expanded nodes

            current_state_str = current_state.tobytes()
            if current_state_str not in closed_list:
                closed_list.add(current_state_str)
                expanded_nodes += 1

                i, j = self.find_empty_tile(current_state)
                directions = ['up', 'down', 'left', 'right']
                for direction in directions:
                    new_state = self.move(current_state, direction)
                    if new_state is not None:
                        new_cost = cost + 1
                        new_path = path + [direction]  # Append the direction to the path
                        heapq.heappush(open_list, (
                            new_cost + heuristic(new_state), new_cost, new_state.tobytes(),
                            new_path))  # Include the new path

        print("No solution found.")
        return None, None, expanded_nodes

    def heuristic_hamming(self, state):
        # Hamming heuristic (number of misplaced tiles)
        hamming_distance = 0
        for i in range(self.size):
            for j in range(self.size):
                if state[i][j] != self.goal_state[i][j] and state[i][j] != 0:
                    hamming_distance += 1
        return hamming_distance

    def heuristic_manhattan(self, state):
        # Calculate the Manhattan distance heuristic for the given state compared to the goal state.
        # This is the sum of the distances each tile is from its goal position.
        distance = 0
        for i in range(self.size):
            for j in range(self.size):
                if state[i][j] != 0:
                    goal_row, goal_col = divmod(state[i][j] - 1, self.size)
                    distance += abs(i - goal_row) + abs(j - goal_col)
        return distance



    def measure_performance(self, heuristic_function, iterations):
        memory_usage = []
        expanded_nodes_list = []
        execution_time = []

        for _ in range(iterations):
            random_state = self.generate_state()
            self.initial_state = random_state

            start_time = time.time()
            path, cost, expanded_nodes = self.a_star_search(heuristic_function)
            end_time = time.time()

            # If a path is found, use the number of expanded nodes; otherwise, append 0 (indicating failure)
            memory_usage.append(expanded_nodes if path else 0)
            expanded_nodes_list.append(expanded_nodes)
            execution_time.append(end_time - start_time)

        # Calculate mean and standard deviation for memory usage, execution time, and expanded nodes
        mean_memory_usage = mean(memory_usage)
        std_memory_usage = stdev(memory_usage)
        mean_execution_time = mean(execution_time)
        std_execution_time = stdev(execution_time)

        # Return the calculated statistics
        return {
            'mean_memory_usage': mean_memory_usage,
            'std_memory_usage': std_memory_usage,
            'mean_execution_time': mean_execution_time,
            'std_execution_time': std_execution_time,

        }


initial_board = [
    [1, 2, 3],
    [4, 6, 0],
    [7, 5, 8]
]

solver = PuzzleSolver()
solver.set_initial_state(initial_board)
solver.a_star_search(solver.heuristic_manhattan)

# USER-INPUT-BOARD-INITIALIZATION!!!
solver.set_initial_state_from_user_input()
if solver.initial_state is not None:
    solver.a_star_search(solver.heuristic_manhattan)
else:
    print("No valid initial state has been set. Using pre-set initial board")

#solver.a_star_search(solver.heuristic_manhattan)


# PERFORMANCE MEASUREMENTS!!! UNCOMMENT TO RUN
"""""
iterations = 100 #Determines how many times it runs

tracemalloc.start()
results_manhattan = solver.measure_performance(solver.heuristic_manhattan, iterations)
print("Performance Metrics for Manhattan Heuristic:")
for key, value in results_manhattan.items():
     print(f"{key}: {value}")
snapshot_manhattan = tracemalloc.take_snapshot()
current_manhattan, peak_manhattan = tracemalloc.get_traced_memory()
print(f"Current memory usage is {current_manhattan / 10**6}MB; Peak was {peak_manhattan / 10**6}MB")
tracemalloc.stop()

print("________________")

tracemalloc.start()
results_hamming = solver.measure_performance(solver.heuristic_hamming, iterations)
print("Performance Metrics for Hamming Heuristic:")
for key, value in results_hamming.items():
     print(f"{key}: {value}")
snapshot_hamming = tracemalloc.take_snapshot()
current_hamming, peak_hamming = tracemalloc.get_traced_memory()
print(f"Current memory usage is {current_hamming / 10**6}MB; Peak was {peak_hamming / 10**6}MB")
tracemalloc.stop()


"""""