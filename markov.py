import json

def pairs(s):
    """ Split the string into pairs of two characters """
    s = s.strip()  # Remove any extra spaces or newline characters
    return [s[i:i+2] for i in range(0, len(s), 2)]

def train(input_file, output_file, order=1):
    """ Train an n-th order Markov model by processing the RPS games and store the matrix in a file """
    with open(input_file, 'r') as f:
        data = f.readlines()

    matrix = {}

    # Build the transition matrix based on the RPS games data
    for game in data:
        curGame = pairs(game)
        for current_order in range(1, order + 1):
            for i in range(len(curGame) - current_order):
                # Create the current game state as a combination of the last 'current_order' pairs
                game_state = tuple(curGame[i:i+current_order])  # Create a tuple for immutability

                # Convert the tuple to a string to make it JSON-serializable
                game_state_str = ','.join(game_state)

                # Add normal game state to matrix
                if game_state_str not in matrix:
                    matrix[game_state_str] = [0, 0, 0]

                # Update counts based on the next move
                if curGame[i + current_order][0] == 'R':
                    matrix[game_state_str][0] += 1
                elif curGame[i + current_order][0] == 'P':
                    matrix[game_state_str][1] += 1
                elif curGame[i + current_order][0] == 'S':
                    matrix[game_state_str][2] += 1

                # Reverse the game state (reverse individual pairs) and add to matrix
                reversed_game_state = tuple(pair[::-1] for pair in curGame[i:i+current_order])

                # Convert the reversed tuple to a string for JSON-compatibility
                reversed_game_state_str = ','.join(reversed_game_state)

                if reversed_game_state_str not in matrix:
                    matrix[reversed_game_state_str] = [0, 0, 0]

                # Update counts based on the reversed game's next move
                if len(curGame[i + current_order]) > 1:
                    if curGame[i + current_order][1] == 'R':
                        matrix[reversed_game_state_str][0] += 1
                    elif curGame[i + current_order][1] == 'P':
                        matrix[reversed_game_state_str][1] += 1
                    elif curGame[i + current_order][1] == 'S':
                        matrix[reversed_game_state_str][2] += 1

    # Save the matrix to a file
    with open(output_file, 'w') as out_file:
        json.dump(matrix, out_file)

    print(f"Training complete. Model saved to {output_file}")

def load_model(file_path):
    """ Load the matrix from a file and return the dictionary object """
    with open(file_path, 'r') as f:
        matrix = json.load(f)
    return matrix

def predict_move(matrix, game_state, order):
    """ Given a game state, predict the most likely next move (from the opponent's perspective) """
    for current_order in range(order, 0, -1):  # Try from higher-order to lower-order Markov chains
        if len(game_state) < current_order:
            continue  # Skip if there aren't enough game states for this order
        
        # Reverse the individual pairs for the game state (to match opponent-first training)
        reversed_game_state = [pair[::-1] for pair in game_state[-current_order:]]
        game_state_str = ','.join(reversed_game_state)

        if game_state_str in matrix:
            # Retrieve counts for the next moves: [R, P, S]
            counts = matrix[game_state_str]
            print(f"Game state {game_state_str} has counts {counts}")
            # Find the most likely next move based on counts
            max_count = max(counts)
            if counts[0] == max_count:
                likely_move = 'R'  # Most likely move is Rock (opponent's move)
            elif counts[1] == max_count:
                likely_move = 'P'  # Most likely move is Paper (opponent's move)
            else:
                likely_move = 'S'  # Most likely move is Scissors (opponent's move)

            # Suggest the move Player 2 should make to win
            if likely_move == 'R':
                counter_move = 'P'  # Paper beats Rock
            elif likely_move == 'P':
                counter_move = 'S'  # Scissors beat Paper
            else:
                counter_move = 'R'  # Rock beats Scissors

            return f"Opponent will play {likely_move}. Counter with {counter_move}."

    # If no prediction could be made with available game states
    return "No prediction available for this game state"


def fineTune(matrix, results_file, order):
    """
    Fine-tune the Markov model based on game outcomes in results.txt.
    Each move in the sequence is stored as a pair, with the player's move first and the opponent's move second.
    This function reverses the pairs and sets the opponent's response as the first letter of the next round.
    """
    with open(results_file, 'r') as f:
        lines = f.readlines()

    # Iterate through each game result in the results file
    for i in range(0, len(lines), 4):  # Each game is 4 lines (sequence, results, outcome, separator)
        game_sequence = lines[i].strip()  # E.g., "PRPRPR"
        
        # Split the game sequence into pairs, then reverse each pair
        curGame = pairs(game_sequence)
        
        # Check that each pair has exactly two characters before reversing
        reversedGame = [(pair[1], pair[0]) for pair in curGame if len(pair) == 2]

        # Iterate over the reversed game states and update the model for the chosen order and lower orders
        for current_order in range(1, order + 1):
            for j in range(len(reversedGame) - current_order):
                # Extract the current game state as a tuple of moves
                game_state = tuple(reversedGame[j:j + current_order])
                
                # The next move (first character of the next pair in the reversed sequence)
                next_move = reversedGame[j + current_order][0]

                # Convert the game state to a string for storage in the matrix
                game_state_str = ','.join([f"{a[0]}{a[1]}" for a in game_state])

                # Update the matrix for this game state
                if game_state_str not in matrix:
                    matrix[game_state_str] = [0, 0, 0]  # Initialize counts for R, P, S

                # Update counts based on the opponent's next move
                if next_move == 'R':
                    matrix[game_state_str][0] += 1
                elif next_move == 'P':
                    matrix[game_state_str][1] += 1
                elif next_move == 'S':
                    matrix[game_state_str][2] += 1

    print("Model fine-tuned.")
    return matrix  # Ensure the updated matrix is returned


def play(matrix, order, results_file="results.txt", continuous=False):
    """ Play a game of RPS with the model, predicting the opponent's moves.
        Automatically continue and keep track of all game states, but only use 'order' states for predictions.
        Store the results in a text file and organize into 'first to 3' games if continuous is False."""
    
    game_state = []  # Stores the previous game states
    round_number = 1
    player_score = 0
    opponent_score = 0
    move_sequence = ""  # Stores the sequence of all moves
    result_sequence = []  # Stores the sequence of results (W, T, L)

    with open(results_file, 'a') as f:
        print("Let's play Rock-Paper-Scissors!")
        print("Recommended Opening: R")
        print("Enter the current game state (e.g., RP, PS) where the first letter is your move and the second letter is the opponent's move.")
        
        while True:
            print(f"\n--- Round {round_number} ---")
            
            # Get the current game state (your move + opponent's move)
            current_game_state = input("Game State: ").upper()
            while len(current_game_state) != 2 or current_game_state[0] not in ['R', 'P', 'S'] or current_game_state[1] not in ['R', 'P', 'S']:
                current_game_state = input("Invalid input. Enter a valid game state (e.g., RP, PS): ").upper()

            player_move = current_game_state[0]
            opponent_move = current_game_state[1]

            # Log the moves in the move sequence
            move_sequence += current_game_state

            # Determine the winner of the round and update the score
            if (player_move == 'R' and opponent_move == 'S') or (player_move == 'S' and opponent_move == 'P') or (player_move == 'P' and opponent_move == 'R'):
                player_score += 1
                result_sequence.append("W")  # Win
            elif player_move == opponent_move:
                result_sequence.append("T")  # Tie
            else:
                opponent_score += 1
                result_sequence.append("L")  # Loss

            # Display current scores
            print(f"You - {player_score}, Opponent - {opponent_score}")

            # Update the game state with the latest move
            game_state.append(current_game_state[::-1])  # Reverse the game state for prediction (opponent's move first)

            # Make a prediction using only the last 'order' game states, but fall back to lower order if necessary
            prediction = predict_move(matrix, game_state, order)
            print(prediction)

            # Check if either player has won the game
            if not continuous and (player_score == 3 or opponent_score == 3):
                winner = "You" if player_score == 3 else "Opponent"
                print(f"{winner} won the game!")
                
                # Log the results in the file
                f.write(move_sequence + "\n")  # Store moves in a single line
                f.write("-".join(result_sequence) + "\n")  # Store results like W-T-L
                f.write(f"{'Win' if player_score == 3 else 'Loss'}\n")
                f.write("-------\n")  # Add separator for end of the game
                
                # Reset for a new game
                player_score = 0
                opponent_score = 0
                move_sequence = ""
                result_sequence = []
                game_state = []  # Reset the remembered game states
                print("\nNew game starting...")
            
            # Automatically move to the next round
            round_number += 1
        f.close()


if __name__ == "__main__":
    order = 3
    selfImproveOn = False

    train("RPS.txt", "markov", order)
    matrix = load_model("markov")
    if(selfImproveOn):
        matrix = fineTune(matrix, "results.txt", order)
        with open('finetunedmarkov.json', 'w') as f:
            json.dump(matrix, f)  # Save the fine-tuned model to a file

    play(matrix, order, "results.txt", continuous=True)



