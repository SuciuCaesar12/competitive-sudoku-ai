import json
from collections import defaultdict
import os
import matplotlib.pyplot as plt
from simulate_game import play_game


def update_results(simulation_result, results, board, time, player1, player2, first_as):
    """Updates the results dictionary with the outcomes and scores of a single simulation."""
    # if first_as == 1:
    #     score1, score2 = simulation_result
    # else:
    #     score2, score1 = simulation_result

    score1, score2 = simulation_result
    
    if score1 > score2:
        result = f"{player1} wins"
    elif score1 < score2:
        result = f"{player2} wins"
    else:
        result = "Draw"

    results.append({
        "player1": player1,
        "player2": player2,
        "board": board,
        "time": time,
        "result": result,
        "score1": score1,
        "score2": score2
    })

def plot_board_results(results, num_simulations):
    """Plots normalized bar charts of scores for each board and saves the figures."""
    for board in set(game["board"] for game in results):
        board_results = [game for game in results if game["board"] == board]
        times = sorted(set(game["time"] for game in board_results))

        # Prepare data for bar plot
        x_labels = []
        team2_scores = {time: defaultdict(float) for time in times}  # Scores by time for team17_A2 vs others

        for time in times:
            time_results = [game for game in board_results if game["time"] == time]

            # Aggregate scores for each opponent
            for opponent in ["team17_A1", "greedy_player"]:
                t_score = sum(game["score1"] for game in time_results if game["player1"] == "team17_A2" and game["player2"] == opponent) + \
                          sum(game["score2"] for game in time_results if game["player2"] == "team17_A2" and game["player1"] == opponent)
                team2_scores[time][opponent] += t_score / num_simulations  # Normalize scores

            x_labels.append(time)

        # Plot bar chart
        x = range(len(times))
        plt.figure(figsize=(12, 6))
        bar_width = 0.3

        for i, opponent in enumerate(["team17_A1", "greedy_player"]):
            scores = [team2_scores[time][opponent] for time in times]
            plt.bar([p + i * bar_width for p in x], scores, width=bar_width, label=f"team17_A2 vs {opponent}")

        plt.xticks([p + bar_width / 2 for p in x], times)
        plt.title(f"Normalized Scores by Time for Board: {board}")
        plt.xlabel("Time")
        plt.ylabel("Normalized Scores")
        plt.legend()
        plt.tight_layout()

        # Save the figure
        figure_file = f"scores_{board.replace('/', '_').replace('.', '_')}.png"
        plt.savefig(figure_file)
        print(f"Figure saved as {figure_file}")
        plt.show()


def main():
    # Parameters
    players = ["team17_A1", "greedy_player"]
    main_player = "team17_A2"
    boards = ["boards/board-2x3.txt", "boards/empty-3x3.txt"]
    times = [1, 2, 3]
    # Number of repetitions
    N = 2
    output_file = "simulation_results_with_scores.json"

    if os.path.isfile(output_file):
        with open(output_file, "r") as f:
            results = json.load(f)["games"]
    else:
        # Organize data into a list for results
        results = []

        # Run simulations and update results
        for opponent in players:
            for board in boards:
                for time in times:
                    for _ in range(N):
                        print(f"Running: {main_player} vs {opponent} on {board} with time {time}")
                        simulation_result = play_game(board, main_player, opponent, time, False)
                        update_results(simulation_result, results, board, time, main_player, opponent, first_as=1)

                        print(f"Running: {opponent} vs {main_player} on {board} with time {time}")
                        simulation_result = play_game(board, opponent, main_player, time, False)
                        update_results(simulation_result, results, board, time, opponent, main_player, first_as=2)

        # Save results to JSON file
        
        with open(output_file, "w") as f:
            json.dump({"games": results}, f, indent=4)

    # Plot bar charts for each board
    plot_board_results(results, N * len(boards) * len(times) * len(players))

    print(f"Results saved to {output_file}")


if __name__ == '__main__':
    main()
