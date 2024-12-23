#  (C) Copyright Wieger Wesselink 2021. Distributed under the GPL-3.0-or-later
#  Software License, (See accompanying file LICENSE or copy at
#  https://www.gnu.org/licenses/gpl-3.0.txt)

import time
from competitive_sudoku.sudoku import GameState, Move, SudokuBoard, TabooMove
import competitive_sudoku.sudokuai
from .Diagnostics import Diagnostics, Logger
import random


class SudokuAI(competitive_sudoku.sudokuai.SudokuAI):
    """
    Sudoku AI that computes a move for a given sudoku configuration.
    """

    def __init__(self):
        super().__init__()

    def compute_best_move(self, game_state: GameState) -> None:
        """
        Computes and proposes the best move for the current game state.
        Implements iterative deepening with alpha-beta pruning and fallback 
        mechanisms.
        Adheres to constraints C0, allowed regions, and taboo moves.
        """
        logger_object = Logger()
        logger = logger_object.get_logger()
        logger.info("Logger handlers: %s",  logger.handlers)
        self.logger = logger
        self.logger.info("SudokuAI initialized.")

        diagnostics = Diagnostics(self.logger)
        manager = GameStateManager(game_state, [], diagnostics, logger)
        minimax = MiniMax(manager, self.logger)

        # Iterative deepening variant of minimax,
        # keeps going until time limit reached
        while True:
            # Minimax computes the best move at the current depth and 
            # increments the depth by 1
            _, best_move_score = minimax.iterative_deepening() 
            # Log the best move - Can be called multiple times,
            # will overwrite the previous best move
            self.propose_move(best_move_score.move)


class MoveScore:
    def __init__(self, move: Move, score: int):
        self.move = move
        self.score = score

    def get_Move(self):
        return self.move
    
    def get_score(self):
        return self.score
    
    def __str__(self) -> str:
        return f'{self.move} has score: {self.score}'


class GameStateManager:
    """Manages the gamestate, with the given moves of the minimax algorithm.
        Basically it can tell the current state,
        which consists of 2 variables: initial_state, moves_performed
        This computes the 'current_state', by applying the 'moves_performed'
        on the 'initial_state'. We compute the 'current_state' before computing
        the legal_moves, and before evaluating the state.
        Then after evaluating the state, we undo the 'moves_performed'
        such that we have a clean initial state.
        By having this structure we don't have to do any deepcopies,
        which significantly speeds up the runtime.
    """
    WINNER_SCORE = 100000
    
    def __init__(
        self,
        initial_state: GameState,
        moves_performed: list[MoveScore],
        diagnostics: Diagnostics,
        logger: Logger,
        scores: list[int] = None,
        steps: int = 0,
        precomputed_constraints: set[TabooMove] = None,
        remaining_earnable_points: int = None,
        points_earned: int = 0,
    ):
        self._initial_state = initial_state
        self._maximizing_player = initial_state.current_player
        self._moves_performed = moves_performed
        self._diagnostics = diagnostics
        self._logger = logger
        self._precomputed_constraints = (
            precomputed_constraints
            if precomputed_constraints
            else self._precompute_constraints()
        )
        self._scores = scores if scores else initial_state.scores[:]
        self._steps = steps
        self._remaining_earnable_points = (
            remaining_earnable_points 
            if remaining_earnable_points 
            else self._compute_remaining_grid_score()
        )
        self._additional_points_earned = points_earned
    
    # Do moves to the current game state --> Return the new state
    def do_move(self, move_score: MoveScore) -> tuple['GameStateManager', int]:
        """
        Apply a move to the current game state and return a new GameStateManager
        and the score of the move.
        """
        # Create a new list of moves by appending the current move
        new_moves = self._moves_performed + [move_score]
        
        # Create a new GameStateManager with the updated moves
        new_manager = GameStateManager(
            self._initial_state,
            new_moves,
            self._diagnostics,
            self._logger,
            self._scores.copy(),
            self._steps + 1,
            self._precomputed_constraints,
            self._remaining_earnable_points,
            self._additional_points_earned + move_score.score,
        )
        
        # Determine who should get the move score
        current_player = self._initial_state.current_player if self._steps % 2 == 0 else 3 - self._initial_state.current_player
        player_index = self._maximizing_player - 1
        opponent_index = 2 - self._maximizing_player - 1
        
        if current_player == self._maximizing_player:
            self._scores[player_index] += move_score.score
        else:
            self._scores[opponent_index] += move_score.score
        
        winner_determined = new_manager._detemine_if_there_if_is_a_winner_already(
            new_manager,
            self._scores[player_index],
            self._scores[opponent_index],
        )
        
        # Undo adding score
        if current_player == self._maximizing_player:
            self._scores[player_index] -= move_score.score
        else:
            self._scores[opponent_index] -= move_score.score
        
        score = winner_determined * GameStateManager.WINNER_SCORE
        
        return new_manager, score
    
    def evaluate_state(self):
        simulated_state = self._apply_moves()
        
        # Evaluate the state
        # Calculate scores for player 1 (current) and player 2 (opponent)
        score_current_player = simulated_state.scores[self.maximizing_player - 1]
        score_opponent = simulated_state.scores[2 - self.maximizing_player]

        # Get the legal moves for the current player and the opponent
        original_player = simulated_state.current_player
        simulated_state.current_player = self.maximizing_player
        legal_moves_current_player = simulated_state.player_squares()
        simulated_state.current_player = 3 - self.maximizing_player
        legal_moves_opponent = simulated_state.player_squares()
        simulated_state.current_player = original_player

        # Determine if there is a winner
        winner_status = self._detemine_if_there_if_is_a_winner_already(
            simulated_state, score_current_player, score_opponent,
            legal_moves_current_player, legal_moves_opponent
        )
        
        # Undo moves to maintain a clean state
        self._undo_moves()
        
        if self.diagnostics:
            self.diagnostics.moves_evaluated_increment()
        
        # Compute the heuristic
        score_heuristic = score_current_player - score_opponent
        move_difference = len(legal_moves_current_player) - len(legal_moves_opponent)
        
        # Incentivize reducing opponent's moves
        if len(legal_moves_opponent) != 0:
            move_difference /= len(legal_moves_opponent)

        return self._heuristic(score_heuristic, move_difference, winner_status)
    
    def _detemine_if_there_if_is_a_winner_already(
        self, gamestate, score_player, score_opponent, player_moves=None, opponent_moves=None
    ):
        # 1 if guaranteed winner. 0 if undecable yet. -1 if lost
        winner_determined = 0

        # Winner is when there is a player who already has a score higher than the max score.
        points_remaining = self.REMAINING_EARNABLE_POINTS - self.additional_points_earned
        score_diff = score_player - score_opponent
        
        # Winner is set.
        if points_remaining < score_diff:
            winner_determined = 1 if score_diff > 0 else -1
            return winner_determined
        
        if player_moves is None:
            return winner_determined
        
        # Someone has no moves left - other player will score all the points
        player_moves, opponent_moves = len(player_moves), len(opponent_moves)
        if player_moves == 0 and opponent_moves == 0:
            winner_determined = 1 if score_diff > 0 else -1
        
        if player_moves == 0:
            # Opponent can score all score diff points
            result = score_diff - points_remaining
            if result > 0:
                winner_determined = 1
            elif result < 0:
                winner_determined = -1
            # else: 0, undecidable yet
            
        if opponent_moves == 0:
            # Player can score all score diff points
            result = score_diff + points_remaining
            if result > 0:
                winner_determined = 1
            elif result < 0:
                winner_determined = -1
            # else: 0, undecidable yet

        return winner_determined

    def possible_legal_moves(self)->list[MoveScore]:
        """Get the possible legal moves and return their score
        Returns:
            list[MoveScore]: Move and its score result
        """
        # Update to current gamestate
        current_game_state = self._apply_moves(for_legal_move_generation=True)
        
        # Get legal moves + score of each move
        moves = self._get_legal_moves(current_game_state)
        if self.diagnostics:
            self.diagnostics.moves_generated_increment(len(moves))
        
        # Revert current gamestate
        self._undo_moves(for_legal_move_generation=True)
        return moves
    
    def _get_legal_moves(self, curr_state: GameState) -> list[MoveScore]:
        player_squares = curr_state.player_squares()
        legal: list[MoveScore] = []

        for square in player_squares:
            move = Move(square, -1)
            score = self._calculate_move_score(curr_state, move)
            result = MoveScore(move, score)
            legal.append(result)

        return legal
    
    def get_legal_value_for_move(self, best_move: MoveScore):
        """
        Get a legal value for the move --> Likely the best move
        We extend this later with a valid move from the sudoku
        """
        # Assumes that it's a legal square
        square = best_move.move.square
        row, column = square

        curr_state = self.initial_gamestate
        dimensions = curr_state.board.N
        taboo_moves = self.initial_gamestate.taboo_moves

        # Exclude already used numbers
        cand_val = set(range(1, dimensions + 1))
        cand_val -= set(curr_state.board.get(
                        (row, i)) for i in range(dimensions))  # Row
        cand_val -= set(curr_state.board.get(
                        (i, column)) for i in range(dimensions))  # Column

        # Constraints for blocks
        start_block_row = (row // curr_state.board.m) * curr_state.board.m
        start_block_col = (column // curr_state.board.n) * curr_state.board.n
        for i in range(start_block_row, start_block_row + curr_state.board.m):
            for j in range(start_block_col, start_block_col + curr_state.board.n):
                cand_val.discard(curr_state.board.get((i, j)))

        # Check the values that remain and taboo filtering
        options = []
        for val in cand_val:
            if TabooMove(square, val) not in taboo_moves:
                move = Move(square, val)
                score = self._calculate_move_score(curr_state, move)
                result = MoveScore(move, score)
                options.append(result)

        # Choose random option to reduce chance of making a taboo move.
        return random.choice(options)

    def _switch_player(self, game_state: GameState) -> int:
        game_state.current_player = 3 - game_state.current_player
        return game_state.current_player

    def _precompute_constraints(self):
        """
        Precomputes numbers present in each row, column,
          and block for faster validation.
        Returns a dict: dictionary containing number
          for rows, columns, and blocks. Telling the items in the row
        """
        board = self.initial_gamestate.board
        rows = [0 for _ in range(board.N)]
        columns = [0 for _ in range(board.N)]
        blocks = [0 for _ in range(board.N)]

        for row in range(board.N):
            for col in range(board.N):
                value = board.get((row, col))
                if value is not None and value > 0:
                    rows[row] += 1
                    columns[col] += 1
                    block_index = (row // board.m) * board.m + (col // board.n)
                    blocks[block_index] += 1

        return {"rows": rows, "columns": columns, "blocks": blocks}

    def _update_constraints(self, move: Move):
        # Update the precomputed constraints with the provided move
        # Move has to be legal (i.e. not in the constraints)
        row, col = move.square
        # value = move.value - not relevan
        board = self.initial_gamestate.board
        block_index = (row // board.m) * board.m + (col // board.n)

        self.precomputed_constraints["rows"][row] += 1
        self.precomputed_constraints["columns"][col] += 1
        self.precomputed_constraints["blocks"][block_index] += 1

    def _remove_constraints(self, move: Move):
        # Assumes that the move was previously applied and was legal
        row, col = move.square
        # value = move.value - Not relevant
        board = self.initial_gamestate.board
        block_index = (row // board.m) * board.m + (col // board.n)

        self.precomputed_constraints["rows"][row] -= 1
        self.precomputed_constraints["columns"][col] -= 1
        self.precomputed_constraints["blocks"][block_index] -= 1

    def _calculate_move_score(self, curr_state: GameState, move: Move) -> int:
        row, col = move.square
        board = curr_state.board
        score = 0

        # Precompute indices for the block
        block_index = (row // board.m) * board.m + (col // board.n)

        # Check multi-region completion
        regions_completed = 0
        if self.precomputed_constraints["rows"][row] == board.N - 1:
            regions_completed += 1
        if self.precomputed_constraints["columns"][col] == board.N - 1:
            regions_completed += 1
        if self.precomputed_constraints["blocks"][block_index] == board.N - 1:
            regions_completed += 1

        # Assign points based on completed regions
        if regions_completed == 1:
            score = 1
        elif regions_completed == 2:
            score = 3
        elif regions_completed == 3:
            score = 7

        return score
    
    # Apply moves to the game state - without deepcopy for speed
    def _apply_moves(self, for_legal_move_generation=False) -> GameState:
        """
        Applies all recorded moves to the initial game state.
        
        Args:
            for_legal_move_generation (bool): Whether the moves are applied
              for legal move generation.
        Returns:
            GameState: The updated game state.
        """
        game_state = self.initial_gamestate
        
        # Save backup of allowed squares and scores
        self.old_scores = game_state.scores.copy()
        self.old_additional_points_earned = self.additional_points_earned
        
        # Extracted logic from simulate_game.py
        for move_score in self.moves:
            # Place move and update fields
            game_state.board.put(move_score.move.square, move_score.move.value)
            game_state.moves.append(move_score)
            if True:  # originally it was: playmode != 'classic'
                game_state.occupied_squares().append(move_score.move.square)
                
            # Only for legal move generation, we need to update constraints
            if for_legal_move_generation:
                self._update_constraints(move_score.move)
            
            # Add score
            game_state.scores[game_state.current_player-1] += move_score.score
            self.additional_points_earned += move_score.score
            
            # Switch to next player
            self._switch_player(game_state)
        
        # Update allowed squares
        return game_state
    
    # Undo moves to the game state - so we make a clean state
    def _undo_moves(self, for_legal_move_generation=False) -> None:
        game_state = self.initial_gamestate
        
        # Extracted logic from simulate_game.py
        for move_score in self.moves[::-1]:
            # Switch to last player
            self._switch_player(game_state)
            
            # Undo the move
            game_state.board.put(move_score.move.square, SudokuBoard.empty)
            game_state.moves.pop()
            if True:  # originally it was: playmode != 'classic'
                game_state.occupied_squares().pop()
                
            # Undo constraint update - only for legal move generation
            if for_legal_move_generation:
                self._remove_constraints(move_score.move)

        # Restore scores
        self.additional_points_earned = self.old_additional_points_earned
        game_state.scores = self.old_scores
    
    # Winner has factor 10000, because no point in score more if you won or lost.
    def _heuristic(self, points_heuristic: int, move_difference,
                   winner_determined, point_weight=20, move_weight=1,
                   winner_weight=10000) -> int:
        score = (points_heuristic * point_weight + move_difference * move_weight
                  + winner_determined * winner_weight)
        return score
    
    # Calculate the remaining points that can be scored in the initial game state of this round
    def _compute_remaining_grid_score(self):
        # mark a list with rows, columns and blocks that have at least one empty square
        board = self.initial_gamestate.board
        # Empty squares in rows, columns, and blocks
        rows = [0 for _ in range(board.N)]
        columns = [0 for _ in range(board.N)]
        blocks = [0 for _ in range(board.N)]
        
        for row in range(board.N):
            for col in range(board.N):
                value = board.get((row, col))
                if value is None or value == 0:
                    rows[row] += 1
                    columns[col] += 1
                    block_index = (row // board.m) * board.m + (col // board.n)
                    blocks[block_index] += 1
                    
        # Remove rows columns and blocks that are already full
        rows = [row for row in rows if row > 0]
        columns = [col for col in columns if col > 0]
        blocks = [block for block in blocks if block > 0]
                    
        # Calculate the maximum score.
        # First make grid, row, block triplets with score 7
        # Then make grid, row, block pairs with score 3
        # Then make grid, row, block singles with score 1
        remaining_score = 0
        while len(rows) > 0 and len(columns) > 0 and len(blocks) > 0:
            remaining_score += 7
            rows.pop()
            columns.pop()
            blocks.pop()
            
        while len(rows) > 0 and len(columns) > 0:
            remaining_score += 3
            rows.pop()
            columns.pop()
        
        while len(blocks) > 0 and len(columns) > 0:
            remaining_score += 3
            blocks.pop()
            columns.pop()
            
        while len(rows) > 0 and len(blocks) > 0:
            remaining_score += 3
            rows.pop()
            blocks.pop()
            
        while len(rows) > 0:
            remaining_score += 1
            rows.pop()
            
        while len(columns) > 0:
            remaining_score += 1
            columns.pop()
            
        while len(blocks) > 0:
            remaining_score += 1
            blocks.pop()
        
        return remaining_score
 
            
class MiniMax:
    """All the logic for the minimax algorithm, and returns the best score. Applies iterative deepening such that we go to the deepest depth possible in the given timestate.
    """
    #not nessesary
    # state_manager: GameStateManager
    # computed_depth = 1
    # diagnostics = None
    
    def __init__(self, state_manager: GameStateManager, logger):
        self.state_manager = state_manager
        self.computed_depth = 1
        self.logger = logger
        self.logger.info("Start Minimax algorithm")

    
    def iterative_deepening(self) -> MoveScore:
        """Computes the best move at the current depth, and returns the best score and move
        """ 
        # Sleep indefinetly/to the end of the turn if we reached a sensible depth
        if self.computed_depth > self.state_manager.initial_gamestate.board.N ** 2:
            self.logger.warning("Reached maximum depth limit")
            time.sleep(10**5)
        
        # Do the minimax
        # best_score, best_move = self.minimax(self.state_manager, self.computed_depth, True)
        best_score, best_move = self.minimax_alpha_beta(self.state_manager, self.computed_depth, True)

        # Fetch the value of the legal move
        best_move_with_value = self.state_manager.get_legal_value_for_move(best_move)
        
        #Log the results
        self._log_results(best_move_with_value, best_score)
        
        # Increase the depth - for iterative deepening
        self.computed_depth += 1
        
        # Return the best move
        return best_score, best_move_with_value
    
    def minimax_alpha_beta(self, state_manager: GameStateManager, depth: int, is_maximizing: bool, alpha=float('-inf'), beta=float('inf')) -> tuple[int, MoveScore]:
        """Computes the best move and score, using minimax

        Args:
            state_manager (GameStateManager): The manager where we can apply moves
            depth (int): _description_
            is_maximizing (bool): Wheter the player is the player maximizing its score 

        Returns:
            value, best_move: returns the best score, and best move
        """
        if depth == 0:
            score = state_manager.evaluate_state()
            self.logger.debug(f"Depth 0 reached, evaluated score: {score}")
            return score, None
        
        #stop, if there are no moves. (terminal state)
        moves = state_manager.possible_legal_moves()
        if len(moves) == 0:
            score = state_manager.evaluate_state()
            self.logger.debug(f"No moves left, evaluated score: {score}")
            return score, None 

        best_move = None
        if is_maximizing:
            max_score = float('-inf')

            #Loop over legal moves to determine best one
            for move in moves:
                #Get new simulated state based on move
                new_state, score = state_manager.do_move(move)

                # If score is not 0, we reached a terminal state where is a winner -> reeturn
                if score < 0 or score > 0:
                    return score, move

                new_score, _ = self.minimax_alpha_beta(new_state, depth - 1, False, alpha, beta) 
                
                #check if better than current best move
                if new_score > max_score:
                    max_score = new_score
                    best_move = move

                if max_score >= beta:
                    break
                alpha = max(alpha, max_score)
            return max_score, best_move
        else:
            #Minimizing the opponent, inverse of the above logic
            min_score = float('inf')
            
            for move in moves:
                new_state, score = state_manager.do_move(move)
                new_score, _ = self.minimax_alpha_beta(new_state, depth - 1, True, alpha, beta)

                # If score is not 0, we reached a terminal state where is a winner -> reeturn
                if score < 0 or score > 0:
                    return score, move
                
                if new_score < min_score:
                    min_score = new_score
                    best_move = move
                if min_score <= alpha:
                    break
                beta = min(beta, min_score)
            return min_score, best_move
            
    def _log_results(self, best_move: MoveScore, best_score: int):
        """Log results of the current depth."""
        self.logger.info("~" * 50)
        self.logger.info(f"Depth computed: {self.computed_depth}")
        self.logger.info(f"Best move at depth {self.computed_depth}: {best_move} - max state score: {best_score}")
        self.logger.info("~" * 50)
        self.logger.info("Moving on to next depth...")
        