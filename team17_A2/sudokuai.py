#  (C) Copyright Wieger Wesselink 2021. Distributed under the GPL-3.0-or-later
#  Software License, (See accompanying file LICENSE or copy at
#  https://www.gnu.org/licenses/gpl-3.0.txt)


import copy
import competitive_sudoku.sudokuai
from itertools import product, combinations
from typing import Dict, List
from competitive_sudoku.sudoku import *

# Scoring dictionary for evaluating completed regions in Sudoku
SCORE_DICT = {0: 0, 1: 1, 2: 3, 3: 7}


class MemorySudokuBoard:
    
    def __init__(self, board: SudokuBoard):
        self.board = board
        self.row_cache = [None] * self.board.N
        self.col_cache = [None] * self.board.N
        self.block_cache = [[None] * (self.board.N // self.board.m) for _ in range(self.board.N // self.board.n)]
        self.VALUES = set(range(1, self.board.N + 1))
        
    def get_row_squares(self, row: int) -> set[Square]:
        """Returns the set of squares in the specified row."""
        return {(row, col) for col in range(self.board.N)}
    
    def get_column_squares(self, col: int) -> set[Square]:
        """Returns the set of squares in the specified column."""
        return {(row, col) for row in range(self.board.N)}
    
    def get_block_squares(self, row: int, col: int) -> set[Square]:
        """
        Returns the set of squares in the block containing the specified square.
        Blocks are determined based on the board's dimensions (m x n).
        """
        start_row, end_row = row - (row % self.board.m), row + (self.board.m - (row % self.board.m))
        start_col, end_col = col - (col % self.board.n), col + (self.board.n - (col % self.board.n))
        return set(product(range(start_row, end_row), range(start_col, end_col)))
    
    def possible_values_in_row(self, row: int) -> set:
        """
        Returns the set of possible values for a row by excluding already filled values.
        Cached results are used if available.
        """
        if self.row_cache[row] is None:
            values = [self.board.get(sq) for sq in self.get_row_squares(row)]
            self.row_cache[row] = self.VALUES - set(values)
        return self.row_cache[row]
    
    def possible_values_in_column(self, col: int) -> set:
        """
        Returns the set of possible values for a column by excluding already filled values.
        Cached results are used if available.
        """
        if self.col_cache[col] is None:
            values = [self.board.get(sq) for sq in self.get_column_squares(col)]
            self.col_cache[col] = self.VALUES - set(values)
        return self.col_cache[col]
    
    def possible_values_in_block(self, row: int, col: int) -> set:
        """
        Returns the set of possible values for a block by excluding already filled values.
        Cached results are used if available.
        """
        i, j = row // self.board.m, col // self.board.n
        if self.block_cache[i][j] is None:
            values = [self.board.get(sq) for sq in self.get_block_squares(row, col)]
            self.block_cache[i][j] = self.VALUES - set(values)
        return self.block_cache[i][j]
    
    def possible_values_for_square(self, square: Square) -> set:
        """
        Returns the set of possible values for a square based on its row, column, and block.
        """
        row, col = square
        return (
            self.possible_values_in_row(row) & 
            self.possible_values_in_column(col) & 
            self.possible_values_in_block(row, col)
        )
    
    def update_cache(self, move: Move, restore: bool = False):
        (row, col), value = move.square, move.value
        i, j = row // self.board.m, col // self.board.n
        if restore: # move is undone
            self.row_cache[row].add(value)
            self.col_cache[col].add(value)
            self.block_cache[i][j].add(value)
        else: # move is done
            self.row_cache[row].remove(value)
            self.col_cache[col].remove(value)
            self.block_cache[i][j].remove(value)


class SudokuAI(competitive_sudoku.sudokuai.SudokuAI):
    """
    An AI player for a competitive Sudoku game. Implements various strategies 
    for move generation and heuristics, as well as Minimax for decision-making.
    """
    def __init__(self):
        super().__init__()

    def generate_legal_moves(self, return_dict=False):
        """
        Generates all legal moves for the current player.
        If `return_dict` is True, returns a dictionary mapping squares to their possible values.
        Otherwise, returns a flat list of moves.
        """
        if not return_dict:
            moves = []
            for sq in self.gs.player_squares():
                moves += [
                    Move(sq, value)
                    for value in self.board.possible_values_for_square(sq)
                    if TabooMove(sq, value) not in self.gs.taboo_moves
                ]
            return moves
        else:
            moves = {}
            for sq in self.gs.player_squares():
                moves[sq] = {
                    value
                    for value in self.board.possible_values_for_square(sq)
                    if TabooMove(sq, value) not in self.gs.taboo_moves
                }
            return moves
        
    def only_squares_rule(self, moves: dict) -> dict:
        """
        Refine possible values for Sudoku squares using the "only squares" rule.
        This rule identifies values that can only exist in one square within a group
        (row, column, or block), eliminating them from other squares.
        """
        def find_unique_candidates(group, legal_values: set[int]) -> set[int]:
            """
            Determine if any values in the square's possible values are unique to it 
            within the group.
            """
            # Collect all possible values in the group
            group_values = set()
            for square in group:
                group_values.update(self.board.possible_values_for_square(square))
            
            unique_values = legal_values - group_values
            return unique_values if unique_values else legal_values

        refined_moves = copy.deepcopy(moves)
        for square, legal_values in moves.items():
            # Skip squares with only one possible value
            if len(legal_values) == 1:
                continue

            # Get groups for the square (excluding the square itself)
            groups = [
                self.board.get_row_squares(square[0]) - {square},
                self.board.get_column_squares(square[1]) - {square},
                self.board.get_block_squares(square[0], square[1]) - {square},
            ]
            # Filter out squares that are not empty
            groups = [{sq for sq in group if self.gs.board.get(sq) == SudokuBoard.empty} for group in groups]

            for group in groups:
                # Apply the rule to find unique candidates
                unique_values = find_unique_candidates(group, legal_values)
                
                # Narrow down possible values to the unique candidates
                refined_moves[square] = legal_values & unique_values   # no need, put it directly
                
                # If only one value remains, stop checking other groups
                if len(refined_moves[square]) == 1:
                    break

        return refined_moves
     
    def hidden_twin_rule(self, moves: dict, group_size=2) -> dict:
        """Refine possible values for Sudoku squares using the hidden twin rule."""
        
        def find_hidden_subset(vals_per_group, legal_values):
            """
            Identify a hidden subset of given size in the group's candidates.
            Returns the subset if found, otherwise returns the original candidates.
            """
            for subset in combinations(legal_values, group_size):
                subset = set(subset)
                # Count squares where the subset is present
                if sum([subset.issubset(group_vals) for group_vals in vals_per_group]) == group_size - 1:
                    return subset  # Valid subset found
            return legal_values
        
        refined_moves = copy.deepcopy(moves)
        for sq, legal_vals in moves.items():
            # Skip squares with too few possible values
            if len(legal_vals) <= group_size:
                continue
            
            # Evaluate groups for the current square
            groups = [
                self.board.get_row_squares(sq[0]) - {sq},
                self.board.get_column_squares(sq[1]) - {sq},
                self.board.get_block_squares(sq[0], sq[1]) - {sq},
            ]
            groups = [{sq for sq in group if self.gs.board.get(sq) == SudokuBoard.empty} for group in groups]
            
            for group in groups:
                # Get candidate values for squares in the group
                group_candidates = [self.board.possible_values_for_square(sq) for sq in group]
                
                # Find hidden subset
                hidden_subset = find_hidden_subset(group_candidates, legal_vals)
                
                if len(hidden_subset) == group_size:
                    # Narrow down the square's possible values to the hidden subset
                    refined_moves[sq] = legal_vals & hidden_subset
                    break
        
        return refined_moves

    def naked_twin_rule(self, moves: dict, group_size=2) -> dict:
        """
        Refine possible values for Sudoku squares using the "naked twin" rule.
        This rule identifies a subset of values that appear in exactly `group_size` squares
        within a group (row, column, or block) and removes these values from other squares in the group.
        """
        def eliminate_naked_subset(group, possible_values: set[int]) -> set[int]:
            """
            Identify a subset of size `group_size` that appears in exactly `group_size` squares
            and eliminate it from the current square's possible values.
            """
            # Collect all possible values for the group
            group_candidates = [self.board.possible_values_for_square(square) for square in group]
            all_candidates = set().union(*group_candidates)

            # Check all subsets of the given size
            for subset in combinations(all_candidates, group_size):
                subset = set(subset)
                
                # Skip subsets that do not intersect with the current square's values
                if not subset & possible_values:
                    continue

                # Count how many squares exactly match this subset
                matching_squares = sum(1 for candidates in group_candidates if subset == candidates)
                if matching_squares == group_size:
                    # Subset is a valid naked twin; remove these values
                    return possible_values - subset

            return possible_values

        refined_moves = copy.deepcopy(moves)
        for square, possible_values in moves.items():
            # Skip squares with too few possible values
            if len(possible_values) <= group_size:
                continue

            groups = [
                self.board.get_row_squares(square[0]) - {square},
                self.board.get_column_squares(square[1]) - {square},
                self.board.get_block_squares(square[0], square[1]) - {square},
            ]
            groups = [{sq for sq in group if self.gs.board.get(sq) == SudokuBoard.empty} for group in groups]

            for group in groups:
                # Apply the rule to eliminate naked subsets
                refined_values = eliminate_naked_subset(group, possible_values)

                if refined_values != refined_moves[square]:
                    # Update the square's possible values if they changed
                    refined_moves[square] = refined_values
                    break

        return refined_moves

    def apply_heuristics(self, moves_dict: Dict[Square, int]) -> List[Move]:
        """
        Applies heuristic rules to refine possible moves.
        """
        moves_dict = self.only_squares_rule(moves_dict)
        moves_dict = self.naked_twin_rule(moves_dict)
        moves_dict = self.hidden_twin_rule(moves_dict)
            
        out = []
        for square, values in moves_dict.items():
            for value in values:
                out.append(Move(square, value))

        return out
  
    def generate_next_moves(self, heuristic: bool = True, first_stage: bool = False) -> List[Move]:
        """
        Generates the next moves for the AI, optionally applying heuristics for optimization.
        """
        moves = self.generate_legal_moves(return_dict=heuristic)
        
        if first_stage:
            if isinstance(moves, dict):
                sq, value = random.choice(list(moves.items()))
                self.propose_move(Move(sq, value))
            else:
                self.propose_move(random.choice(moves))

        if heuristic:
            moves = self.apply_heuristics(moves)
        
        # Sort moves by region completion score
        moves = sorted(moves, key=lambda move: self.compute_region_completion_score(move), reverse=True)

        return moves

    def eval(self) -> int:
        return self.gs.scores[self.my_player - 1] - self.gs.scores[self.opponent_player - 1]

    def is_terminal(self) -> bool:
        return not any([self.gs.board.get(sq) == SudokuBoard.empty for sq in self.gs.player_squares()])

    def compute_region_completion_score(self, move: Move) -> int:
        (row, col), value = move.square, move.value
        return SCORE_DICT[sum([
            self.board.possible_values_in_row(row) == {value},
            self.board.possible_values_in_column(col) == {value},
            self.board.possible_values_in_block(row, col) == {value}
        ])]

    def forward(self, move: Move):
        """
        Updates the game state by applying the given move.
        """
        score = self.compute_region_completion_score(move)
        self.gs.scores[self.gs.current_player - 1] += score
        self.history_scores.append(score)
        
        self.board.update_cache(move, restore=False)
        
        self.gs.board.put(move.square, move.value)
        self.gs.occupied_squares().append(move.square)
        self.gs.moves.append(move)

        self.gs.current_player = 3 - self.gs.current_player

    def backward(self):
        """
        Reverts the game state to its previous state (undoing the last move).
        """
        self.gs.current_player = 3 - self.gs.current_player
        
        move = self.gs.moves.pop()
        self.gs.occupied_squares().pop()
        self.gs.board.put(move.square, SudokuBoard.empty)
        
        self.board.update_cache(move, restore=True)
    
        self.gs.scores[self.gs.current_player - 1] -= self.history_scores.pop()
    
    def minimax(
        self, 
        depth: int, 
        maximizing: bool, 
        alpha: float, 
        beta: float, 
        first_stage: bool, 
        moves: List[Move] = None,
        heuristic: bool = True
    ) -> float:
        if depth == 0 or self.is_terminal():
            return self.eval()

        if moves is None:
            moves = self.generate_next_moves(heuristic=heuristic)

        if maximizing:
            max_score = float('-inf')
            for move in moves:
                self.forward(move)
                score = self.minimax(
                    depth=depth - 1, 
                    maximizing=False, 
                    alpha=alpha, 
                    beta=beta, 
                    first_stage=False,
                    heuristic=heuristic
                )
                self.backward()
                
                if score > max_score:
                    max_score = score
                    if first_stage and score > self.global_max_score:
                        self.global_max_score = score
                        self.propose_move(move)

                alpha = max(alpha, score)
                if alpha >= beta:
                    break

            return max_score
        else:
            min_score = float('-inf')
            for move in moves:
                self.forward(move)
                score = self.minimax(
                    depth=depth - 1, 
                    maximizing=True, 
                    alpha=alpha, 
                    beta=beta, 
                    first_stage=False,
                    heuristic=heuristic
                )
                self.backward()

                if score < min_score:
                    min_score = score

                beta = min(beta, score)
                if beta <= alpha:
                    break

            return min_score

    def setup(self, game_state: GameState) -> None:
        """
        Sets up the AI with the given game state, initializing caches and variables.
        The game state is not used in the next turn since the setup is done at the beginning of each turn.
        """
        self.gs = game_state
        self.my_player = self.gs.current_player
        self.opponent_player = 3 - self.gs.current_player

        self.history_scores = []
        self.board = MemorySudokuBoard(self.gs.board)
        
        self.global_max_score = float('-inf')

    def compute_best_move(self, game_state: GameState) -> None:
        self.setup(game_state)
        depth, heuristic = 1, False
        
        moves = self.generate_next_moves(heuristic=heuristic, first_stage=True)
        self.propose_move(moves[0])

        while True:
            self.minimax(
                depth=depth,
                maximizing=True,
                alpha=float('-inf'),
                beta=float('inf'),
                first_stage=True,
                moves=moves,
                heuristic=heuristic
            )
            depth += 1
