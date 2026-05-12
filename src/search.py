import torch
import chess
from src.nnue import SimpleNNUE, get_position_eval
from src.model import FischerNet
from src.encode import board_to_tensor, get_legal_mask, index_to_move


class HybridEngine:
    """
    Hybrid chess engine combining:
    - Fischer policy model for move ordering
    - NNUE eval for position evaluation
    - Minimax + alpha-beta pruning for search
    """

    def __init__(self, policy_model: FischerNet, eval_model: SimpleNNUE, device: str = "cpu"):
        self.policy_model = policy_model
        self.eval_model = eval_model
        self.device = device
        self.eval_model.eval()
        self.policy_model.eval()

        self.nodes_searched = 0

    def get_move_scores(self, board: chess.Board) -> list[tuple[chess.Move, float]]:
        """
        Use Fischer policy model to score and rank legal moves.
        Higher score = model thinks Fischer would play it.
        """
        legal_moves = list(board.legal_moves)

        if not legal_moves:
            return []

        with torch.no_grad():
            tensor = board_to_tensor(board, flip=False)
            tensor = torch.from_numpy(tensor).unsqueeze(0).to(self.device)

            logits = self.policy_model(tensor).squeeze(0).cpu().numpy()

            # Get scores for each legal move
            move_scores = []
            for move in legal_moves:
                from_idx = move.from_square
                to_idx = move.to_square
                move_idx = from_idx * 64 + to_idx

                score = logits[move_idx]
                move_scores.append((move, score))

            # Sort by model confidence (descending)
            move_scores.sort(key=lambda x: x[1], reverse=True)

        return move_scores

    def evaluate_material(self, board: chess.Board) -> float:
        """Simple material evaluation. Positive = white advantage."""
        piece_values = {chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330, chess.ROOK: 500, chess.QUEEN: 900}

        white_material = 0
        black_material = 0

        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                value = piece_values.get(piece.piece_type, 0)
                if piece.color == chess.WHITE:
                    white_material += value
                else:
                    black_material += value

        return white_material - black_material

    def evaluate_king_safety(self, board: chess.Board) -> float:
        """
        Evaluate king safety for both sides.
        Positive = White's king safer (or Black's king in danger).
        """
        white_king_sq = board.king(chess.WHITE)
        black_king_sq = board.king(chess.BLACK)

        # Castling rights = more king safety
        white_castling = 0
        if board.has_kingside_castling_rights(chess.WHITE):
            white_castling += 30
        if board.has_queenside_castling_rights(chess.WHITE):
            white_castling += 30

        black_castling = 0
        if board.has_kingside_castling_rights(chess.BLACK):
            black_castling += 30
        if board.has_queenside_castling_rights(chess.BLACK):
            black_castling += 30

        # Pawn shield (pawns near king = safer)
        white_pawn_shield = len([sq for sq in chess.SQUARES if board.piece_at(sq) == chess.Piece(chess.PAWN, chess.WHITE) and abs(sq - white_king_sq) < 12])
        black_pawn_shield = len([sq for sq in chess.SQUARES if board.piece_at(sq) == chess.Piece(chess.PAWN, chess.BLACK) and abs(sq - black_king_sq) < 12])

        return (white_castling + white_pawn_shield * 10) - (black_castling + black_pawn_shield * 10)

    def evaluate_piece_activity(self, board: chess.Board) -> float:
        """
        Reward active pieces (attacking, centralized, advanced).
        Fischer preferred active pieces over passive defense.
        """
        activity_bonus = 0

        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if not piece:
                continue

            rank, file = divmod(square, 8)

            # Pieces in center (d4, e4, d5, e5) = bonus
            if 3 <= file <= 4 and 3 <= rank <= 4:
                bonus = 15 if piece.color == chess.WHITE else -15
                activity_bonus += bonus

            # Pieces on 4th rank or better (White) / 5th rank or better (Black) = aggressive
            if piece.color == chess.WHITE and rank >= 3:
                activity_bonus += 20
            elif piece.color == chess.BLACK and rank <= 4:
                activity_bonus -= 20

            # Attacking opponent pieces = big bonus
            attacked_squares = board.attacks(square)
            num_attacks = len([sq for sq in attacked_squares if board.piece_at(sq) and board.piece_at(sq).color != piece.color])
            activity_bonus += num_attacks * (15 if piece.color == chess.WHITE else -15)

        return activity_bonus

    def evaluate_forcing_moves(self, board: chess.Board) -> float:
        """
        Bonus for forcing positions (checks, captures imminent, passed pawns).
        Fischer loved concrete, forcing play.
        """
        forcing_bonus = 0

        # Checks available
        for move in board.legal_moves:
            if board.is_check():
                forcing_bonus += 30
                break

        # Passed pawns (very valuable, especially advanced)
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.piece_type == chess.PAWN:
                rank, file = divmod(square, 8)

                # Check if passed (no enemy pawns in front)
                is_passed = True
                if piece.color == chess.WHITE:
                    for check_rank in range(rank + 1, 8):
                        for check_file in [file - 1, file, file + 1]:
                            if 0 <= check_file < 8:
                                check_sq = check_rank * 8 + check_file
                                enemy_piece = board.piece_at(check_sq)
                                if enemy_piece and enemy_piece.piece_type == chess.PAWN and enemy_piece.color == chess.BLACK:
                                    is_passed = False
                    if is_passed:
                        forcing_bonus += (7 - rank) * 20  # More advanced = more valuable

                else:  # Black pawn
                    for check_rank in range(rank - 1, -1, -1):
                        for check_file in [file - 1, file, file + 1]:
                            if 0 <= check_file < 8:
                                check_sq = check_rank * 8 + check_file
                                enemy_piece = board.piece_at(check_sq)
                                if enemy_piece and enemy_piece.piece_type == chess.PAWN and enemy_piece.color == chess.WHITE:
                                    is_passed = False
                    if is_passed:
                        forcing_bonus -= rank * 20

        return forcing_bonus

    def evaluate_position(self, board: chess.Board) -> float:
        """
        Evaluate position using NNUE + material + Fischer-style bonuses.
        Returns eval in centipawns from side-to-move's perspective.
        """
        # Get NNUE eval
        nnue_eval = get_position_eval(self.eval_model, board, device=self.device)

        # Get material eval
        material_eval = self.evaluate_material(board)

        # Fischer-style bonuses
        king_safety = self.evaluate_king_safety(board)
        piece_activity = self.evaluate_piece_activity(board)
        forcing_moves = self.evaluate_forcing_moves(board)

        # Blend with Fischer aggressive weighting:
        # - NNUE: 50% (strategic understanding)
        # - Material: 20% (foundation)
        # - Activity: 20% (Fischer loved active pieces)
        # - King Safety: 5% (awareness, not defensive)
        # - Forcing: 5% (love of concrete play)
        eval_score = (
            nnue_eval * 0.50 +
            material_eval * 0.20 +
            piece_activity * 0.20 +
            king_safety * 0.05 +
            forcing_moves * 0.05
        )

        # Flip if black to move (return from side-to-move perspective)
        if board.turn == chess.BLACK:
            eval_score = -eval_score

        return eval_score

    def minimax(
        self, board: chess.Board, depth: int, alpha: float = -99999, beta: float = 99999, maximizing: bool = True
    ) -> float:
        """
        Minimax with alpha-beta pruning.
        Returns best eval score.
        """
        self.nodes_searched += 1

        # Terminal node: evaluate
        if depth == 0 or board.is_game_over():
            if board.is_checkmate():
                return 99999 if not board.turn == chess.WHITE else -99999
            if board.is_stalemate() or board.is_insufficient_material():
                return 0
            return self.evaluate_position(board)

        # Get moves sorted by Fischer model confidence
        move_scores = self.get_move_scores(board)

        if maximizing:  # White maximizing
            max_eval = -99999
            for move, _ in move_scores:
                board.push(move)
                eval_score = self.minimax(board, depth - 1, alpha, beta, False)
                board.pop()

                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)

                if beta <= alpha:
                    break  # Beta cutoff

            return max_eval

        else:  # Black minimizing
            min_eval = 99999
            for move, _ in move_scores:
                board.push(move)
                eval_score = self.minimax(board, depth - 1, alpha, beta, True)
                board.pop()

                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)

                if beta <= alpha:
                    break  # Alpha cutoff

            return min_eval

    def choose_move(self, board: chess.Board, depth: int = 3) -> chess.Move:
        """
        Choose best move using minimax search.
        """
        self.nodes_searched = 0

        # Get all moves ranked by Fischer model
        move_scores = self.get_move_scores(board)

        if not move_scores:
            return None

        best_move = None
        best_eval = -99999 if board.turn == chess.WHITE else 99999

        maximizing = board.turn == chess.WHITE

        for move, model_score in move_scores:
            board.push(move)
            eval_score = self.minimax(board, depth - 1, -99999, 99999, not maximizing)
            board.pop()

            is_better = eval_score > best_eval if maximizing else eval_score < best_eval

            if is_better:
                best_eval = eval_score
                best_move = move

        return best_move
