import numpy as np
import chess


def board_to_tensor(board: chess.Board, flip: bool = False) -> np.ndarray:
    """
    Encode a chess position as an (18, 8, 8) float32 tensor.

    Channels:
      0-5:   Side-to-move's own pieces (P, N, B, R, Q, K)
      6-11:  Opponent's pieces (P, N, B, R, Q, K)
      12:    Constant ones plane
      13:    Side-to-move kingside castling right
      14:    Side-to-move queenside castling right
      15:    Opponent kingside castling right
      16:    Opponent queenside castling right
      17:    En passant target square

    If flip=True, vertically flip the board (useful when Fischer played Black).
    """
    tensor = np.zeros((18, 8, 8), dtype=np.float32)

    # Map piece types to channel offsets
    piece_to_channel = {
        chess.PAWN: 0,
        chess.KNIGHT: 1,
        chess.BISHOP: 2,
        chess.ROOK: 3,
        chess.QUEEN: 4,
        chess.KING: 5,
    }

    stm = board.turn  # side to move
    opponent = not stm

    # Fill piece channels
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece is None:
            continue

        rank, file = divmod(sq, 8)
        if flip:
            rank = 7 - rank

        channel_offset = piece_to_channel[piece.piece_type]

        if piece.color == stm:
            tensor[channel_offset, rank, file] = 1.0
        else:
            tensor[6 + channel_offset, rank, file] = 1.0

    # Constant ones plane
    tensor[12, :, :] = 1.0

    # Castling rights
    if board.has_kingside_castling_rights(stm):
        tensor[13, :, :] = 1.0
    if board.has_queenside_castling_rights(stm):
        tensor[14, :, :] = 1.0
    if board.has_kingside_castling_rights(opponent):
        tensor[15, :, :] = 1.0
    if board.has_queenside_castling_rights(opponent):
        tensor[16, :, :] = 1.0

    # En passant
    if board.ep_square is not None:
        rank, file = divmod(board.ep_square, 8)
        if flip:
            rank = 7 - rank
        tensor[17, rank, file] = 1.0

    return tensor


def move_to_index(move: chess.Move, flip: bool = False) -> int:
    """Convert a chess.Move to an integer index 0-4095 (from_sq * 64 + to_sq)."""
    from_sq = move.from_square
    to_sq = move.to_square

    if flip:
        from_sq = (7 - (from_sq // 8)) * 8 + (from_sq % 8)
        to_sq = (7 - (to_sq // 8)) * 8 + (to_sq % 8)

    return from_sq * 64 + to_sq


def index_to_move(idx: int, board: chess.Board, flip: bool = False) -> chess.Move:
    """
    Convert an integer index (0-4095) back to a chess.Move.
    If the resulting move is not legal, try with queen promotion.
    """
    from_sq = idx // 64
    to_sq = idx % 64

    if flip:
        from_sq = (7 - (from_sq // 8)) * 8 + (from_sq % 8)
        to_sq = (7 - (to_sq // 8)) * 8 + (to_sq % 8)

    move = chess.Move(from_sq, to_sq)

    if move in board.legal_moves:
        return move

    # Try with queen promotion
    move_queen = chess.Move(from_sq, to_sq, promotion=chess.QUEEN)
    if move_queen in board.legal_moves:
        return move_queen

    # Fallback: return the non-promotion move (may be illegal)
    return move


def get_legal_mask(board: chess.Board, flip: bool = False) -> np.ndarray:
    """Return a binary mask of shape (4096,) indicating legal moves."""
    mask = np.zeros(4096, dtype=np.float32)
    for move in board.legal_moves:
        idx = move_to_index(move, flip)
        mask[idx] = 1.0
    return mask
