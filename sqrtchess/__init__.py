"""根号棋（Square Root Chess）引擎。

    from sqrtchess import Game, START_FEN, Recorder, replay

    game = Game.new()                 # 新的一局
    print(game.fen())                 # 局面文本
    info = game.push("1b2-b3")        # 走一步
    print(game.legal_moves())         # 合法着法
    print(game.result)                # 结果（未结束时为 None）

更多接口见 README.md、docs/PROTOCOL.md。
"""

from .core import (
    BLACK, BOARD_SIZE, DEFAULT_NO_CAPTURE_LIMIT, FILES, INITIAL_EDGE_ROW,
    INITIAL_FRONT_ROW, NUM_SQUARES, REASON_TEXT, SQUARE_NAMES, WHITE,
    Game, GameOver, GameResult, IllegalMove, Move, MoveInfo,
    color_from_letter, color_letter, color_name, format_move_text,
    initial_cells, is_legal_displacement, move_offsets, opponent,
    parse_move_text, parse_square, perft, perft_divide, square_name,
)
from .fen import (START_FEN, FenData, FenError, format_board_field, format_fen,
                  parse_board_field, parse_fen)
from .record import (RESULT_BLACK, RESULT_DRAW, RESULT_UNKNOWN, RESULT_WHITE,
                     RULES_TAG, Record, Recorder, RecordError, Replay,
                     build_record, default_headers, format_record,
                     parse_record, parse_records, replay, result_winner,
                     winner_result)
from .ai import (AI_LEVEL_NAMES, AI_LEVELS, BaseAI, CaptureAI, GreedyAI,
                 RandomAI, SearchAI, choose_move, evaluate, make_ai)
from .protocol import (EngineSession, PROTOCOL_VERSION, RULES_SUMMARY,
                       handle_ai_request, run_ai_stdio, run_stdio, serve_forever)

__version__ = "1.0.0"

__all__ = [
    "__version__",
    # core
    "Game", "Move", "MoveInfo", "GameResult", "IllegalMove", "GameOver",
    "WHITE", "BLACK", "BOARD_SIZE", "NUM_SQUARES", "FILES", "SQUARE_NAMES",
    "DEFAULT_NO_CAPTURE_LIMIT", "INITIAL_EDGE_ROW", "INITIAL_FRONT_ROW", "REASON_TEXT",
    "initial_cells", "move_offsets", "is_legal_displacement", "square_name",
    "parse_square", "format_move_text", "parse_move_text", "color_name",
    "color_letter", "color_from_letter", "opponent", "perft", "perft_divide",
    # fen
    "START_FEN", "FenData", "FenError", "format_fen", "parse_fen",
    "format_board_field", "parse_board_field",
    # record
    "Record", "Recorder", "Replay", "RecordError", "parse_record", "parse_records",
    "format_record", "build_record", "replay", "default_headers", "result_winner",
    "winner_result", "RULES_TAG", "RESULT_WHITE", "RESULT_BLACK", "RESULT_DRAW",
    "RESULT_UNKNOWN",
    # ai
    "BaseAI", "RandomAI", "GreedyAI", "SearchAI", "CaptureAI", "make_ai", "choose_move",
    "evaluate", "AI_LEVELS", "AI_LEVEL_NAMES",
    # protocol
    "EngineSession", "PROTOCOL_VERSION", "RULES_SUMMARY", "handle_ai_request",
    "run_stdio", "run_ai_stdio", "serve_forever",
]
