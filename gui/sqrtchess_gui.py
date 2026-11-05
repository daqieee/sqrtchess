"""根号棋图形界面（tkinter 实现，只用标准库）。

用法::

    python gui/sqrtchess_gui.py            # 直接运行
    python -m sqrtchess gui                # 通过引擎命令行启动
    start_gui.bat                          # Windows 双击启动
    python gui/sqrtchess_gui.py --selftest # 自检（自动跑一遍界面流程后退出）

功能：连接引擎实时对局或观战、悔棋、翻转棋盘、载入棋谱（.sqg）、写入 FEN 查看任意局面、
双方 AI 自对弈、外部 AI 程序对弈、导出棋谱与 FEN。
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from sqrtchess import (BLACK, DEFAULT_NO_CAPTURE_LIMIT, Game, GameOver,
                       IllegalMove, START_FEN, WHITE, color_name, default_headers,
                       format_record, make_ai, parse_move_text, replay, square_name)
from sqrtchess.fen import FenError
from sqrtchess.record import Record, RecordError

CELL = 52
MARGIN = 42
BOARD_PX = CELL * 10 + MARGIN * 2
CELL_MIN = 28
CELL_MAX = 110          # 自动适配时的上限
CELL_MAX_ABS = 132      # 手动放大时的硬上限
FILES = "abcdefghijk"

PLAYER_CHOICES: Tuple[Tuple[str, str], ...] = (
    ("human", "人类"),
    ("azero", "AlphaZero 网络"),
    ("hybrid", "AlphaZero 混合搜索"),
    ("ab", "Alpha-Beta 引擎（3 层）"),
    ("capture", "吃子基线 AI"),
    ("greedy", "贪心 AI"),
    ("random", "随机 AI"),
    ("search2", "搜索 2 层"),
    ("external", "外部程序"),
)
PLAYER_KEYS = [key for key, _ in PLAYER_CHOICES]
PLAYER_LABELS = [label for _, label in PLAYER_CHOICES]
LABEL_TO_KEY = {label: key for key, label in PLAYER_CHOICES}
KEY_TO_LABEL = {key: label for key, label in PLAYER_CHOICES}

COLOR_BG = "#f2dfb6"
COLOR_LINE = "#9a6b3f"
COLOR_POINT = "#6d4726"
COLOR_WHITE_PIECE = "#fbfaf5"
COLOR_BLACK_PIECE = "#33383d"
COLOR_SELECT = "#1769ff"
COLOR_TARGET = "#1f8b4c"
COLOR_CAPTURE = "#c62828"
COLOR_LAST = "#f7b731"


# --------------------------------------------------------------------------- 外部进程


class LineProcess:
    """以 JSON 行协议通信的外部引擎/AI 程序。"""

    def __init__(self, command: str, expect_greeting: bool = False, name: str = "外部程序"):
        self.command = command
        self.name = name
        env = dict(os.environ)
        env["PYTHONUTF8"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        # 让 "python -m sqrtchess ..." 在任何工作目录下都能找到引擎包
        env["PYTHONPATH"] = str(ROOT_DIR) + os.pathsep + env.get("PYTHONPATH", "")
        creation = 0
        if os.name == "nt":
            creation = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        self.process = subprocess.Popen(
            command, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace",
            bufsize=1, env=env, creationflags=creation)
        self.queue: "queue.Queue[str]" = queue.Queue()
        self.errors: List[str] = []
        self.greeting: Optional[dict] = None
        threading.Thread(target=self._pump_stdout, daemon=True).start()
        threading.Thread(target=self._pump_stderr, daemon=True).start()
        if expect_greeting:
            line = self._read_line(timeout=15.0)
            if line:
                try:
                    self.greeting = json.loads(line)
                except json.JSONDecodeError:
                    self.greeting = {"raw": line}

    # -------------------------------------------------- 线程

    def _pump_stdout(self) -> None:
        assert self.process.stdout is not None
        for line in self.process.stdout:
            self.queue.put(line)
        self.queue.put("")

    def _pump_stderr(self) -> None:
        assert self.process.stderr is not None
        for line in self.process.stderr:
            self.errors.append(line.rstrip())
            del self.errors[:-50]

    def _read_line(self, timeout: float = 20.0) -> str:
        try:
            line = self.queue.get(timeout=timeout)
        except queue.Empty:
            return ""
        return line.strip()

    # -------------------------------------------------- 接口

    @property
    def alive(self) -> bool:
        return self.process.poll() is None

    def request(self, payload: dict, timeout: float = 20.0) -> dict:
        if not self.alive:
            raise RuntimeError("外部程序已经退出：%s" % (self.errors[-1] if self.errors else ""))
        assert self.process.stdin is not None
        self.process.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self.process.stdin.flush()
        line = self._read_line(timeout=timeout)
        if not line:
            raise RuntimeError("外部程序没有响应（超时或已退出）")
        return json.loads(line)

    def close(self) -> None:
        try:
            if self.alive and self.process.stdin is not None:
                self.process.stdin.write(json.dumps({"cmd": "quit"}) + "\n")
                self.process.stdin.flush()
        except Exception:
            pass
        try:
            self.process.terminate()
        except Exception:
            pass


# --------------------------------------------------------------------------- 界面


class SqrtChessApp:
    """根号棋图形界面主程序。"""

    def __init__(self, root: tk.Tk, no_capture_limit=DEFAULT_NO_CAPTURE_LIMIT,
                 silent: bool = False):
        self.root = root
        self.limit = no_capture_limit
        self.silent = silent

        # 对局状态
        self.start_fen: str = START_FEN
        self.line: List[str] = []
        self.game: Game = Game.new(no_capture_limit=no_capture_limit)
        self.headers: Dict[str, str] = default_headers()
        self.declared: Optional[str] = None
        self.selected: Optional[int] = None
        self.targets: Dict[int, bool] = {}
        self.flip: bool = False
        self.result_shown_for: Optional[int] = None

        # 播放与 AI
        self.white_key = "human"
        self.black_key = "greedy"
        self.az_players = {}
        self.auto_play = False
        self.delay_ms = 400
        self.thinking = False
        self.ai_queue: "queue.Queue[Tuple[str, str]]" = queue.Queue()
        self.external_players: Dict[str, LineProcess] = {}
        self.engine_conn: Optional[LineProcess] = None
        self.ai_failures = 0
        self.last_tick = 0
        # 斗蛐蛐（多局外部 AI 对战）
        self.duel_total = 0
        self.duel_played = 0
        self.duel_scores: Dict[str, float] = {}
        self.duel_labels: Dict[int, str] = {}

        self._build_style()
        self.base_cell = self._autofit_cell()
        self.zoom_factor = 1.0
        self._set_cell(self.base_cell)
        self._build_menu()
        self._build_layout()
        self.new_game()
        self.root.after(120, self._tick)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    # -------------------------------------------------- 构建界面

    def _build_style(self) -> None:
        self.font_ui = ("Microsoft YaHei UI", 10)
        self.font_small = ("Microsoft YaHei UI", 9)
        self.font_mono = ("Consolas", 9)
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("TLabel", font=self.font_ui)
        style.configure("TButton", font=self.font_ui)
        style.configure("TCheckbutton", font=self.font_ui)
        style.configure("TLabelframe.Label", font=self.font_ui)
        style.configure("Status.TLabel", font=("Microsoft YaHei UI", 11, "bold"))
        self.root.title("根号棋 Square Root Chess")
        # 右侧面板控件较多，窗口需要足够宽；直接最大化最省事，也能自动适配任何 DPI 缩放
        try:
            self.root.state("zoomed")
        except Exception:  # noqa: BLE001
            self.root.geometry("1240x900")
        self.root.minsize(980, 680)
        self.root.configure(bg="#2f3336")

    def _build_menu(self) -> None:
        menubar = tk.Menu(self.root)

        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="新对局", accelerator="Ctrl+N", command=self.new_game)
        file_menu.add_command(label="打开棋谱…", accelerator="Ctrl+O", command=self.open_record)
        file_menu.add_command(label="保存棋谱…", accelerator="Ctrl+S", command=self.save_record)
        file_menu.add_separator()
        file_menu.add_command(label="复制当前 FEN", command=self.copy_fen)
        file_menu.add_command(label="载入 FEN 文本…", command=self.focus_fen_entry)
        file_menu.add_separator()
        file_menu.add_command(label="退出", command=self.on_close)
        menubar.add_cascade(label="文件", menu=file_menu)

        game_menu = tk.Menu(menubar, tearoff=0)
        game_menu.add_command(label="悔棋", accelerator="Ctrl+Z", command=self.undo)
        game_menu.add_command(label="翻转棋盘", accelerator="F", command=self.flip_board)
        game_menu.add_separator()
        game_menu.add_command(label="当前方认输", command=self.resign)
        game_menu.add_command(label="判和", command=self.agree_draw)
        game_menu.add_separator()
        game_menu.add_command(label="自动播放棋谱", command=self.toggle_autoplay)
        game_menu.add_command(label="双方 AI 自对弈", command=self.self_play)
        game_menu.add_command(label="斗蛐蛐（多局外部 AI 对战）", command=self.start_duel)
        game_menu.add_command(label="回到开局", accelerator="Home", command=lambda: self.goto(0))
        game_menu.add_command(label="跳到最新", accelerator="End", command=self.goto_end)
        menubar.add_cascade(label="对局", menu=game_menu)

        engine_menu = tk.Menu(menubar, tearoff=0)
        engine_menu.add_command(label="连接外部引擎进程…", command=self.connect_engine)
        engine_menu.add_command(label="查看引擎信息", command=self.show_engine_info)
        engine_menu.add_command(label="断开连接", command=self.disconnect_engine)
        engine_menu.add_separator()
        engine_menu.add_command(label="引擎/协议说明", command=self.show_protocol_help)
        menubar.add_cascade(label="引擎", menu=engine_menu)

        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="根号棋规则", command=self.show_rules)
        help_menu.add_command(label="记谱与 FEN 说明", command=self.show_notation_help)
        help_menu.add_command(label="关于", command=self.show_about)
        menubar.add_cascade(label="帮助", menu=help_menu)

        self.root.config(menu=menubar)
        self.root.bind("<Control-n>", lambda event: self.new_game())
        self.root.bind("<Control-o>", lambda event: self.open_record())
        self.root.bind("<Control-s>", lambda event: self.save_record())
        self.root.bind("<Control-z>", lambda event: self.undo())
        self.root.bind("<Left>", lambda event: self.step(-1))
        self.root.bind("<Right>", lambda event: self.step(1))
        self.root.bind("<Home>", lambda event: self.goto(0))
        self.root.bind("<End>", lambda event: self.goto_end())
        self.root.bind("<f>", lambda event: self.flip_board())
        self.root.bind("<Control-plus>", lambda event: self.zoom_step(1))
        self.root.bind("<Control-equal>", lambda event: self.zoom_step(1))
        self.root.bind("<Control-minus>", lambda event: self.zoom_step(-1))

    # -------------------------------------------------- 棋盘尺寸

    def _autofit_cell(self) -> int:
        """按屏幕大小挑一个够大的格子（右侧面板约 700 像素）。"""
        try:
            screen_w = self.root.winfo_screenwidth()
            screen_h = self.root.winfo_screenheight()
        except Exception:  # noqa: BLE001
            return 52
        # 画布总高约 11.6 个格子（10 格 + 上下留白），右侧面板约 700 像素
        by_width = (screen_w - 700 - 80) / 11.6
        by_height = (screen_h - 200 - 60) / 11.6
        return int(max(CELL_MIN, min(CELL_MAX, min(by_width, by_height))))

    def _set_cell(self, cell: int) -> None:
        global CELL, MARGIN, BOARD_PX
        cell = int(max(CELL_MIN, min(CELL_MAX_ABS, cell)))
        CELL = cell
        MARGIN = max(26, int(round(cell * 0.8)))
        BOARD_PX = CELL * 10 + MARGIN * 2
        canvas = getattr(self, "canvas", None)
        if canvas is not None:
            canvas.configure(width=BOARD_PX, height=BOARD_PX)
            self.update_all()

    def change_zoom(self, value) -> None:
        try:
            factor = float(value)
        except (TypeError, ValueError):
            return
        self.zoom_factor = factor
        self._set_cell(int(round(self.base_cell * factor)))
        if hasattr(self, "var_zoom"):
            self.var_zoom.set("%d%%" % round(factor * 100))

    def zoom_step(self, delta: int) -> None:
        factor = max(0.6, min(1.5, self.zoom_factor + 0.1 * delta))
        if abs(factor - self.zoom_factor) < 1e-6:
            return
        self.zoom_factor = factor
        self._set_cell(int(round(self.base_cell * factor)))
        if hasattr(self, "var_zoom"):
            self.var_zoom.set("%d%%" % round(factor * 100))
        if hasattr(self, "zoom_scale"):
            self.zoom_scale.set(factor)

    def on_zoom_wheel(self, event) -> str:
        self.zoom_step(1 if getattr(event, "delta", 0) > 0 else -1)
        return "break"

    def _build_layout(self) -> None:
        outer = ttk.Frame(self.root, padding=6)
        outer.pack(fill="both", expand=True)

        left = ttk.Frame(outer)
        left.pack(side="left", fill="y")
        self.canvas = tk.Canvas(left, width=BOARD_PX, height=BOARD_PX, bg=COLOR_BG,
                                highlightthickness=1, highlightbackground="#7a5a34")
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<Control-MouseWheel>", self.on_zoom_wheel)

        right = ttk.Frame(outer, padding=(10, 0, 0, 0))
        right.pack(side="left", fill="both", expand=True)

        # 状态
        status = ttk.LabelFrame(right, text="局面状态", padding=6)
        status.pack(fill="x")
        self.var_turn = tk.StringVar(value="轮到：白方")
        self.var_nocap = tk.StringVar(value="无吃子计数：0 / 100")
        self.var_white = tk.StringVar(value="白方：18 子，大小和 56")
        self.var_black = tk.StringVar(value="黑方：18 子，大小和 56")
        self.var_result = tk.StringVar(value="对局进行中")
        ttk.Label(status, textvariable=self.var_turn, style="Status.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(status, textvariable=self.var_result).grid(row=0, column=1, sticky="w", padx=(12, 0))
        ttk.Label(status, textvariable=self.var_white).grid(row=1, column=0, sticky="w")
        ttk.Label(status, textvariable=self.var_black).grid(row=1, column=1, sticky="w", padx=(12, 0))
        ttk.Label(status, textvariable=self.var_nocap).grid(row=2, column=0, sticky="w", columnspan=2)
        status.columnconfigure(1, weight=1)

        # 按钮
        buttons = ttk.Frame(right)
        buttons.pack(fill="x", pady=4)
        ttk.Button(buttons, text="新对局", command=self.new_game).grid(row=0, column=0, sticky="ew", padx=1, pady=1)
        ttk.Button(buttons, text="悔棋", command=self.undo).grid(row=0, column=1, sticky="ew", padx=1, pady=1)
        ttk.Button(buttons, text="翻转", command=self.flip_board).grid(row=0, column=2, sticky="ew", padx=1, pady=1)
        ttk.Button(buttons, text="认输", command=self.resign).grid(row=0, column=3, sticky="ew", padx=1, pady=1)
        ttk.Button(buttons, text="判和", command=self.agree_draw).grid(row=1, column=0, sticky="ew", padx=1, pady=1)
        ttk.Button(buttons, text="复制 FEN", command=self.copy_fen).grid(row=1, column=1, sticky="ew", padx=1, pady=1)
        ttk.Button(buttons, text="打开棋谱", command=self.open_record).grid(row=1, column=2, sticky="ew", padx=1, pady=1)
        ttk.Button(buttons, text="保存棋谱", command=self.save_record).grid(row=1, column=3, sticky="ew", padx=1, pady=1)
        ttk.Button(buttons, text="斗蛐蛐（多局 AI 对战）", command=self.start_duel).grid(
            row=2, column=0, columnspan=4, sticky="ew", padx=1, pady=(6, 1))
        for column in range(4):
            buttons.columnconfigure(column, weight=1, uniform="action")

        # 棋盘缩放
        zoom_row = ttk.Frame(right)
        zoom_row.pack(fill="x", pady=(2, 2))
        ttk.Label(zoom_row, text="棋盘大小").pack(side="left")
        self.zoom_scale = ttk.Scale(zoom_row, from_=0.6, to=1.5, value=1.0, orient="horizontal",
                                    length=170, command=self.change_zoom)
        self.zoom_scale.pack(side="left", padx=6)
        self.var_zoom = tk.StringVar(value="100%")
        ttk.Label(zoom_row, textvariable=self.var_zoom, width=5).pack(side="left")
        ttk.Button(zoom_row, text="－", width=3, command=lambda: self.zoom_step(-1)).pack(side="left", padx=(8, 2))
        ttk.Button(zoom_row, text="＋", width=3, command=lambda: self.zoom_step(1)).pack(side="left")

        # 步进与播放
        nav = ttk.LabelFrame(right, text="棋谱浏览与播放", padding=6)
        nav.pack(fill="x", pady=4)
        ttk.Button(nav, text="⏮", width=3, command=lambda: self.goto(0)).grid(row=0, column=0)
        ttk.Button(nav, text="◀", width=3, command=lambda: self.step(-1)).grid(row=0, column=1)
        ttk.Button(nav, text="▶", width=3, command=lambda: self.step(1)).grid(row=0, column=2)
        ttk.Button(nav, text="⏭", width=3, command=self.goto_end).grid(row=0, column=3)
        self.var_autoplay = tk.BooleanVar(value=False)
        ttk.Checkbutton(nav, text="自动播放", variable=self.var_autoplay,
                        command=self.toggle_autoplay).grid(row=0, column=4, padx=(8, 0))
        ttk.Label(nav, text="速度").grid(row=0, column=5, padx=(8, 0))
        self.var_speed = tk.IntVar(value=self.delay_ms)
        ttk.Scale(nav, from_=80, to=1600, variable=self.var_speed,
                  command=lambda value: self.set_speed(int(float(value)))).grid(row=0, column=6, sticky="ew")
        nav.columnconfigure(6, weight=1)

        # 手动走子
        manual = ttk.Frame(right)
        manual.pack(fill="x")
        ttk.Label(manual, text="走子").pack(side="left")
        self.var_move_entry = tk.StringVar()
        entry = ttk.Entry(manual, textvariable=self.var_move_entry, font=self.font_mono)
        entry.pack(side="left", fill="x", expand=True, padx=4)
        entry.bind("<Return>", lambda event: self.play_entry_move())
        ttk.Button(manual, text="执行", width=6, command=self.play_entry_move).pack(side="left")

        # 着法列表
        moves_frame = ttk.LabelFrame(right, text="棋谱着法（点击可跳转）", padding=4)
        moves_frame.pack(fill="both", expand=True, pady=4)
        scrollbar = ttk.Scrollbar(moves_frame, orient="vertical")
        self.move_list = tk.Listbox(moves_frame, height=9, font=self.font_mono,
                                    yscrollcommand=scrollbar.set, exportselection=False)
        scrollbar.config(command=self.move_list.yview)
        self.move_list.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        self.move_list.bind("<ButtonRelease-1>", self.on_move_selected)

        # 选项卡
        notebook = ttk.Notebook(right)
        notebook.pack(fill="both", expand=True)

        fen_tab = ttk.Frame(notebook, padding=6)
        notebook.add(fen_tab, text="局面 / FEN")
        ttk.Label(fen_tab, text="把 FEN 粘贴到下面，点“载入局面”即可查看该局面：",
                  wraplength=420).pack(anchor="w")
        self.fen_text = tk.Text(fen_tab, height=4, width=42, font=self.font_mono, wrap="char")
        self.fen_text.pack(fill="both", expand=True, pady=4)
        fen_buttons = ttk.Frame(fen_tab)
        fen_buttons.pack(fill="x")
        ttk.Button(fen_buttons, text="载入局面", command=self.load_fen_from_text).pack(side="left")
        ttk.Button(fen_buttons, text="复制 FEN", command=self.copy_fen).pack(side="left", padx=4)
        ttk.Button(fen_buttons, text="初始局面", command=lambda: self.new_game()).pack(side="left")

        record_tab = ttk.Frame(notebook, padding=6)
        notebook.add(record_tab, text="棋谱")
        ttk.Label(record_tab, text="可粘贴棋谱文本后载入，或用“文件”菜单打开 .sqg 棋谱：",
                  wraplength=420).pack(anchor="w")
        self.record_text = tk.Text(record_tab, height=4, width=42, font=self.font_mono, wrap="none")
        self.record_text.pack(fill="both", expand=True, pady=4)
        record_buttons = ttk.Frame(record_tab)
        record_buttons.pack(fill="x")
        ttk.Button(record_buttons, text="载入文本棋谱", command=self.load_record_from_text).pack(side="left")
        ttk.Button(record_buttons, text="打开文件…", command=self.open_record).pack(side="left", padx=4)
        ttk.Button(record_buttons, text="保存文件…", command=self.save_record).pack(side="left")

        setup_tab = ttk.Frame(notebook, padding=6)
        notebook.add(setup_tab, text="对局设置")
        ttk.Label(setup_tab, text="白方（下方）").grid(row=0, column=0, sticky="w")
        self.var_white_player = tk.StringVar(value=KEY_TO_LABEL[self.white_key])
        white_box = ttk.Combobox(setup_tab, textvariable=self.var_white_player,
                                 values=PLAYER_LABELS, state="readonly", width=14)
        white_box.grid(row=0, column=1, sticky="w", padx=4)
        white_box.bind("<<ComboboxSelected>>", lambda event: self.apply_player_settings())
        ttk.Label(setup_tab, text="黑方（上方）").grid(row=1, column=0, sticky="w")
        self.var_black_player = tk.StringVar(value=KEY_TO_LABEL[self.black_key])
        black_box = ttk.Combobox(setup_tab, textvariable=self.var_black_player,
                                 values=PLAYER_LABELS, state="readonly", width=14)
        black_box.grid(row=1, column=1, sticky="w", padx=4)
        black_box.bind("<<ComboboxSelected>>", lambda event: self.apply_player_settings())
        ttk.Label(setup_tab, text="外部程序（白方）").grid(
            row=2, column=0, columnspan=3, sticky="w", pady=(8, 0))
        self.var_external_white = tk.StringVar(value=default_external_command("ai-deepseek"))
        ttk.Entry(setup_tab, textvariable=self.var_external_white, font=self.font_mono,
                  width=24).grid(row=3, column=0, columnspan=2, sticky="ew", pady=2)
        ttk.Button(setup_tab, text="浏览…", command=lambda: self.browse_external("white")).grid(
            row=3, column=2, padx=2)
        ttk.Label(setup_tab, text="外部程序（黑方）").grid(row=4, column=0, columnspan=3,
                                                          sticky="w", pady=(4, 0))
        self.var_external_black = tk.StringVar(value=default_external_command("ai-gpt"))
        ttk.Entry(setup_tab, textvariable=self.var_external_black, font=self.font_mono,
                  width=24).grid(row=5, column=0, columnspan=2, sticky="ew", pady=2)
        ttk.Button(setup_tab, text="浏览…", command=lambda: self.browse_external("black")).grid(
            row=5, column=2, padx=2)
        preset_row = ttk.Frame(setup_tab)
        preset_row.grid(row=6, column=0, columnspan=3, sticky="w", pady=(6, 0))
        ttk.Label(preset_row, text="快速填充：").pack(side="left")
        for label, first, second in (("Claude vs DeepSeek", "ai-claude", "ai-deepseek"),
                                     ("Claude vs GPT", "ai-claude", "ai-gpt"),
                                     ("DeepSeek vs GPT", "ai-deepseek", "ai-gpt")):
            ttk.Button(preset_row, text=label, width=17,
                       command=lambda a=first, b=second: self.apply_preset(a, b)).pack(side="left", padx=2)
        ttk.Label(setup_tab, text="外部 AI 每步时限 (ms)").grid(row=7, column=0, sticky="w", pady=(6, 0))
        self.var_external_time = tk.StringVar(value="300")
        ttk.Entry(setup_tab, textvariable=self.var_external_time, width=8).grid(
            row=7, column=1, sticky="w", padx=4)
        ttk.Label(setup_tab, text="无吃子步数上限").grid(row=8, column=0, sticky="w", pady=(8, 0))
        self.var_limit = tk.StringVar(value=str(no_capture_limit_label(self.limit)))
        limit_box = ttk.Combobox(setup_tab, textvariable=self.var_limit, width=8, state="readonly",
                                 values=["50", "80", "100", "150", "200", "off"])
        limit_box.grid(row=8, column=1, sticky="w", padx=4, pady=(8, 0))
        limit_box.bind("<<ComboboxSelected>>", lambda event: self.apply_limit())
        ttk.Button(setup_tab, text="立即 AI 走子", command=lambda: self.request_ai_move(force=True)).grid(
            row=9, column=0, columnspan=2, sticky="w", pady=(10, 0))
        ttk.Label(setup_tab, text="AlphaZero 权重文件（.json，训练产物 runs/.../best.json）",
                  wraplength=420).grid(row=11, column=0, columnspan=3, sticky="w", pady=(8, 0))
        self.var_model = tk.StringVar(value=default_model_path())
        ttk.Entry(setup_tab, textvariable=self.var_model, font=self.font_mono,
                  width=40).grid(row=12, column=0, columnspan=2, sticky="ew", pady=2)
        ttk.Button(setup_tab, text="浏览…", command=self.browse_model).grid(row=12, column=2, padx=2)
        ttk.Label(setup_tab, text="MCTS 每步模拟次数").grid(row=13, column=0, sticky="w", pady=(6, 0))
        self.var_az_sims = tk.StringVar(value="120")
        ttk.Entry(setup_tab, textvariable=self.var_az_sims, width=8).grid(row=13, column=1,
                                                                         sticky="w", padx=4)
        ttk.Label(setup_tab, text="提示：把白/黑都设为「外部程序」再点“斗蛐蛐”，即可让两个 AI 对打。",
                  wraplength=330).grid(row=10, column=0, columnspan=3, sticky="w", pady=(8, 0))
        setup_tab.columnconfigure(0, weight=1)

        log_tab = ttk.Frame(notebook, padding=6)
        notebook.add(log_tab, text="日志")
        self.log_text = tk.Text(log_tab, height=6, width=42, font=self.font_mono, wrap="word",
                               state="disabled")
        self.log_text.pack(fill="both", expand=True)

        self.var_status = tk.StringVar(value="就绪")
        ttk.Label(self.root, textvariable=self.var_status, anchor="w", padding=(8, 2)).pack(fill="x")

    # -------------------------------------------------- 状态与绘制

    @property
    def ply(self) -> int:
        return self.game.ply

    def new_game(self, fen: Optional[str] = None, headers: Optional[Dict[str, str]] = None) -> None:
        text = fen or START_FEN
        try:
            game = Game.from_fen(text, no_capture_limit=self.limit)
        except FenError as exc:
            self.report("FEN 无法解析：%s" % exc, error=True)
            return
        self.start_fen = game.fen()
        self.game = game
        self.line = []
        self.headers = dict(headers) if headers else default_headers(
            white=KEY_TO_LABEL.get(self.white_key, "?"),
            black=KEY_TO_LABEL.get(self.black_key, "?"))
        self.declared = None
        self.selected = None
        self.targets = {}
        self.result_shown_for = None
        self.update_all()
        self.log("新对局开始：%s" % ("标准初始局面" if self.start_fen == START_FEN else "自定义局面"))
        self.sync_engine("new")

    def update_all(self) -> None:
        self.draw_board()
        self.update_status()
        self.update_move_list()
        self.update_fen_box()
        self.check_game_over()

    def update_status(self) -> None:
        white_count, white_size = self.game.piece_stats(WHITE)
        black_count, black_size = self.game.piece_stats(BLACK)
        self.var_turn.set("轮到：%s" % color_name(self.game.side))
        limit_text = "不限" if self.game.no_capture_limit is None else str(self.game.no_capture_limit)
        self.var_nocap.set("无吃子计数：%d / %s     回合：%d" %
                           (self.game.no_capture, limit_text, self.game.move_number))
        self.var_white.set("白方：%d 子，大小和 %d" % (white_count, white_size))
        self.var_black.set("黑方：%d 子，大小和 %d" % (black_count, black_size))
        if self.game.result is not None:
            self.var_result.set(self.game.result.text())
        elif self.declared:
            self.var_result.set("已宣布结果：%s" % self.declared)
        elif self.game.ply < len(self.line):
            self.var_result.set("正在浏览历史（第 %d / %d 步），按 End 回到最新"
                                % (self.game.ply, len(self.line)))
        else:
            self.var_result.set("对局进行中（第 %d 步）" % self.game.ply)

    def update_move_list(self) -> None:
        self.move_list.delete(0, tk.END)
        for index, token in enumerate(self.line):
            ply = index + 1
            prefix = "%d." % ((ply + 1) // 2) if ply % 2 == 1 else "%d..." % (ply // 2)
            self.move_list.insert(tk.END, "%s %s" % (prefix, token))
        if 0 < self.game.ply <= len(self.line):
            self.move_list.selection_clear(0, tk.END)
            self.move_list.selection_set(self.game.ply - 1)
            self.move_list.see(self.game.ply - 1)

    def update_fen_box(self) -> None:
        self.fen_text.delete("1.0", tk.END)
        self.fen_text.insert("1.0", self.game.fen())

    def to_pixel(self, sq: int) -> Tuple[int, int]:
        x = sq % 11
        y = sq // 11
        if self.flip:
            return MARGIN + (10 - x) * CELL, MARGIN + y * CELL
        return MARGIN + x * CELL, MARGIN + (10 - y) * CELL

    def to_square(self, px: int, py: int) -> Optional[int]:
        for sq in range(121):
            cx, cy = self.to_pixel(sq)
            if abs(px - cx) <= CELL * 0.48 and abs(py - cy) <= CELL * 0.48:
                return sq
        return None

    def draw_board(self) -> None:
        canvas = self.canvas
        canvas.delete("all")
        for index in range(11):
            offset = MARGIN + index * CELL
            canvas.create_line(MARGIN, offset, MARGIN + 10 * CELL, offset, fill=COLOR_LINE)
            canvas.create_line(offset, MARGIN, offset, MARGIN + 10 * CELL, fill=COLOR_LINE)
        for sq in range(121):
            cx, cy = self.to_pixel(sq)
            canvas.create_oval(cx - 2, cy - 2, cx + 2, cy + 2, fill=COLOR_POINT, outline="")
        # 坐标
        for index in range(11):
            cx = MARGIN + index * CELL
            cy = MARGIN + index * CELL
            file_index = 10 - index if self.flip else index
            rank_label = str(index + 1) if self.flip else str(11 - index)
            canvas.create_text(cx, BOARD_PX - MARGIN + 22, text=FILES[file_index],
                               fill="#5a3d20", font=self.font_small)
            canvas.create_text(MARGIN - 24, cy, text=rank_label, fill="#5a3d20", font=self.font_small)
        # 上一步
        last = self.game.last_move
        if last is not None:
            for sq in (last.move.frm, last.move.to):
                cx, cy = self.to_pixel(sq)
                canvas.create_oval(cx - CELL * 0.42, cy - CELL * 0.42, cx + CELL * 0.42, cy + CELL * 0.42,
                                   outline=COLOR_LAST, width=3)
        # 可落点
        for sq, is_capture in self.targets.items():
            cx, cy = self.to_pixel(sq)
            if is_capture:
                canvas.create_oval(cx - 18, cy - 18, cx + 18, cy + 18, outline=COLOR_CAPTURE, width=3)
            else:
                canvas.create_oval(cx - 7, cy - 7, cx + 7, cy + 7, fill=COLOR_TARGET, outline="")
        # 棋子
        for sq, color, size in self.game.pieces():
            self.draw_piece(sq, color, size)
        if self.selected is not None:
            cx, cy = self.to_pixel(self.selected)
            canvas.create_oval(cx - 22, cy - 22, cx + 22, cy + 22, outline=COLOR_SELECT, width=3)

    def draw_piece(self, sq: int, color: int, size: int) -> None:
        cx, cy = self.to_pixel(sq)
        radius = 20 if len(str(size)) < 3 else 20
        fill = COLOR_WHITE_PIECE if color == WHITE else COLOR_BLACK_PIECE
        outline = "#2b2b2b" if color == WHITE else "#0d0f11"
        text_color = "#1b1b1b" if color == WHITE else "#f2f2f2"
        self.canvas.create_oval(cx - radius, cy - radius, cx + radius, cy + radius,
                                fill=fill, outline=outline, width=2)
        digits = len(str(size))
        font = ("Consolas", 15 if digits <= 2 else (12 if digits == 3 else 10), "bold")
        self.canvas.create_text(cx, cy, text=str(size), fill=text_color, font=font)

    # -------------------------------------------------- 交互

    def on_click(self, event) -> None:
        if self.game.is_over or self.thinking:
            return
        sq = self.to_square(event.x, event.y)
        if sq is None:
            return
        if self.is_human_turn():
            if self.game.cells[sq] != 0 and (self.game.cells[sq] > 0) == (self.game.side > 0):
                self.select_square(sq)
                return
            if self.selected is not None and sq in self.targets:
                self.play_move(self.selected, sq)
                return
        self.selected = None
        self.targets = {}
        self.draw_board()

    def select_square(self, sq: int) -> None:
        self.selected = sq
        self.targets = {}
        for move in self.game.moves_from(sq):
            self.targets[move.to] = self.game.cells[move.to] != 0
        self.draw_board()

    def is_human_turn(self) -> bool:
        key = self.white_key if self.game.side == WHITE else self.black_key
        return key == "human"

    def play_move(self, frm: int, to: int) -> None:
        token = "%s-%s" % (square_name(frm), square_name(to))
        if self.game.cells[to] != 0:
            token = "%sx%s" % (square_name(frm), square_name(to))
        self.play_token(token)

    def play_token(self, token: str) -> bool:
        if self.game.is_over:
            self.report("对局已经结束。")
            return False
        try:
            info = self.game.push(token)
        except (IllegalMove, ValueError) as exc:
            self.report("无法走这一步：%s" % exc, error=True)
            return False
        except GameOver as exc:
            self.report(str(exc))
            return False
        del self.line[self.game.ply - 1:]
        self.line.append(info.notation)
        self.declared = None
        self.selected = None
        self.targets = {}
        self.log("第 %d 步 %s" % (info.ply, info.describe()))
        self.update_all()
        self.sync_engine("move", token=info.notation)
        return True

    def play_entry_move(self) -> None:
        text = self.var_move_entry.get().strip()
        if not text:
            return
        try:
            frm, to, _ = parse_move_text(text)
        except ValueError as exc:
            self.report(str(exc), error=True)
            return
        if self.play_token("%s-%s" % (square_name(frm), square_name(to))):
            self.var_move_entry.set("")

    def on_move_selected(self, event) -> None:
        selection = self.move_list.curselection()
        if not selection:
            return
        self.goto(int(selection[0]) + 1)

    def goto(self, ply: int) -> None:
        target = max(0, min(int(ply), len(self.line)))
        while self.game.ply > target:
            self.game.pop()
        while self.game.ply < target:
            token = self.line[self.game.ply]
            try:
                self.game.push(token)
            except (IllegalMove, GameOver, ValueError) as exc:
                self.report("回放失败：%s" % exc, error=True)
                break
        self.selected = None
        self.targets = {}
        self.update_all()

    def goto_end(self) -> None:
        self.goto(len(self.line))

    def step(self, delta: int) -> None:
        self.goto(self.game.ply + delta)

    def undo(self) -> None:
        if self.game.ply <= 0:
            self.report("已经是开局局面。")
            return
        self.game.pop()
        del self.line[self.game.ply:]
        self.declared = None
        self.selected = None
        self.targets = {}
        self.update_all()
        self.log("悔棋一步，当前第 %d 步" % self.game.ply)
        self.sync_engine("undo")

    def flip_board(self) -> None:
        self.flip = not self.flip
        self.draw_board()

    def resign(self) -> None:
        if self.game.is_over:
            self.report("对局已经结束。")
            return
        loser = self.game.side
        winner = BLACK if loser == WHITE else WHITE
        self.game.declare_result(winner, "resign", "%s认输" % color_name(loser))
        self.declared = self.game.result.score
        self.update_all()
        self.log("%s认输，%s胜" % (color_name(loser), color_name(winner)))

    def agree_draw(self) -> None:
        if self.game.is_over:
            self.report("对局已经结束。")
            return
        self.game.declare_result(None, "agreement", "双方同意和棋")
        self.declared = self.game.result.score
        self.update_all()
        self.log("双方同意和棋")

    def check_game_over(self) -> None:
        result = self.game.result
        if result is None or self.result_shown_for == self.game.ply:
            return
        self.result_shown_for = self.game.ply
        self.log("对局结束：" + result.text())
        self.var_status.set("对局结束：" + result.text())
        if self.duel_total:
            self._record_duel_result(result)
            return
        if not self.silent:
            messagebox.showinfo("对局结束", result.text())

    # -------------------------------------------------- 斗蛐蛐（多局对战）

    def start_duel(self, total: Optional[int] = None) -> None:
        """让白/黑两路外部程序轮流执先，下若干局并统计比分。"""
        if self.duel_total:
            self.report("斗蛐蛐正在进行：已下 %d / %d 局。" % (self.duel_played, self.duel_total))
            return
        white_cmd = self.var_external_white.get().strip()
        black_cmd = self.var_external_black.get().strip()
        if not white_cmd or not black_cmd:
            self.report("请先在“对局设置”里填写白方和黑方的外部程序命令行。", error=True)
            return
        if total is None:
            dialog = _TextDialog(self.root, "斗蛐蛐", "双方轮流执先，共下几局？（建议 4~10）", "6")
            text = (dialog.result or "").strip()
            if not text:
                return
            try:
                total = int(text)
            except ValueError:
                self.report("局数必须是整数。", error=True)
                return
        if total < 1:
            self.report("局数至少为 1。", error=True)
            return
        total = min(total, 100)
        self.white_key = "external"
        self.black_key = "external"
        self.var_white_player.set(KEY_TO_LABEL["external"])
        self.var_black_player.set(KEY_TO_LABEL["external"])
        self.duel_total = total
        self.duel_played = 0
        self.duel_scores = {}
        self.duel_labels = {WHITE: command_label(white_cmd), BLACK: command_label(black_cmd)}
        self.log("斗蛐蛐开始：白=%s，黑=%s，共 %d 局（每局交换先后），每步 %d ms"
                 % (self.duel_labels[WHITE], self.duel_labels[BLACK], total,
                    self.external_time_ms()))
        self.new_game()
        self.request_ai_move(force=True)

    def current_duel_labels(self) -> Tuple[str, str]:
        return (self.duel_labels.get(WHITE, "白方"), self.duel_labels.get(BLACK, "黑方"))

    def abort_duel(self, reason: str) -> None:
        if not self.duel_total:
            return
        self.log("斗蛐蛐中止：%s（已下 %d / %d 局）"
                 % (reason, self.duel_played, self.duel_total))
        self.var_status.set("斗蛐蛐中止：" + reason)
        self.duel_total = 0
        self.duel_played = 0
        self.duel_scores = {}
        self.duel_labels = {}

    def _record_duel_result(self, result) -> None:
        white_label, black_label = self.current_duel_labels()
        score = result.score
        if score == "1-0":
            white_point, black_point, text = 1.0, 0.0, "白胜"
        elif score == "0-1":
            white_point, black_point, text = 0.0, 1.0, "黑胜"
        else:
            white_point, black_point, text = 0.5, 0.5, "和棋"
        self.duel_scores[white_label] = self.duel_scores.get(white_label, 0.0) + white_point
        self.duel_scores[black_label] = self.duel_scores.get(black_label, 0.0) + black_point
        self.duel_played += 1
        self.log("斗蛐蛐第 %d/%d 局：%s %s %s（%s）"
                 % (self.duel_played, self.duel_total, white_label, text, black_label,
                    result.text()))
        self.var_status.set("斗蛐蛐：第 %d/%d 局结束（%s vs %s）"
                            % (self.duel_played, self.duel_total, white_label, black_label))
        self.root.after(1200, self._next_duel_game)

    def _next_duel_game(self) -> None:
        if not self.duel_total:
            return
        if self.duel_played >= self.duel_total:
            parts = ["%s %.1f" % (name, value)
                     for name, value in sorted(self.duel_scores.items(), key=lambda kv: -kv[1])]
            summary = " / ".join(parts)
            self.log("斗蛐蛐结束：%d 局，比分 %s" % (self.duel_total, summary))
            self.var_status.set("斗蛐蛐结束：" + summary)
            self.duel_total = 0
            self.duel_played = 0
            self.duel_scores = {}
            self.duel_labels = {}
            return
        white_cmd = self.var_external_white.get()
        black_cmd = self.var_external_black.get()
        self.var_external_white.set(black_cmd)
        self.var_external_black.set(white_cmd)
        self.duel_labels[WHITE], self.duel_labels[BLACK] = (
            self.duel_labels[BLACK], self.duel_labels[WHITE])
        self.log("交换先后手：白=%s，黑=%s" % self.current_duel_labels())
        self.new_game()
        self.request_ai_move(force=True)

    # -------------------------------------------------- AI 与自动播放

    def set_speed(self, value: int) -> None:
        self.delay_ms = max(30, int(value))

    def toggle_autoplay(self) -> None:
        self.auto_play = bool(self.var_autoplay.get())
        self.log("自动播放：" + ("开" if self.auto_play else "关"))

    def self_play(self) -> None:
        if self.white_key == "human":
            self.white_key = "greedy"
            self.var_white_player.set(KEY_TO_LABEL["greedy"])
        if self.black_key == "human":
            self.black_key = "greedy"
            self.var_black_player.set(KEY_TO_LABEL["greedy"])
        self.log("双方 AI 自对弈（白方 %s，黑方 %s）" %
                 (KEY_TO_LABEL[self.white_key], KEY_TO_LABEL[self.black_key]))
        self.request_ai_move(force=True)

    def apply_player_settings(self) -> None:
        self.white_key = LABEL_TO_KEY.get(self.var_white_player.get(), "human")
        self.black_key = LABEL_TO_KEY.get(self.var_black_player.get(), "human")
        self.headers["White"] = KEY_TO_LABEL.get(self.white_key, "?")
        self.headers["Black"] = KEY_TO_LABEL.get(self.black_key, "?")
        self.log("对局设置：白方 %s，黑方 %s" %
                 (KEY_TO_LABEL[self.white_key], KEY_TO_LABEL[self.black_key]))
        self.request_ai_move(force=True)

    def apply_limit(self) -> None:
        text = self.var_limit.get().strip().lower()
        self.limit = None if text in ("off", "none", "不限") else int(text)
        self.new_game(self.start_fen)
        self.log("无吃子步数上限设为：%s" % ("不限" if self.limit is None else self.limit))

    def browse_model(self) -> None:
        path = filedialog.askopenfilename(title="选择 AlphaZero 权重文件",
                                          filetypes=[("AlphaZero 网络", "*.json"),
                                                     ("所有文件", "*.*")])
        if path:
            self.var_model.set(path)

    def browse_external(self, side: str = "white") -> None:
        path = filedialog.askopenfilename(
            title="选择外部 AI 程序（%s）" % ("白方" if side == "white" else "黑方"),
            filetypes=[("可执行/脚本", "*.exe;*.bat;*.cmd;*.py"), ("所有文件", "*.*")])
        if not path:
            return
        if path.lower().endswith(".py"):
            command = '%s -B "%s"' % (_python_executable(), path)
        else:
            command = '"%s"' % path
        target = self.var_external_white if side == "white" else self.var_external_black
        target.set(command)

    def apply_preset(self, first: str, second: str) -> None:
        """一键把白/黑外部程序填成指定的两个内置 AI。"""
        self.white_key = "external"
        self.black_key = "external"
        self.var_white_player.set(KEY_TO_LABEL["external"])
        self.var_black_player.set(KEY_TO_LABEL["external"])
        self.var_external_white.set(default_external_command(first))
        self.var_external_black.set(default_external_command(second))
        self.log("外部程序预设：白=%s，黑=%s"
                 % (command_label(self.var_external_white.get()),
                    command_label(self.var_external_black.get())))
        self.request_ai_move(force=True)

    def external_command_for(self, side: int) -> str:
        var = self.var_external_white if side == WHITE else self.var_external_black
        return var.get().strip()

    def external_time_ms(self) -> int:
        try:
            return max(50, min(60000, int(self.var_external_time.get())))
        except ValueError:
            return 300

    def current_player_key(self) -> str:
        return self.white_key if self.game.side == WHITE else self.black_key

    def request_ai_move(self, force: bool = False) -> None:
        if self.thinking or self.game.is_over:
            return
        if self.game.ply != len(self.line) and not force:
            return
        key = self.current_player_key()
        if key == "human":
            return
        self.thinking = True
        self.var_status.set("AI 正在思考…（%s）" % KEY_TO_LABEL.get(key, key))
        fen = self.game.fen()
        level = key
        # 所有 tkinter 读取都在主线程完成，工作线程只拿纯数据
        command = self.external_command_for(self.game.side)
        think_ms = self.external_time_ms()
        model_path = self.var_model.get().strip()
        try:
            az_sims = max(1, int(self.var_az_sims.get()))
        except ValueError:
            az_sims = 120

        def worker() -> None:
            try:
                if level == "external":
                    result = self.ask_external_ai(command, fen, think_ms)
                elif level in ("azero", "hybrid"):
                    result = self.ask_alphazero(model_path, az_sims, fen, level)
                else:
                    game = Game.from_fen(fen, no_capture_limit=self.limit)
                    engine = make_ai(level, depth=4 if level == "capture" else 3 if level == "ab" else None)
                    move = engine.choose(game)
                    result = game.move_dict(move)["notation"] if move else ""
                self.ai_queue.put(("move", result))
            except Exception as exc:  # noqa: BLE001
                self.ai_queue.put(("error", "%s: %s" % (type(exc).__name__, exc)))

        threading.Thread(target=worker, daemon=True).start()

    def ask_alphazero(self, model_path: str, sims: int, fen: str, mode: str = "azero") -> str:
        """用训练好的 AlphaZero 网络走一步。

        注意：本方法在工作线程里执行，所以参数都由主线程读好后传进来，
        这里绝不能访问任何 tkinter 控件（tkinter 不是线程安全的）。
        """
        if not model_path:
            raise RuntimeError("还没有选择 AlphaZero 权重文件（可在“对局设置”里指定 best.json）")
        cache_key = (model_path, sims, mode, Path(model_path).stat().st_mtime_ns)
        player = self.az_players.get(cache_key)
        if player is None:
            try:
                from alphazero.players import load_player
            except ImportError as exc:
                raise RuntimeError("加载 AlphaZero 需要 numpy：%s" % exc)
            player = load_player(model_path, sims=sims, seed=1,
                                 search_kind="hybrid" if mode == "hybrid" else "mcts",
                                 ab_depth=2, time_limit=0.3)
            self.az_players[cache_key] = player
        game = Game.from_fen(fen, no_capture_limit=self.limit)
        move = player.choose(game)
        if move is None:
            return ""
        return game.move_dict(move)["notation"]

    def ask_external_ai(self, command: str, fen: str, time_ms: int = 300) -> str:
        if not command:
            raise RuntimeError("还没有填写外部程序命令行（在“对局设置”里填写）")
        process = self.external_players.get(command)
        if process is None or not process.alive:
            process = LineProcess(command, expect_greeting=False, name="外部 AI")
            self.external_players[command] = process
        timeout = max(30.0, float(time_ms) / 1000.0 * 10.0)
        response = process.request({"request": "move", "fen": fen, "level": "greedy",
                                    "time_ms": int(time_ms)}, timeout=timeout)
        if not response.get("ok"):
            raise RuntimeError(response.get("error", "外部程序返回了错误"))
        return response.get("notation") or response.get("move") or ""

    def _tick(self) -> None:
        try:
            while True:
                try:
                    kind, payload = self.ai_queue.get_nowait()
                except queue.Empty:
                    break
                self.thinking = False
                if kind == "move":
                    if payload and self.play_token(payload):
                        self.ai_failures = 0
                    else:
                        self.log("AI 没有返回可用着法：%r" % (payload,))
                        self.handle_ai_failure(payload or "空着法")
                else:
                    self.log("AI 出错：" + payload)
                    self.handle_ai_failure(payload)
            self._drive()
        except Exception:  # noqa: BLE001
            self.log(traceback.format_exc())
        finally:
            self.root.after(120, self._tick)

    def handle_ai_failure(self, reason: str) -> None:
        """AI 连续失败时交出控制权，避免界面陷入死循环。"""
        self.ai_failures += 1
        if self.ai_failures < 3:
            return
        self.ai_failures = 0
        side = self.game.side
        if side == WHITE:
            self.white_key = "human"
            self.var_white_player.set(KEY_TO_LABEL["human"])
        else:
            self.black_key = "human"
            self.var_black_player.set(KEY_TO_LABEL["human"])
        self.report("AI 连续 3 次未能走出合法着法（%s），已改由人类接管。" % reason, error=True)
        self.abort_duel("外部 AI 连续失败：%s" % reason)

    def _drive(self) -> None:
        if self.game.is_over:
            return
        if self.auto_play:
            if self.game.ply < len(self.line):
                self.goto(self.game.ply + 1)
            else:
                self.var_autoplay.set(False)
                self.auto_play = False
                self.log("棋谱播放完毕")
            return
        if self.thinking:
            return
        if self.game.ply < len(self.line):
            return  # 正在浏览历史
        if self.current_player_key() == "human":
            return
        self.request_ai_move()

    # -------------------------------------------------- FEN 与棋谱

    def focus_fen_entry(self) -> None:
        self.fen_text.focus_set()

    def copy_fen(self) -> None:
        text = self.game.fen()
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        self.var_status.set("FEN 已复制到剪贴板")
        self.log("复制 FEN：" + text)

    def load_fen_from_text(self) -> None:
        text = self.fen_text.get("1.0", tk.END).strip()
        if not text:
            self.report("请先粘贴 FEN 文本。", error=True)
            return
        self.new_game(text)
        self.var_status.set("已按 FEN 载入局面")

    def load_record_from_text(self) -> None:
        text = self.record_text.get("1.0", tk.END).strip()
        if not text:
            self.report("请先粘贴棋谱文本。", error=True)
            return
        self.load_record_text(text)

    def load_record_text(self, text: str) -> None:
        try:
            report = replay(text, no_capture_limit=self.limit)
        except RecordError as exc:
            self.report("棋谱格式错误：%s" % exc, error=True)
            return
        if not report.ok:
            self.report("棋谱回放出错：%s" % "；".join(report.errors), error=True)
            return
        self.start_fen = report.start_fen
        self.headers = dict(report.headers) or default_headers()
        self.line = [info.notation for info in report.moves]
        self.declared = None if report.declared_result in (None, "*") else report.declared_result
        self.game = Game.from_fen(self.start_fen, no_capture_limit=self.limit)
        self.result_shown_for = None
        self.goto(len(self.line))
        termination = self.headers.get("Termination")
        if self.declared and termination in ("resign", "agreement"):
            from sqrtchess import result_winner
            self.game.declare_result(result_winner(self.declared), termination, "棋谱声明的结果")
        self.white_key = "human"
        self.black_key = "human"
        self.var_white_player.set(KEY_TO_LABEL["human"])
        self.var_black_player.set(KEY_TO_LABEL["human"])
        self.update_all()
        self.record_text.delete("1.0", tk.END)
        self.record_text.insert("1.0", format_record(Record(headers=self.headers, moves=self.line,
                                                            result=self.declared or "*",
                                                            start_fen=self.start_fen)))
        self.log("已载入棋谱：%d 步，起始局面 %s" %
                 (len(self.line), "标准初始局面" if self.start_fen == START_FEN else "自定义"))
        self.var_status.set("棋谱已载入（可用 ◀ ▶ 或自动播放浏览）")

    def open_record(self) -> None:
        path = filedialog.askopenfilename(
            title="打开根号棋棋谱",
            filetypes=[("根号棋棋谱", "*.sqg"), ("文本文件", "*.txt"), ("所有文件", "*.*")])
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8-sig") as handle:
                text = handle.read()
        except OSError as exc:
            self.report("无法读取文件：%s" % exc, error=True)
            return
        self.load_record_text(text)
        self.log("已打开棋谱文件：%s" % path)

    def save_record(self) -> None:
        path = filedialog.asksaveasfilename(
            title="保存棋谱", defaultextension=".sqg",
            initialfile="根号棋对局.sqg",
            filetypes=[("根号棋棋谱", "*.sqg"), ("文本文件", "*.txt")])
        if not path:
            return
        headers = dict(self.headers)
        result = self.declared or (self.game.result.score if self.game.result else "*")
        if self.game.result is not None:
            headers["Termination"] = self.game.result.reason
            if self.game.result.detail:
                headers["ResultDetail"] = self.game.result.detail
        record = Record(headers=headers, moves=list(self.line), result=result,
                        start_fen=self.start_fen)
        try:
            with open(path, "w", encoding="utf-8", newline="\n") as handle:
                handle.write(format_record(record))
        except OSError as exc:
            self.report("无法写入文件：%s" % exc, error=True)
            return
        self.log("棋谱已保存：%s" % path)
        self.var_status.set("棋谱已保存：" + path)

    # -------------------------------------------------- 外部引擎连接

    def connect_engine(self, command: Optional[str] = None) -> None:
        if command is None:
            default = "%s -m sqrtchess proto" % _python_executable()
            dialog = _TextDialog(self.root, "连接外部引擎",
                                 "外部引擎命令行（需实现 JSON 行控制协议）：", default)
            command = dialog.result
        if not command:
            return
        try:
            process = LineProcess(command, expect_greeting=True, name="外部引擎")
        except Exception as exc:  # noqa: BLE001
            self.report("连接失败：%s" % exc, error=True)
            return
        if self.engine_conn is not None:
            self.engine_conn.close()
        self.engine_conn = process
        greeting = process.greeting or {}
        self.log("已连接外部引擎：%s %s（协议 %s）" %
                 (greeting.get("hello") or greeting.get("name", "?"),
                  greeting.get("version", ""), greeting.get("protocol", "?")))
        self.var_status.set("已连接外部引擎：" + str(greeting.get("hello") or "?"))
        self.sync_engine("new")

    def show_engine_info(self) -> None:
        if self.engine_conn is None or not self.engine_conn.alive:
            self.report("尚未连接外部引擎。可用“引擎 → 连接外部引擎进程…”连接。")
            return
        try:
            info = self.engine_conn.request({"cmd": "info"})
        except Exception as exc:  # noqa: BLE001
            self.report("引擎无响应：%s" % exc, error=True)
            return
        self.report("引擎：%s %s\n规则：%s\n协议版本：%s\n支持命令：%s" %
                    (info.get("name"), info.get("version"), info.get("rule"),
                     info.get("protocol"), ", ".join(info.get("commands", []))))

    def disconnect_engine(self) -> None:
        if self.engine_conn is not None:
            self.engine_conn.close()
            self.engine_conn = None
            self.log("已断开外部引擎连接")

    def sync_engine(self, action: str, token: Optional[str] = None) -> None:
        """把本地局面同步给已连接的外部引擎，并核对 FEN 是否一致。"""
        process = self.engine_conn
        if process is None or not process.alive:
            return
        try:
            if action == "new":
                process.request({"cmd": "new", "fen": self.start_fen})
            elif action == "move" and token:
                process.request({"cmd": "move", "move": token})
            elif action == "undo":
                process.request({"cmd": "restart"})
                for item in self.line[:self.game.ply]:
                    process.request({"cmd": "move", "move": item})
            elif action == "fen":
                process.request({"cmd": "fen", "fen": self.game.fen()})
            response = process.request({"cmd": "fen"})
            remote_fen = response.get("fen")
            if remote_fen and remote_fen != self.game.fen():
                self.log("警告：外部引擎局面与本地不一致\n  本地 %s\n  引擎 %s"
                         % (self.game.fen(), remote_fen))
        except Exception as exc:  # noqa: BLE001
            self.log("外部引擎通信失败：%s" % exc)

    # -------------------------------------------------- 帮助

    def show_rules(self) -> None:
        self.report(
            "根号棋规则\n"
            "1. 11x11 格点棋盘，双方镜像布局：靠边一行 4 8 5 2 9 2 5 8 4，靠中心一行 9 枚 1。\n"
            "2. 大小为 n 的棋子每步必须恰好移动 sqrt(n) 格，即位移 (dx,dy) 满足 dx*dx+dy*dy=n；\n"
            "   无视路径，只看落点：空点可走、己方棋子不可走、敌方棋子则吃子。\n"
            "3. 吃子后自身大小 += 被吃棋子的大小（例如 5 吃 3 变成 8）。\n"
            "4. 一方棋子被全部吃光即判负。\n"
            "5. 任意一步后双方都还有棋子，但有一方所有棋子都没有合法落点 → 平局判定。\n"
            "6. 双方合计连续 100 步（半回合）无吃子 → 平局判定。\n"
            "7. 平局判定：比较双方棋子数与大小总和，一方两项都严格更大则该方胜，否则平局。",
            title="根号棋规则")

    def show_notation_help(self) -> None:
        self.report(
            "记谱：着法写成 大小+起点+分隔+终点，例如 5b2xc3 表示 b2 上大小 5 的棋子吃 c3。\n"
            "x 表示吃子，- 表示走到空点；也可以省略大小写成 b2-c3。\n\n"
            "FEN（局面文本）：棋盘 行棋方 无吃子步数 回合数\n"
            "棋盘 11 行用 / 分隔（第 11 行在前），每行 11 个记号：. 空点，W<大小> 白子，B<大小> 黑子。\n"
            "例如初始局面第 1 行为 .W4W8W5W2W9W2W5W8W4.\n\n"
            "棋谱文件（.sqg）头部记录 Event/White/Black/Result 等，非标准起始局面会写入 FEN 头。",
            title="记谱与 FEN")

    def show_protocol_help(self) -> None:
        self.report(
            "引擎提供两种机器接口（详见 docs/PROTOCOL.md）：\n\n"
            "1) 控制协议：python -m sqrtchess proto\n"
            "   每行一个 JSON，如 {\"cmd\":\"moves\"}、{\"cmd\":\"move\",\"move\":\"1b2-b3\"}。\n\n"
            "2) AI 应答协议：python -m sqrtchess ai\n"
            "   每行一个 JSON，如 {\"request\":\"move\",\"fen\":\"...\"}，返回 {\"move\":\"5b2xc3\"}。\n\n"
            "3) TCP 服务：python -m sqrtchess serve --port 8765\n"
            "本界面的“普通 AI”和“外部程序”都会用到这些接口。",
            title="引擎与协议")

    def show_about(self) -> None:
        self.report("根号棋 Square Root Chess\n图形界面基于 Python 标准库 tkinter。\n"
                    "引擎源码位于 sqrtchess 包，棋谱为 .sqg 文本。", title="关于")

    # -------------------------------------------------- 输出

    def log(self, message: str) -> None:
        self.log_text.configure(state="normal")
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.log_text.configure(state="disabled")
        self.var_status.set(message.splitlines()[0][:120])

    def report(self, message: str, error: bool = False, title: str = "根号棋") -> None:
        self.log(("错误：" if error else "") + message)
        if self.silent:
            return
        if error:
            messagebox.showerror(title, message)
        else:
            messagebox.showinfo(title, message)

    def on_close(self) -> None:
        for process in list(self.external_players.values()):
            process.close()
        if self.engine_conn is not None:
            self.engine_conn.close()
        self.root.destroy()


def default_model_path() -> str:
    """如果本地已经有训练产物，就默认填上最新的 best.json。"""
    preferred = ROOT_DIR / "runs" / "az6" / "best.json"
    if preferred.exists():
        return str(preferred)
    runs = ROOT_DIR / "runs"
    if runs.is_dir():
        candidates = sorted(runs.glob("*/best.json"), key=lambda item: item.stat().st_mtime,
                            reverse=True)
        if candidates:
            return str(candidates[0])
    return ""


def default_external_command(folder: str) -> str:
    """优先使用随包携带的对手（ai-deepseek / ai-gpt / ai-claude），否则退回内置 AI 协议。"""
    for name in ("my_work.py", "strong_ai.py", "main.py"):
        script = ROOT_DIR / folder / name
        if script.exists():
            return '%s -B "%s"' % (_python_executable(), script)
    return "python -m sqrtchess ai --level greedy"


def command_label(command: str) -> str:
    """从命令行里猜一个短名字，用于比分显示。"""
    low = command.lower()
    if "deepseek" in low:
        return "DeepSeek"
    if "gpt" in low or "rootknight" in low:
        return "GPT"
    if "claude" in low or "strong_ai" in low:
        return "Claude"
    for token in command.replace('"', " ").split():
        if token.lower().endswith(".py"):
            name = os.path.basename(token)[:-3]
            if name:
                return name
    return "外部AI"


def no_capture_limit_label(limit) -> str:
    return "off" if limit is None else str(limit)


def _python_executable() -> str:
    exe = sys.executable or "python"
    return '"%s"' % exe if " " in exe else exe


class _TextDialog:
    """简单的单行输入对话框。"""

    def __init__(self, parent: tk.Tk, title: str, prompt: str, default: str = ""):
        self.result: Optional[str] = None
        window = tk.Toplevel(parent)
        window.title(title)
        window.transient(parent)
        window.grab_set()
        ttk.Label(window, text=prompt, wraplength=520, padding=8).pack(anchor="w")
        var = tk.StringVar(value=default)
        entry = ttk.Entry(window, textvariable=var, width=70, font=("Consolas", 9))
        entry.pack(fill="x", padx=8)
        entry.focus_set()
        entry.icursor(tk.END)

        def confirm() -> None:
            self.result = var.get().strip()
            window.destroy()

        buttons = ttk.Frame(window, padding=8)
        buttons.pack(fill="x")
        ttk.Button(buttons, text="确定", command=confirm).pack(side="right")
        ttk.Button(buttons, text="取消", command=window.destroy).pack(side="right", padx=6)
        entry.bind("<Return>", lambda event: confirm())
        window.wait_window()


# --------------------------------------------------------------------------- 入口


def run_selftest(app: "SqrtChessApp") -> int:
    """自动跑一遍界面流程，用于回归测试。"""
    problems: List[str] = []

    def check(name: str, ok: bool, extra: str = "") -> None:
        print("[%s] %s%s" % ("OK  " if ok else "FAIL", name, ("  " + extra) if extra else ""))
        if not ok:
            problems.append(name)

    app.new_game()
    check("初始局面 22 步", len(app.game.legal_moves()) == 22)
    check("初始 FEN 正确", app.game.fen() == START_FEN)
    app.play_token("1b2-b3")
    check("走子后着法行长度 1", app.line == ["1b2-b3"])
    check("画布棋子数量正确", len(app.game.pieces()) == 36)
    app.undo()
    check("悔棋回到开局", app.game.ply == 0 and app.line == [])

    app.white_key = "greedy"
    app.black_key = "greedy"
    app.play_token("1b2-b3")
    app.request_ai_move(force=True)
    import time
    for _ in range(300):
        app.root.update()
        if app.game.ply >= 2:
            break
        time.sleep(0.02)
    app.white_key = "human"
    app.black_key = "human"
    check("AI 通过线程走出一步", app.game.ply >= 2, "ply=%d" % app.game.ply)
    custom_fen = (".........../.........../.........../.........../.........../.........../"
                  "....W9....../.........../.........../.........../B1.......... w 0 1")
    app.new_game(custom_fen)
    check("载入自定义 FEN", app.game.ply == 0 and app.game.piece_stats(WHITE)[0] == 1)
    check("自定义 FEN 往返", app.game.fen() == custom_fen)
    app.play_token("e5-e8")
    check("自定义局面走子", app.game.piece_at("e8") == 9)

    app.white_key = "human"
    app.black_key = "human"
    recorder_line = ["1b2-b3", "1i10-i9", "1c2-c3", "1h10-h9"]
    text = ("[Event \"selftest\"]\n[White \"A\"]\n[Black \"B\"]\n[Result \"*\"]\n\n"
            "1. 1b2-b3 1i10-i9\n2. 1c2-c3 1h10-h9\n*\n")
    app.load_record_text(text)
    check("载入棋谱", app.line == recorder_line, str(app.line))
    app.goto(2)
    check("棋谱跳转到第 2 步", app.game.ply == 2)
    app.goto_end()
    check("跳到末尾", app.game.ply == 4)
    app.var_autoplay.set(False)

    app.new_game()
    app.resign()
    check("认输结束对局", app.game.result is not None and app.game.result.score == "0-1")

    # 吃子基线 AI（引擎内置的「尽可能多吃子」对手）
    app.new_game()
    app.white_key = "capture"
    app.black_key = "human"
    app.request_ai_move(force=True)
    for _ in range(300):
        app.root.update()
        if app.game.ply >= 1:
            break
        time.sleep(0.02)
    check("吃子基线 AI 走子", app.game.ply >= 1, "ply=%d" % app.game.ply)

    # AlphaZero 网络（如果本地已有训练产物）
    model = default_model_path()
    if model:
        app.var_model.set(model)
        app.var_az_sims.set("12")
        app.new_game()
        app.white_key = "azero"
        app.black_key = "human"
        app.request_ai_move(force=True)
        for _ in range(600):
            app.root.update()
            if app.game.ply >= 1:
                break
            time.sleep(0.02)
        check("AlphaZero 网络走子", app.game.ply >= 1, "ply=%d" % app.game.ply)
    else:
        check("AlphaZero 网络走子", True, "跳过（还没有训练产物）")
    app.white_key = "human"
    app.black_key = "human"

    # 通过 JSON 行控制协议连接一个真正的外部引擎进程
    try:
        app.connect_engine("%s -m sqrtchess proto" % _python_executable())
        connected = app.engine_conn is not None and app.engine_conn.alive
        check("连接外部引擎进程", connected,
              str(app.engine_conn.greeting) if app.engine_conn else "无连接")
        app.new_game()
        app.play_token("1b2-b3")
        check("外部引擎与本地局面同步", app.game.ply == 1)
        app.disconnect_engine()
    except Exception as exc:  # noqa: BLE001
        check("连接外部引擎进程", False, "%s: %s" % (type(exc).__name__, exc))

    print()
    if problems:
        print("界面自检失败 %d 项：%s" % (len(problems), "，".join(problems)))
        return 1
    print("界面自检全部通过。")
    return 0


def install_crash_log() -> Path:
    """把未捕获异常写进文件，便于 pythonw 无控制台启动时排查。"""
    log_path = Path(__file__).resolve().parent / "sqrtchess_gui.log"

    def write_log(text: str) -> None:
        try:
            with open(log_path, "a", encoding="utf-8") as handle:
                handle.write(text + "\n")
        except OSError:
            pass
        try:
            messagebox.showerror("根号棋出错", text[-1500:])
        except Exception:
            pass

    def hook(exc_type, exc_value, tb) -> None:
        write_log("".join(traceback.format_exception(exc_type, exc_value, tb)))

    sys.excepthook = hook

    def tk_hook(self, exc_type, exc_value, tb) -> None:  # type: ignore[no-untyped-def]
        hook(exc_type, exc_value, tb)

    tk.Tk.report_callback_exception = tk_hook  # type: ignore[assignment]
    return log_path


def enable_dpi_awareness() -> None:
    """Windows 上声明 DPI 感知，否则系统会把窗口整体放大 125%/150% 导致超出屏幕。"""
    if sys.platform != "win32":
        return
    try:
        import ctypes

        ctypes.windll.shcore.SetProcessDpiAwareness(1)   # 系统 DPI 感知
    except Exception:  # noqa: BLE001
        try:
            import ctypes

            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:  # noqa: BLE001
            pass


def apply_dpi_scaling(root: tk.Tk) -> None:
    """让字体跟着实际 DPI 缩放（配合 DPI 感知，否则高 DPI 下字会很小）。"""
    try:
        dpi = float(root.winfo_fpixels("1i"))
        if dpi > 0:
            root.tk.call("tk", "scaling", dpi / 72.0)
    except Exception:  # noqa: BLE001
        pass


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    install_crash_log()
    enable_dpi_awareness()
    selftest = "--selftest" in argv
    limit = DEFAULT_NO_CAPTURE_LIMIT
    if "--limit" in argv:
        index = argv.index("--limit")
        if index + 1 < len(argv):
            value = argv[index + 1].strip().lower()
            limit = None if value in ("off", "none") else int(value)
    root = tk.Tk()
    apply_dpi_scaling(root)
    if selftest:
        root.withdraw()
    app = SqrtChessApp(root, no_capture_limit=limit, silent=selftest)
    if selftest:
        code = 1
        try:
            code = run_selftest(app)
        except Exception:  # noqa: BLE001
            traceback.print_exc()
        finally:
            app.on_close()
        return code
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())