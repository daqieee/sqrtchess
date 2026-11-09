// Square Root Chess rules and GreedyAI, ported from sqrtchess/core.py and ai.py.
// Squares are numbered y * 11 + x, with a1 = 0; positive pieces are white.
export const BOARD_SIZE = 11;
export const WHITE = 1;
export const BLACK = -1;
const NUM_SQUARES = BOARD_SIZE * BOARD_SIZE;
const EDGE = [4, 8, 5, 2, 9, 2, 5, 8, 4];
const offsetsCache = new Map();

export function initialCells() {
  const cells = Array(NUM_SQUARES).fill(0);
  EDGE.forEach((size, index) => {
    const x = index + 1;
    cells[x] = size;
    cells[10 * BOARD_SIZE + x] = -size;
    cells[BOARD_SIZE + x] = 1;
    cells[9 * BOARD_SIZE + x] = -1;
  });
  return cells;
}

export function moveOffsets(size) {
  size = Math.trunc(size);
  if (offsetsCache.has(size)) return offsetsCache.get(size);
  const offsets = [];
  if (size > 0) {
    const limit = Math.floor(Math.sqrt(size));
    for (let dx = -limit; dx <= limit; dx++) {
      for (let dy = -limit; dy <= limit; dy++) {
        if (dx * dx + dy * dy === size) offsets.push([dx, dy]);
      }
    }
  }
  offsetsCache.set(size, offsets);
  return offsets;
}

export function squareName(sq) {
  if (!Number.isInteger(sq) || sq < 0 || sq >= NUM_SQUARES) throw new RangeError('Invalid square');
  return 'abcdefghijk'[sq % BOARD_SIZE] + (Math.floor(sq / BOARD_SIZE) + 1);
}

export class Game {
  constructor({cells = initialCells(), side = WHITE, halfmove = 0, noCaptureLimit = 100} = {}) {
    if (!Array.isArray(cells) || cells.length !== NUM_SQUARES || !cells.every(Number.isInteger)) {
      throw new Error('Board must contain 121 integer cells');
    }
    if (side !== WHITE && side !== BLACK) throw new Error('Invalid side');
    this.cells = cells.slice();
    this.side = side;
    this.halfmove = halfmove;
    this.noCaptureLimit = noCaptureLimit;
    this.result = null;
    this.history = [];
    this._snapshots = [];
    this._refreshResult();
  }

  stats(color) {
    let count = 0, total = 0;
    for (const value of this.cells) {
      if (value * color > 0) { count++; total += Math.abs(value); }
    }
    return {count, total};
  }

  movesFrom(sq) {
    if (!Number.isInteger(sq) || sq < 0 || sq >= NUM_SQUARES) return [];
    const value = this.cells[sq];
    if (!value) return [];
    const x = sq % BOARD_SIZE, y = Math.floor(sq / BOARD_SIZE), moves = [];
    for (const [dx, dy] of moveOffsets(Math.abs(value))) {
      const nx = x + dx, ny = y + dy;
      if (nx < 0 || nx >= BOARD_SIZE || ny < 0 || ny >= BOARD_SIZE) continue;
      const to = ny * BOARD_SIZE + nx, target = this.cells[to];
      if (!target || target * value < 0) moves.push({from: sq, to});
    }
    return moves;
  }

  legalMoves(color = this.side) {
    if (this.result) return [];
    const moves = [];
    for (let sq = 0; sq < NUM_SQUARES; sq++) {
      if (this.cells[sq] * color > 0) moves.push(...this.movesFrom(sq));
    }
    return moves;
  }

  _hasMoves(color) {
    for (let sq = 0; sq < NUM_SQUARES; sq++) {
      if (this.cells[sq] * color > 0 && this.movesFrom(sq).length) return true;
    }
    return false;
  }

  _refreshResult() {
    const white = this.stats(WHITE), black = this.stats(BLACK);
    if (!white.count || !black.count) {
      const winner = !white.count && !black.count ? null : white.count ? WHITE : BLACK;
      this.result = {winner, reason: 'all-captured'};
    } else if (!this._hasMoves(WHITE) || !this._hasMoves(BLACK)) {
      this.result = this._adjudicate('no-moves', white, black);
    } else if (this.noCaptureLimit !== null && this.halfmove >= this.noCaptureLimit) {
      this.result = this._adjudicate('no-capture-limit', white, black);
    } else {
      this.result = null;
    }
    return this.result;
  }

  _adjudicate(reason, white, black) {
    const winner = white.count > black.count && white.total > black.total ? WHITE :
      black.count > white.count && black.total > white.total ? BLACK : null;
    return {winner, reason};
  }

  push(move) {
    if (this.result) throw new Error('Game is over');
    const {from, to} = move ?? {};
    if (!Number.isInteger(from) || !Number.isInteger(to) || from < 0 || from >= NUM_SQUARES || to < 0 || to >= NUM_SQUARES ||
        this.cells[from] * this.side <= 0 || !this.movesFrom(from).some(candidate => candidate.to === to)) {
      throw new Error('Illegal move');
    }
    const mover = this.side, size = Math.abs(this.cells[from]), captured = Math.abs(this.cells[to]);
    const newSize = size + captured;
    this._snapshots.push({cells: this.cells.slice(), side: this.side, halfmove: this.halfmove, result: this.result});
    this.cells[from] = 0;
    this.cells[to] = mover * newSize;
    this.halfmove = captured ? 0 : this.halfmove + 1;
    this.side = -mover;
    const info = {from, to, mover, size, captured, newSize, ply: this.history.length + 1, halfmove: this.halfmove};
    this.history.push(info);
    this._refreshResult();
    return info;
  }

  pop() {
    if (!this._snapshots.length) throw new Error('No moves to undo');
    const previous = this._snapshots.pop(), info = this.history.pop();
    this.cells = previous.cells;
    this.side = previous.side;
    this.halfmove = previous.halfmove;
    this.result = previous.result;
    return info;
  }
}

// Same score, terminal bonus and default ±0.6 random noise as Python's GreedyAI.
// An injected random function makes behaviour reproducible in tests.
export function chooseMove(game, random = Math.random) {
  const moves = game.legalMoves();
  if (!moves.length) return null;
  let best = -Infinity, pool = [];
  for (const move of moves) {
    const value = game.cells[move.from], size = Math.abs(value);
    const captured = Math.abs(game.cells[move.to]), mover = Math.sign(value);
    let score = captured * 10 + (size + captured) * 0.4;
    game.push(move);
    try {
      let worst = 0;
      for (let sq = 0; sq < NUM_SQUARES; sq++) {
        const other = game.cells[sq];
        if (other * mover < 0 && game.movesFrom(sq).some(reply => reply.to === move.to)) {
          worst = Math.max(worst, Math.abs(other));
        }
      }
      score -= worst * 3;
      if (game.result) score += game.result.winner === null ? -2 : game.result.winner === mover ? 100000 : -100000;
    } finally {
      game.pop();
    }
    score += (random() * 2 - 1) * 0.6;
    if (score > best + 1e-9) { best = score; pool = [move]; }
    else if (score >= best - 1e-9) pool.push(move);
  }
  return pool[Math.floor(random() * pool.length)];
}
