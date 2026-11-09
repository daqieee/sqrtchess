import test from 'node:test';
import assert from 'node:assert/strict';
import {Game, WHITE, BLACK, moveOffsets, chooseMove} from './engine.js';

const sq = (x, y) => y * 11 + x;
const position = (pieces, options = {}) => {
  const cells = Array(121).fill(0);
  for (const [x, y, value] of pieces) cells[sq(x, y)] = value;
  return new Game({cells, ...options});
};
function perft(game, depth) {
  if (!depth || game.result) return 1;
  let n = 0;
  for (const move of game.legalMoves()) {
    game.push(move);
    n += perft(game, depth - 1);
    game.pop();
  }
  return n;
}

test('initial perft matches Python reference through depth 4', () => {
  const game = new Game(), original = game.cells.slice();
  for (const [depth, count] of [[1, 22], [2, 484], [3, 12342], [4, 314721]]) {
    assert.equal(perft(game, depth), count, `depth ${depth}`);
  }
  assert.deepEqual(game.cells, original);
  assert.equal(game.side, WHITE);
  assert.equal(game.halfmove, 0);
  assert.deepEqual(game.history, []);
  assert.equal(game.result, null);
});

test('displacements are exact, jump over pieces, avoid own piece and capture enemy', () => {
  assert.deepEqual(moveOffsets(3), []);
  const game = position([[1, 1, 4], [1, 2, 1], [1, 3, -2], [3, 1, 1], [10, 10, -1]]);
  const moves = game.movesFrom(sq(1, 1));
  assert.ok(moves.some(m => m.to === sq(1, 3)), 'jumps over blocker and captures');
  assert.ok(!moves.some(m => m.to === sq(3, 1)), 'cannot land on own piece');
  assert.ok(moves.every(m => {
    const dx = m.to % 11 - 1, dy = Math.floor(m.to / 11) - 1;
    return dx * dx + dy * dy === 4;
  }));
});

test('capture adds sizes, resets no-capture clock, and undo restores state', () => {
  const game = position([[1, 1, 4], [1, 3, -2], [10, 10, -1]], {halfmove: 17});
  const before = game.cells.slice();
  const info = game.push({from: sq(1, 1), to: sq(1, 3)});
  assert.deepEqual([info.size, info.captured, info.newSize], [4, 2, 6]);
  assert.equal(game.cells[sq(1, 3)], 6);
  assert.equal(game.halfmove, 0);
  assert.equal(game.side, BLACK);
  assert.equal(game.result.reason, 'no-moves'); // captured piece grew into immobile size 6
  assert.equal(game.pop(), info);
  assert.deepEqual(game.cells, before);
  assert.equal(game.halfmove, 17);
  assert.equal(game.result, null);
});

test('end conditions include all captured, either side stuck, and 100 quiet plies', () => {
  let game = position([[0, 0, 1], [1, 0, -1]]);
  game.push({from: sq(0, 0), to: sq(1, 0)});
  assert.deepEqual(game.result, {winner: WHITE, reason: 'all-captured'});
  assert.deepEqual(game.legalMoves(), []);
  assert.throws(() => game.push({from: 1, to: 2}));

  game = position([[0, 0, 3], [10, 10, -1]], {side: BLACK});
  assert.deepEqual(game.result, {winner: null, reason: 'no-moves'});
  game = position([[0, 0, 3], [1, 0, 1], [10, 10, -1]]);
  assert.equal(game.result, null, 'immobile piece alone does not end game');
  game = position([[0, 0, 1], [1, 0, 1], [10, 10, -1]], {halfmove: 99});
  game.push({from: sq(0, 0), to: sq(0, 1)});
  assert.deepEqual(game.result, {winner: WHITE, reason: 'no-capture-limit'});
  game.pop();
  assert.equal(game.result, null);
  assert.equal(game.halfmove, 99);
  game = position([[0, 0, 4], [10, 10, -1]], {halfmove: 100});
  assert.deepEqual(game.result, {winner: null, reason: 'no-capture-limit'}, 'both count and sum must be greater');
});

test('moves for both colors and invalid moves are safely handled', () => {
  const game = new Game(), initial = game.cells.slice();
  assert.equal(game.legalMoves(BLACK).length, 22);
  assert.throws(() => game.push({from: sq(1, 9), to: sq(1, 8)}));
  assert.throws(() => game.push({from: sq(1, 1), to: sq(1, 5)}));
  assert.throws(() => game.pop());
  assert.deepEqual(game.cells, initial);
});

test('GreedyAI chooses the immediate win and leaves the game unchanged', () => {
  const game = position([[0, 0, 1], [1, 0, -1]]);
  const before = game.cells.slice();
  assert.deepEqual(chooseMove(game, () => 0.5), {from: sq(0, 0), to: sq(1, 0)});
  assert.deepEqual(game.cells, before);
  assert.equal(game.side, WHITE);
  assert.deepEqual(game.history, []);
  assert.equal(game.result, null);
  game.push({from: sq(0, 0), to: sq(1, 0)});
  assert.equal(chooseMove(game), null);
});
