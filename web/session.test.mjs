import test from 'node:test';
import assert from 'node:assert/strict';
import {resolveSide, boardPosition, undoPlies} from './session.js';
import {Game, chooseMove} from './engine.js';
test('first, second and both random outcomes',()=>{
 assert.equal(resolveSide('first'),1);assert.equal(resolveSide('second'),-1);
 assert.equal(resolveSide('random',()=>0.1),1);assert.equal(resolveSide('random',()=>0.9),-1);
});
test('human pieces remain at bottom for either side',()=>{
 assert.deepEqual(boardPosition(1,1),{x:1,y:10});
 assert.deepEqual(boardPosition(111,-1),{x:9,y:10});
});
test('both starting sides: AI opening and full-turn undo',()=>{
 for(const side of [1,-1]){
  const game=new Game();const records=[];
  const push=m=>{records.push({mover:game.side});game.push(m);};
  if(side===-1)push(chooseMove(game,()=>0.5));
  assert.equal(game.side,side);assert.equal(undoPlies(records,side),0);
  const opening=game.cells.slice();
  push(game.legalMoves()[0]);assert.equal(undoPlies(records,side),1);
  push(chooseMove(game,()=>0.5));assert.equal(undoPlies(records,side),2);
  for(let i=0;i<2;i++){game.pop();records.pop();}
  assert.equal(game.side,side);assert.deepEqual(game.cells,opening);
 }
});
