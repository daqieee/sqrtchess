import {Game, chooseMove} from './engine.js';
import {resolveSide, boardPosition, undoPlies} from './session.js';
const $ = id => document.getElementById(id);
let game = new Game(), selected = null, busy = false, timer = null, generation = 0, records = [], humanSide = 1;
const square = n => 'abcdefghijk'[n % 11] + (Math.floor(n / 11) + 1);
function render() {
  const focused = document.activeElement?.dataset?.square;
  const legal = selected === null ? [] : game.movesFrom(selected);
  let html = '<div class="grid-lines"></div>';
  for (const x of [3,5,7]) for (const y of [3,5,7]) html += `<i class="star" style="left:${5.5+x*8.9}%;top:${5.5+y*8.9}%"></i>`;
  for (let i=0;i<11;i++) html += `<span class="coordinate" style="left:${5.5+i*8.9}%;top:98%">${'abcdefghijk'[humanSide===1?i:10-i]}</span><span class="coordinate" style="left:1%;top:${5.5+i*8.9}%">${humanSide===1?11-i:i+1}</span>`;
  const last = records.at(-1);
  for (let n=0;n<121;n++) {
    const pos=boardPosition(n,humanSide), v=game.cells[n], target=legal.some(m=>m.to===n), prev=last&&(last.from===n||last.to===n);
    html += `<button class="point ${v>0?'white':v<0?'black':''} ${selected===n?'selected':''} ${target?'target':''} ${target&&v*humanSide<0?'capture':''} ${prev?'previous':''}" style="left:${5.5+pos.x*8.9}%;top:${5.5+pos.y*8.9}%" data-square="${n}" aria-label="${square(n)} ${v?(v>0?'白棋':'黑棋')+' '+Math.abs(v):'空位'}${target?' 可落子':''}" aria-pressed="${selected===n}">${v?`<span class="piece">${Math.abs(v)}</span>`:''}</button>`;
  }
  $('board').innerHTML=html;
  for (const [id,color] of [['white-stats',humanSide],['black-stats',-humanSide]]) {const s=game.stats(color);$(id).innerHTML=`${s.count} <small>枚棋子</small><b>${s.total} <small>大小和</small></b>`;}
  $('ai-label').textContent=(humanSide===1?'黑方':'白方')+' · 贪心策略';
  $('human-label').textContent=(humanSide===1?'白方':'黑方')+' · 从这里开始';
  $('human-order').textContent=humanSide===1?'先手':'后手';
  $('ai-avatar').className='avatar '+(humanSide===1?'dark':'light');
  $('human-avatar').className='avatar '+(humanSide===1?'light':'dark');
  const result=game.result;
  $('status').textContent=result?(result.winner===humanSide?'你赢了！':result.winner===-humanSide?'AI 获胜':'本局平局'):busy?'AI 正在思考…':'轮到你了';
  $('hint').textContent=result?`${{'all-captured':'一方棋子已全部被吃光。','no-moves':'有一方已无子可动，按数量与大小和判定。','no-capture-limit':'连续 100 步未吃子，按数量与大小和判定。'}[result.reason]||'对局结束。'}`:busy?'它正在权衡吃子收益与落点风险。':selected===null?`选择一枚${humanSide===1?'白':'黑'}棋，查看它能到达的位置。`:legal.length?'点击绿色标记移动；点击空白取消选择。':'这枚棋子暂时没有合法落点，试试其他棋子。';
  $('round').textContent=String(Math.floor(records.length/2)+1).padStart(2,'0');
  $('quiet').textContent=`${game.halfmove} / 100`;
  $('undo').disabled=undoPlies(records,humanSide)===0;
  $('selected-number').textContent=selected===null?'√n':String(Math.abs(game.cells[selected]));
  $('selected-title').textContent=selected===null?'数字决定步伐':`${square(selected)} · ${humanSide===1?'白方':'黑方'}棋子`;
  $('selected-info').textContent=selected===null?'点击棋子，探索它的走法':`移动 √${Math.abs(game.cells[selected])} 格 · ${legal.length} 个落点`;
  $('move-count').textContent=records.length+' 步';
  $('history').innerHTML=records.length?records.map((r,i)=>`<div class="history-row"><span>${String(i+1).padStart(2,'0')}</span><span>${r.mover===humanSide?'你':'AI'}</span><strong>${r.size}${square(r.from)}${r.capture?' × ':' → '}${square(r.to)}</strong>${r.capture?`<span>+${r.capture}</span>`:'<span>—</span>'}</div>`).join(''):'<div class="empty-history"><span>↗</span><p>故事，从你的第一步开始。</p></div>';
  $('history').scrollTop=$('history').scrollHeight;
  if(focused!==undefined)document.querySelector(`[data-square="${focused}"]`)?.focus({preventScroll:true});
}
function move(m) {const detail={...m,mover:game.side,size:Math.abs(game.cells[m.from]),capture:Math.abs(game.cells[m.to])};game.push(m);records.push(detail);selected=null;}
function playAI() {
  if(game.result||game.side===humanSide)return;
  busy=true;render();const token=generation;
  timer=setTimeout(()=>{if(token!==generation)return;try {const m=chooseMove(game);if(m)move(m);busy=false;render();}catch(error){busy=false;render();$('status').textContent='AI 暂时无法落子';$('hint').textContent='请悔棋或开始新对局重试。';console.error(error);}},350);
}
$('board').addEventListener('click',e=>{const button=e.target.closest('[data-square]');if(!button||busy||game.result||game.side!==humanSide)return;const n=Number(button.dataset.square);if(selected!==null){const m=game.movesFrom(selected).find(m=>m.to===n);if(m){move(m);render();playAI();return;}}selected=game.cells[n]*humanSide>0&&n!==selected?n:null;render();});
function reset(){generation++;clearTimeout(timer);busy=false;game=new Game();records=[];selected=null;humanSide=resolveSide($('side-choice').value);render();playAI();}
$('restart').onclick=()=>{if(records.length)$('confirm').showModal();else reset();};
$('confirm-reset').onclick=()=>{$('confirm').close();reset();};$('cancel-reset').onclick=()=>$('confirm').close();
$('undo').onclick=()=>{const count=undoPlies(records,humanSide);if(!count)return;generation++;clearTimeout(timer);busy=false;for(let i=0;i<count;i++){game.pop();records.pop();}selected=null;render();};
$('rules-open').onclick=()=>$('rules').showModal();$('rules-close').onclick=()=>$('rules').close();
for(const id of ['rules','confirm'])$(id).addEventListener('click',e=>{if(e.target===$(id)){const r=$(id).getBoundingClientRect();if(e.clientX<r.left||e.clientX>r.right||e.clientY<r.top||e.clientY>r.bottom)$(id).close();}});
render();
