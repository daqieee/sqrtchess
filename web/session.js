export function resolveSide(choice, random = Math.random) {
  return choice === 'second' ? -1 : choice === 'random' ? (random() < 0.5 ? 1 : -1) : 1;
}
export function boardPosition(square, humanSide) {
  const x = square % 11, y = Math.floor(square / 11);
  return humanSide === 1 ? {x, y: 10-y} : {x: 10-x, y};
}
export function undoPlies(records, humanSide) {
  for (let i = records.length - 1; i >= 0; i--) {
    if (records[i].mover === humanSide) return records.length - i;
  }
  return 0;
}
