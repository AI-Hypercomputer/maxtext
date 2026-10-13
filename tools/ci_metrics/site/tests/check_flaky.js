const fs=require('fs');const {JSDOM}=require('jsdom');
const html=require('./loadpage.js')();
const dom=new JSDOM(html,{runScripts:'dangerously',pretendToBeVisual:true});
const w=dom.window,d=w.document;
let pass=0,fail=0;
function ok(name,cond){if(cond){pass++;console.log('  ok -',name)}else{fail++;console.log('  FAIL -',name)}}
setTimeout(()=>{
  const host=d.querySelector('#flaky');
  ok('renamed card title',d.body.textContent.includes('Flaky jobs and tests'));
  // default range = last 14 days -> 20 merged PRs
  ok('mount populated',host&&host.innerHTML.length>500);
  ok('summary: 14-day window of 20 PRs',host.textContent.includes('The last 14 days')&&host.textContent.includes('20 merged pull requests'));
  let rows=host.querySelectorAll('[data-flane]');
  ok('3 rows in 14-day window',rows.length===3);
  ok('20 cells per row',rows[0].querySelectorAll('[data-cell]').length===20);
  ok('gpu-i first at 15% (3 of 20)',rows[0].textContent.includes('gpu-integration')&&rows[0].textContent.includes('15%')&&rows[0].textContent.includes('3 of 20 runs'));
  ok('gpu-i: 3 amber cells in window',Array.from(rows[0].querySelectorAll('[data-cell]')).filter(c=>c.getAttribute('style').includes('var(--warn)')).length===3);
  const rowT=Array.from(rows).find(r=>r.textContent.includes('tpu-unit'));
  ok('tpu-pre-u: 1 amber + 1 outlined fail cell',
    Array.from(rowT.querySelectorAll('[data-cell]')).filter(c=>c.getAttribute('style').includes('var(--warn)')).length===1&&
    Array.from(rowT.querySelectorAll('[data-cell]')).filter(c=>c.getAttribute('style').includes('2px solid var(--crit)')).length===1);
  ok('caption shows window endpoints',host.textContent.includes('#4894 (08/06)')&&host.textContent.includes('#4908 (08/20)'));
  // 30-day range -> full 30
  d.querySelector('#date-range').value='30';w.renderFlaky();
  rows=d.querySelectorAll('#flaky [data-flane]');
  ok('30d: gpu-i 17% (5 of 30)',rows[0].textContent.includes('17%')&&rows[0].textContent.includes('5 of 30 runs'));
  // custom narrow range -> under the n>=10 gate, zero rescues
  d.querySelector('#date-range').value='custom';
  d.querySelector('#d-from').value='2026-08-17';d.querySelector('#d-to').value='2026-08-18';
  w.renderFlaky();
  ok('custom 2d: 6 PRs + gate sentence',host.textContent.includes('6 merged pull requests')&&host.textContent.includes('has only 6'));
  d.querySelector('#date-range').value='14';w.renderFlaky();
  // explicit x axis under the strips
  const axis=host.querySelector('[data-flaxis]');
  ok('axis row exists',!!axis);
  // The 34px drift: the first grid column was minmax(220px,290px), so it sized from its own
  // content - wide on a job row, narrow on the axis - and the cell strip started in a different
  // place on each. Same template and same min-width in both is what keeps them lined up.
  ok('axis and job rows share one grid geometry',(()=>{
    const ax=d.querySelector('#flaky [data-flaxis]'), row=d.querySelector('#flaky .issue-row');
    const g=e=>/grid-template-columns:([^;"]+)/.exec(e.getAttribute('style'))[1].trim();
    const m=e=>/min-width:(\d+)px/.exec(e.getAttribute('style'))[1];
    return !!ax&&!!row&&g(ax)===g(row)&&m(ax)===m(row)&&!/minmax/.test(g(ax))})());
  ok('the strip width leaves room for the row padding and border',(()=>{
    const row=d.querySelector('#flaky .issue-row');
    const min=+/min-width:(\d+)px/.exec(row.getAttribute('style'))[1];
    const cells=row.querySelectorAll('[data-cell]').length;
    return min===290+16+(cells*20+(cells-1)*2)+16+340+34})());
  ok('axis: columns 20px apart so the rotated lines do not run together',Array.from(d.querySelectorAll('[data-flaxis] [style*="min-width"]')).every(e=>parseInt(/min-width:(\d+)px/.exec(e.getAttribute('style'))[1],10)>=20));
  ok('axis direction matches the strip: cells run oldest to newest',(()=>{const row=d.querySelector('.issue-row [data-cell]');if(!row)return false;
    const idxs=[...d.querySelectorAll('.issue-row')][0].querySelectorAll('[data-flidx]');
    return +idxs[0].getAttribute('data-flidx')< +idxs[idxs.length-1].getAttribute('data-flidx')})());
  ok('axis: title says merged pull request',axis.textContent.includes('merged pull request')&&axis.textContent.includes('oldest'));
  // tests list (now the ranked summary at the top of the card; window back at default 14d here)
  let tRows=host.querySelectorAll('[data-ftlane]');
  ok('tests list: 5 rows at 14d (2 real + 3 synthetic)',tRows.length===5);
  const ring=()=>Array.from(d.querySelectorAll('#flaky [data-ftlane]')).find(r=>r.textContent.includes('test_ring_attention_multihost'));
  ok('tests list: owning job named',ring().textContent.includes('TPU Pretrain Tests (tpu-unit)'));
  // HW filter still dims
  try{d.querySelector('.fchip[data-hw="gpu"]').click()}catch(e){}
  const rows2=d.querySelectorAll('#flaky [data-flane]');
  ok('HW=gpu dims TPU rows',Array.from(rows2).filter(r=>r.getAttribute('data-flane')==='TPU').every(r=>r.getAttribute('style').includes('opacity:0.28')));
  try{d.querySelector('.fchip[data-hw="all"]').click()}catch(e){}
  ok('a job that was never re-run is a failure, not a rescue',(()=>{
     const rows=w.flakyCalc().rows;
     return rows.every(r=>r.evIn.every(e=>!r.j.fails||!r.j.fails.includes(e.pr)))&&
            rows.some(r=>r.failIn.length>0)})());
  // the reported symptom: a rate above 100% and a NaN interval
  ok('wilsonCI refuses inputs that cannot describe a share',
     w.wilsonCI(5,0)===null&&w.wilsonCI(161,41)===null&&w.wilsonCI(-1,10)===null&&
     w.wilsonCI(3,20).every(v=>!isNaN(v)));
  ok('no rate anywhere exceeds 100% and no interval prints NaN',(()=>{
     const t=d.getElementById('flaky').textContent;
     if(/NaN/.test(t))return false;
     return (t.match(/(\d+)%/g)||[]).every(m=>parseInt(m,10)<=100)})());
  ok('a job re-run twice in one pull request counts once',(()=>{
     const j=w.eval('RESCUES')[0];
     const before=w.flakyCalc().rows.find(r=>r.j.id===j.id);
     const n=before?before.prIn.length:0;
     if(!n)return true;                        // nothing of this job falls in the range
     j.events.push({...before.evIn[0]});       // same pull request, a second re-run
     const after=w.flakyCalc().rows.find(r=>r.j.id===j.id);
     const same=after.prIn.length===n&&after.evIn.length===before.evIn.length+1;
     j.events.pop();
     return same})());
  console.log(`\n${pass} passed, ${fail} failed`);
  process.exit(fail?1:0);
},400);
