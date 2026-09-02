// Date range drives every card (2026-08-31). Drives the REAL header controls.
const fs=require('fs');const {JSDOM}=require('jsdom');
const dom=new JSDOM(require('./loadpage.js')(),{runScripts:'dangerously',pretendToBeVisual:true});
const w=dom.window,d=w.document;let pass=0,fail=0;
const ok=(n,c)=>{if(c){pass++;console.log('  ok -',n)}else{fail++;console.log('  FAIL -',n)}};
const T=n=>n?n.textContent.replace(/\s+/g,' ').trim():'';
const E0='No merged pull requests fall inside this date range.';
const vis=()=>[...d.querySelectorAll('body *:not(script):not(style)')].map(n=>[...n.childNodes].filter(c=>c.nodeType===3).map(c=>c.textContent).join(' ')).join(' ');  // visible text only (body.textContent includes script source)
setTimeout(()=>{
  const sel=d.getElementById('date-range');const setP=v=>{sel.value=v;sel.dispatchEvent(new w.Event('change'))};
  const counts=()=>({win:w.getWindows().current.length,fl:w.flakyCalc().n,
    tl:d.querySelectorAll('#timeline svg a.axlink').length,wk:d.querySelectorAll('#worker-chart svg a.axlink').length,
    dev:d.querySelectorAll('#devlines svg a.axlink').length,
    flAx:d.querySelectorAll('[data-flaxis] span').length});
  const exp={'7':14,'14':20,'30':30,'90':30};
  for(const [v,n] of Object.entries(exp)){
    setP(v);const c=counts();
    ok(`preset ${v}: every card draws the same ${n} PRs (${JSON.stringify(c)})`,Object.values(c).every(x=>x===n));
    const lab=`The last ${v} days covers ${n} merged pull requests`;
    ok(`preset ${v}: captions name the range and count`,T(d.getElementById('tl-zoombar')).includes(lab)&&T(d.getElementById('wk-meta')).includes(lab)&&T(d.getElementById('devlines-meta')).includes(lab)&&T(d.getElementById('flaky')).includes(lab));
    ok(`preset ${v}: one guide button per card after re-render`,[...new Set([...d.querySelectorAll('[data-gbtn]')].map(b=>b.getAttribute('data-gbtn')))].every(k=>d.querySelectorAll(`[data-gbtn="${k}"]`).length===1));
  ok(`preset ${v}: no stale wording (this window / UTC / NaN / Infinity)`,!/this window|UTC|NaN|Infinity/.test(vis()));
  }
  setP('14');
  ok('14d: brand dot follows the newest PR in range (pass = good)',d.getElementById('hdot').style.background==='var(--good)'&&d.getElementById('hdot').title.includes('#4908'));
  setP('30');
  setP('7');
  // zoom state resets on range change
  setP('14');w.tlZoomTo(4,7);w.devZoomTo(2,5);
  ok('zooms active before the switch',w.eval('TL_ZOOM')!==null&&w.eval('DEV_ZOOM')!==null);
  setP('7');
  ok('range change clears both brush zooms',w.eval('TL_ZOOM')===null&&w.eval('DEV_ZOOM')===null);
  // custom range + cancel
  d.getElementById('d-from').value='2026-08-19';d.getElementById('d-to').value='2026-08-20';sel.value='custom';
  const apply=[...d.querySelectorAll('#date-custom button')].find(b=>T(b)==='Apply');apply.click();
  const cc=counts();
  ok(`custom Aug 19-20: 7 PRs everywhere (${JSON.stringify(cc)})`,Object.values(cc).every(x=>x===7)&&w.getWindows().label==='The selected date range');
  ok('custom: captions say the selected date range with real endpoints',T(d.getElementById('tl-zoombar')).includes('The selected date range covers 7 merged pull requests, #4920 (Aug 19) to #4908 (Aug 20)')&&T(d.getElementById('flaky')).includes('in the selected date range'));
  const cancel=[...d.querySelectorAll('#date-custom button')].find(b=>T(b)==='✕');cancel.click();
  ok('cancel (✕) resets to 14 days AND re-renders',sel.value==='14'&&counts().win===20&&T(d.getElementById('tl-zoombar')).includes('The last 14 days covers 20'));
  // empty range
  d.getElementById('d-from').value='2026-07-01';d.getElementById('d-to').value='2026-07-10';sel.value='custom';apply.click();
  const hosts=['timeline','wk-meta','devlines','flaky'];
  ok('empty range: every card shows the one sentence, nothing throws',hosts.every(id=>T(d.getElementById(id)).includes(E0))&&T(d.getElementById('flaky')).includes('No merged pull requests')&&d.getElementById('hdot').style.background==='var(--muted)');
  ok('empty range: no NaN / Infinity anywhere',!/NaN|Infinity/.test(vis()));
  // single-PR range
  d.getElementById('d-from').value='2026-08-20';d.getElementById('d-to').value='2026-08-20';apply.click();
  ok('single-day range: 2 bars and no median line',counts().tl===2&&![...d.querySelectorAll('#timeline svg text')].some(t=>t.textContent.startsWith('median ')));
  console.log(`\n${pass} passed, ${fail} failed`);process.exit(fail?1:0);
},500);
