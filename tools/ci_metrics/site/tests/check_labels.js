// Direct labels and card copy: the "This chart shows ..." purpose sentence on every card, the value
// labels the suite chart reveals on hover or pin, the hover card's grid and headers, and the axis
// labels of the device and queue charts. Grew out of the 2026-08-27 review batch.
const fs=require('fs');const {JSDOM}=require('jsdom');
const html=require('./loadpage.js')();
const dom=new JSDOM(html,{runScripts:'dangerously',pretendToBeVisual:true});
const w=dom.window,d=w.document;
let pass=0,fail=0;
function ok(name,cond){if(cond){pass++;console.log('  ok -',name)}else{fail++;console.log('  FAIL -',name)}}
setTimeout(()=>{
  const body=d.body.textContent;
  // The page is a bare fragment with no <head>. A server that sends "text/html" with no
  // charset - python -m http.server does exactly that - leaves the browser guessing, and it
  // guesses latin-1, so every glyph in the file (the chip tick, the axis arrows, the fail
  // cross) comes out as mojibake. The declaration has to be in the first 1024 bytes.
  ok('page declares utf-8 before anything else',/^<meta charset="utf-8">/.test(html)&&html.indexOf('<meta charset="utf-8">')<1024);
  // Purpose sentences
  ok('worker purpose',body.includes("This chart shows where each merged pull request's CI time went, one bar per device lane."));
  ok('timeline purpose',body.includes('This chart shows how long CI took on every merged pull request on main'));
  ok('devlines purpose',body.includes("This chart shows how many minutes the TPU, GPU and CPU test jobs spent running their tests on each merged pull request"));
  ok('rerun purpose',body.includes('This chart shows which jobs are flaky'));
  // Worker copy fixes
  ok('no "Pick one suite above"',!body.includes('Pick one suite above'));
  ok('dropdown has no "(stacked bars)"',!d.querySelector('#worker-chart, body').textContent.includes('stacked bars')||!html.includes('All worker suites (stacked bars)'));
  // Devlines legend on its own line
  const dm=d.querySelector('#devlines-meta');
  ok('devlines meta has 3 block lines (plus the guide panel)',dm&&dm.querySelectorAll(':scope > div:not(.card-guide)').length===3&&dm.querySelectorAll(':scope > div.card-guide').length===1);
  // Devlines full axis: 30 rotated labels
  const dl=[...d.querySelectorAll('#devlines svg')].find(x=>x.querySelector('a.axlink'));  // skip the guide button's icon svg
  const dlLabels=dl?Array.from(dl.querySelectorAll('a.axlink text')):[];
  const winN=w.getWindows().current.length;
  const dlDates=dl?Array.from(dl.querySelectorAll('text')).filter(t=>/^\d{2}\/\d{2}$/.test(t.textContent)).length:0,dlTimes=dl?Array.from(dl.querySelectorAll('text')).filter(t=>/^\d{2}:\d{2}$/.test(t.textContent)).length:0;
  ok('devlines: one linked hash + date per window commit, no times',dlLabels.length===winN&&winN>0&&dlDates===winN&&dlTimes===0);
  ok('devlines: labels 12px',dlLabels.every(t=>t.getAttribute('font-size')==='12'));
  console.log(`\n${pass} passed, ${fail} failed`);
  process.exit(fail?1:0);
},400);
