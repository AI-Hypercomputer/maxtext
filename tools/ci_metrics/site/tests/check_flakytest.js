// Flakiest tests rows point at the test, not at the job it runs in (2026-09-01 review).
// A row click opens the test's own popup, and hovering a row lights up, in the Re-runs per
// job strips, exactly the pull requests where that test failed on its first attempt.
const {JSDOM}=require('jsdom'),fs=require('fs'),path=require('path');
const html=require('./loadpage.js')();
const dom=new JSDOM(html,{runScripts:'dangerously',pretendToBeVisual:true});
const w=dom.window,d=w.document;let p=0,f=0;
const ok=(c,m)=>{if(c){p++;console.log('  ok - '+m)}else{f++;console.log('  FAIL - '+m)}};
setTimeout(()=>{
  // --- 2. flakiest tests rows point at the test ---
  const flakyRows=[...d.querySelectorAll('#flaky .sum-row')].filter(r=>r.getAttribute('onclick')||'');
  const testRow=[...d.querySelectorAll('#flaky .sum-row')].find(r=>(r.getAttribute('onclick')||'').includes('openFlakyTestPop'));
  ok(!!testRow,'a Flakiest tests row opens openFlakyTestPop, not openFlakyPop');
  ok((testRow.getAttribute('onclick')||'').includes('test_gpu_convergence_a100_bf16'),'row carries its own test name');
  ok(!!testRow.getAttribute('onmouseenter'),'row has a hover handler');
  // cells are addressable
  const cells=[...d.querySelectorAll('#flaky [data-cell]')];
  ok(cells.length>0&&cells.every(x=>x.hasAttribute('data-fljob')&&x.hasAttribute('data-flidx')),'every strip cell carries job + index');
  // hover highlights only that test's PRs
  // The page now passes the job id as a JSON string ("..."), so accept either quote.
  const m=(testRow.getAttribute('onmouseenter')||'').match(/flakyTestHl\(["']([^"']*)["'],'([^']*)'/);
  ok(!!m,'hover handler parses'); 
  w.flakyTestHl(m[1],m[2],true);
  const want=new Set(m[2].split(',').filter(Boolean));
  const lit=cells.filter(x=>x.style.opacity==='1');
  const dim=cells.filter(x=>x.style.opacity==='0.22');
  ok(lit.length===want.size,'lit cells == the test\'s PRs: '+lit.length+' vs '+want.size);
  ok(lit.every(x=>x.getAttribute('data-fljob')===m[1]),'lit cells all belong to the owning job');
  ok(dim.length>0,'other cells dim');
  w.flakyTestHl(m[1],m[2],false);
  ok(cells.every(x=>!x.style.opacity),'hover out clears every cell');
  console.log(`\n${p} passed, ${f} failed`); process.exit(f?1:0);
},400);
