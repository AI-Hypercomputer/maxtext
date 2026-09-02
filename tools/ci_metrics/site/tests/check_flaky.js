const fs=require('fs');const {JSDOM}=require('jsdom');
const html=fs.readFileSync(require('path').join(__dirname,'..','index.html'),'utf-8');
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
  ok('summary: 5 rescues, 66 minutes',host.textContent.includes('5 failed first attempts')&&host.textContent.includes('66 extra machine minutes'));
  let rows=host.querySelectorAll('[data-flane]');
  ok('3 rows in 14-day window',rows.length===3);
  ok('20 cells per row',rows[0].querySelectorAll('[data-cell]').length===20);
  ok('gpu-i first at 15% (3 of 20)',rows[0].textContent.includes('gpu-integration')&&rows[0].textContent.includes('15%')&&rows[0].textContent.includes('3 of 20 first attempts'));
  ok('gpu-i: 3 amber cells in window',Array.from(rows[0].querySelectorAll('[data-cell]')).filter(c=>c.getAttribute('style').includes('var(--warn)')).length===3);
  const rowT=Array.from(rows).find(r=>r.textContent.includes('tpu-unit'));
  ok('tpu-pre-u: 1 amber + 1 outlined fail cell',
    Array.from(rowT.querySelectorAll('[data-cell]')).filter(c=>c.getAttribute('style').includes('var(--warn)')).length===1&&
    Array.from(rowT.querySelectorAll('[data-cell]')).filter(c=>c.getAttribute('style').includes('2px solid var(--crit)')).length===1);
  ok('caption shows window endpoints',host.textContent.includes('#4894 (Aug 06)')&&host.textContent.includes('#4908 (Aug 20)'));
  // popup respects the window
  w.openFlakyPop('gpu-i');
  const body=d.querySelector('#wkpop-body');
  ok('popup: 3 of 20 + last 14 days',body.textContent.includes('3 of 20 merged pull requests')&&body.textContent.includes('the last 14 days'));
  ok('popup wilson for 3/20 = 5-36%',/interval 5[–-]36%/.test(body.textContent));
  ok('popup lists only window events',(body.innerHTML.match(/m wasted/g)||[]).length===3);
  w.closeWkPop();
  // 30-day range -> full 30
  d.querySelector('#date-range').value='30';w.renderFlaky();
  rows=d.querySelectorAll('#flaky [data-flane]');
  ok('30d: 30 cells, 8 rescues, 106m',rows[0].querySelectorAll('[data-cell]').length===30&&host.textContent.includes('8 failed first attempts')&&host.textContent.includes('106 extra machine minutes'));
  ok('30d: gpu-i 17% (5 of 30)',rows[0].textContent.includes('17%')&&rows[0].textContent.includes('5 of 30 first attempts'));
  // 7-day range -> 14 PRs, pathways row drops out
  d.querySelector('#date-range').value='7';w.renderFlaky();
  rows=d.querySelectorAll('#flaky [data-flane]');
  ok('7d: 2 rows, 14 cells',rows.length===2&&rows[0].querySelectorAll('[data-cell]').length===14);
  ok('7d: 2 rescues, 33 minutes',host.textContent.includes('2 failed first attempts')&&host.textContent.includes('33 extra machine minutes'));
  // custom narrow range -> under the n>=10 gate, zero rescues
  d.querySelector('#date-range').value='custom';
  d.querySelector('#d-from').value='2026-08-17';d.querySelector('#d-to').value='2026-08-18';
  w.renderFlaky();
  ok('custom 2d: 6 PRs + gate sentence',host.textContent.includes('6 merged pull requests')&&host.textContent.includes('has only 6'));
  ok('custom 2d: zero-state sentence',host.textContent.includes('No job needed a re-run to pass in the selected date range'));
  d.querySelector('#date-range').value='14';w.renderFlaky();
  // cell tooltip outcomes
  w.flakyCellTip({clientX:200,clientY:200},'gpu-i',11);
  const tip=d.querySelector('#tip');
  ok('tooltip: rescue cell cause + wasted',tip.textContent.includes('CUDA OOM')&&tip.textContent.includes('8.5m wasted'));
  w.flakyCellTip({clientX:200,clientY:200},'tpu-pre-u',17);
  ok('tooltip: fail cell says never re-run',tip.textContent.includes('never re-run'));
  w.flakyCellTip({clientX:200,clientY:200},'gpu-i',20);
  ok('tooltip: clean cell passed first attempt',tip.textContent.includes('Passed on the first attempt'));
  w.flakyTipHide();
  // cell tooltip lists the failed tests of THAT run only (part E)
  const tipTxt=()=>tip.textContent.replace(/\s+/g,' ');
  w.flakyCellTip({clientX:200,clientY:200},'gpu-i',15);  // #4904
  ok('tooltip #4904: 3 tests, repeat offenders first, each with its own failure line',tipTxt().includes('3 tests failed on the first attempt and passed on the re-run')&&tipTxt().indexOf('test_gpu_convergence_a100_bf16')<tipTxt().indexOf('test_gpu_moe_expert_parallel_bf16')&&tipTxt().indexOf('test_gpu_moe_expert_parallel_bf16')<tipTxt().indexOf('test_gpu_checkpoint_restore_sharded')&&tipTxt().includes('NCCL timeout after 1800s (rank 1)'));
  ok('tooltip #4904: outcome line drops the duplicate cause, keeps minutes',tipTxt().includes('Failed first, passed on re-run · 12m wasted')&&tip.style.maxWidth==='430px');
  w.flakyCellTip({clientX:200,clientY:200},'gpu-i',23);  // #4920
  ok('tooltip #4920: single test wording',tipTxt().includes('1 test failed on the first attempt')&&tipTxt().includes('test_gpu_convergence_a100_bf16'));
  w.flakyCellTip({clientX:200,clientY:200},'tpu-pw-i',12);  // #4898 worker crash
  ok('tooltip pathways: no single test failed + cause',tipTxt().includes('No single test failed in this run: the job failed at step "Start Pathways Daemons"'));
  w.flakyCellTip({clientX:200,clientY:200},'tpu-pre-u',17);  // #4940 never re-run, r=0
  ok('tooltip #4940: no test ran, cancelled before pytest',tipTxt().includes('never re-run')&&/No test ran: the job waited \d+ minutes for a runner and was cancelled before pytest started/.test(tipTxt()));
  w.eval("for(let k=0;k<5;k++)FLAKY_TESTS.push({test:'zz_temp_'+k,job:'gpu-i',prs:['#4896'],cause:'temp'})");
  w.flakyCellTip({clientX:200,clientY:200},'gpu-i',11);  // #4896 now 8 tests
  ok('tooltip cap: 6 listed + "and 2 more failed in the same run"',tipTxt().includes('8 tests failed')&&(tip.innerHTML.match(/zz_temp_/g)||[]).length===3&&tipTxt().includes('and 2 more failed in the same run'));
  w.eval("FLAKY_TESTS.splice(-5,5)");
  w.flakyTipHide();
  ok('hide resets the tooltip width',tip.style.maxWidth==='');
  ok('description promises the hover content',d.body.textContent.includes('Hover a cell to see which tests failed in that run'));
  // explicit x axis under the strips
  const axis=host.querySelector('[data-flaxis]');
  ok('axis row exists',!!axis);
  const axLabels=axis?axis.querySelectorAll('span'):[];
  ok('axis: one hash label per cell (20 at 14d)',axLabels.length===20);
  ok('axis: first #4894 Aug 06, last #4908 Aug 20 (dates only)',axLabels[0]&&axLabels[0].textContent==='#4894 Aug 06'&&axLabels[axLabels.length-1].textContent==='#4908 Aug 20');
  ok('axis: labels rotated, >=12px',Array.from(axLabels).every(s=>s.getAttribute('style').includes('rotate(-55deg)')&&s.getAttribute('style').includes('font-size:12px')));
  ok('axis: title says merged pull request',axis.textContent.includes('merged pull request')&&axis.textContent.includes('oldest'));
  // tests list (now the ranked summary at the top of the card; window back at default 14d here)
  let tRows=host.querySelectorAll('[data-ftlane]');
  ok('tests list: 5 rows at 14d (2 real + 3 synthetic)',tRows.length===5);
  ok('tests list: gpu convergence first, 3 of 20, 32m',tRows[0].textContent.includes('test_gpu_convergence_a100_bf16')&&tRows[0].textContent.includes('3 of 20')&&tRows[0].textContent.includes('32m'));
  const ring=()=>Array.from(d.querySelectorAll('#flaky [data-ftlane]')).find(r=>r.textContent.includes('test_ring_attention_multihost'));
  ok('tests list: ring test 1 of 20, 22m, last #4920',!!ring()&&ring().textContent.includes('1 of 20')&&ring().textContent.includes('22m')&&ring().textContent.includes('last in #4920'));
  ok('tests list: owning job named',ring().textContent.includes('TPU Pretrain Tests (tpu-unit)'));
  ok('tests list: pathways crash exclusion sentence',host.textContent.includes('One of the 5 rescued runs (TPU Pathways) has no test here: the Pathways jobs publish no test results, so no test can be named'));
  ok('tests list: no-backfill disclosure',host.textContent.includes('GitHub deletes them after about a day, so older runs cannot be added'));
  d.querySelector('#date-range').value='30';w.renderFlaky();
  tRows=d.querySelectorAll('#flaky [data-ftlane]');
  ok('tests list 30d: convergence 5 of 30 / 52m first, ring 2 of 30 / 41m',tRows[0].textContent.includes('5 of 30')&&tRows[0].textContent.includes('52m')&&ring().textContent.includes('2 of 30')&&ring().textContent.includes('41m'));
  d.querySelector('#date-range').value='custom';
  d.querySelector('#d-from').value='2026-08-17';d.querySelector('#d-to').value='2026-08-18';
  w.renderFlaky();
  ok('tests table custom 2d: zero-state',host.textContent.includes('No test needed a re-run to pass in the selected date range'));
  d.querySelector('#date-range').value='14';w.renderFlaky();
  const small=(host.innerHTML.match(/font-size:(\d+(?:\.\d+)?)px/g)||[]).map(s=>parseFloat(s.slice(10))).filter(v=>v<12);
  ok('no font below 12px in card',small.length===0);
  // HW filter still dims
  try{d.querySelector('.fchip[data-hw="gpu"]').click()}catch(e){}
  const rows2=d.querySelectorAll('#flaky [data-flane]');
  ok('HW=gpu dims TPU rows',Array.from(rows2).filter(r=>r.getAttribute('data-flane')==='TPU').every(r=>r.getAttribute('style').includes('opacity:0.28')));
  try{d.querySelector('.fchip[data-hw="all"]').click()}catch(e){}
  // footnote = facts only; rules moved into the How to read panel (2026-08-31 part Q)
  w.initGuides();  // the observer re-attaches guides on a microtask; force it so the assert below sees the rebuilt panel
  const flFoot=host.querySelector('[data-flfoot]'),flPanel=d.getElementById('guide-flaky');
  ok('footnote states the coverage fact only',!!flFoot&&/^The other \d+ of \d+ jobs needed no re-run in the (last \d+ days|selected date range)\.$/.test(flFoot.textContent.trim()));
  ok('rules moved into the guide panel',!!flPanel&&flPanel.textContent.includes('do not count toward the rate')&&flPanel.textContent.includes('at least 10 first attempts')&&flPanel.textContent.includes('An amber rate means at least 1 in 10'));
  ok('Flakiest tests lead lives in a guide panel, no window jargon',!!d.getElementById('guide-flaky-sum')&&d.getElementById('guide-flaky-sum').textContent.includes('Tests with the most such runs come first')&&d.getElementById('guide-flaky-sum').textContent.includes('"3 of 20 runs" means that in 3 of those 20 runs the test failed on the first attempt')&&d.getElementById('guide-flaky-sum').textContent.includes('in the other 17 runs it passed first time')&&d.getElementById('guide-flaky-sum').textContent.includes('The 20 is the same for every row')&&host.querySelector('.sum-head').textContent.includes('Failed first try')&&d.getElementById('guide-flaky-sum').textContent.includes('in the last 14 days')&&!host.textContent.includes('this window'));
  console.log(`\n${pass} passed, ${fail} failed`);
  process.exit(fail?1:0);
},400);
