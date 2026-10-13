// 2026-08-31 part D: summary-first blocks on Main cards 3 (device), 4 (flaky), 5 (suite health).
const fs=require('fs');const {JSDOM}=require('jsdom');
const html=require('./loadpage.js')();
const dom=new JSDOM(html,{runScripts:'dangerously',pretendToBeVisual:true});
const w=dom.window,d=w.document;
let pass=0,fail=0;
function ok(name,cond){if(cond){pass++;console.log('  ok -',name)}else{fail++;console.log('  FAIL -',name)}}
const txt=n=>n?n.textContent.replace(/\s+/g,' ').trim():'';
setTimeout(()=>{
  // ---------- card 3: Test run time by device ----------
  const dev=d.querySelector('#devlines');
  const ds=dev.querySelector('.sum[data-sum="dev"]');
  ok('device summary present and placed before the phase chips',!!ds&&dev.children[0].contains(ds)&&!!dev.children[1].querySelector('.fchip'));
  const dr=[...ds.querySelectorAll('[data-devsum]')];
  ok('3 device rows',dr.length===3);
  ok('TPU first: #4940 no tests ran',dr[0].getAttribute('data-devsum')==='tpu'&&txt(dr[0]).includes('#4940: no tests ran'));
  ok('TPU: biggest rise +9m at #4916, inside the usual range',txt(dr[0]).includes('+9m at #4916')&&txt(dr[0]).includes('inside the usual range'));
  ok('CPU second: flagged +12m at #4916 (35m vs usual 23m), back to usual',dr[1].getAttribute('data-devsum')==='cpu'&&txt(dr[1]).includes('flagged')&&txt(dr[1]).includes('+12m at #4916')&&txt(dr[1]).includes('35m against a usual 23m')&&txt(dr[1]).includes('back to usual'));
  ok('GPU last: inside the usual range, latest 24m',dr[2].getAttribute('data-devsum')==='gpu'&&txt(dr[2]).includes('inside the usual range')&&txt(dr[2]).includes('latest run 24m'));
  ok('rank badges: crit / warn / neutral',dr[0].querySelector('.sum-rank').style.borderColor==='var(--crit)'&&dr[1].querySelector('.sum-rank').style.borderColor==='var(--warn)'&&dr[2].querySelector('.sum-rank').style.borderColor==='var(--muted)');
  const rings=[...dev.querySelectorAll('svg circle')].filter(c=>c.getAttribute('stroke')==='#F59E0B');
  ok('chart: exactly one anomaly ring (CPU #4916) after the prior-direction fix',rings.length===1);
  dr[1].click();
  ok('CPU row click opens the #4916 commit modal',d.querySelector('#modal').textContent.includes('#4916'));
  w.closeModal();
  w.setDevPhase('queue');
  ok('phase switch re-words the lead (queue wait time)',txt(d.querySelector('#devlines .sum .sum-lead')).includes('queue wait time'));
  w.setDevPhase('run');
  // ---------- card 4: Flaky jobs and tests ----------
  const fl=d.querySelector('#flaky');
  const fs4=()=>fl.querySelector('.sum[data-sum="flaky"]');
  ok('flaky summary present, before the job strips',!!fs4()&&(fs4().compareDocumentPosition(fl.querySelector('[data-flane]'))&4)===4);
  const fr=()=>[...fl.querySelectorAll('.sum [data-ftlane]')];
  const names=()=>fr().map(r=>r.querySelector('.mono').textContent.trim());
  ok('14d: 5 tests ranked by re-runs, then minutes',JSON.stringify(names())===JSON.stringify(['test_gpu_convergence_a100_bf16','test_gpu_moe_expert_parallel_bf16','test_ring_attention_multihost','test_gpu_checkpoint_restore_sharded','test_gpu_flash_attention_backward']));
  ok('all test rows live in the summary (old bottom table gone)',fl.querySelectorAll('[data-ftlane]').length===5);
  ok('job strips unchanged: 3 rows, gpu-i 15%, sub-heading "Re-runs per job"',fl.querySelectorAll('[data-flane]').length===3&&txt(fl.querySelector('[data-flane]')).includes('15%')&&fl.textContent.includes('Re-runs per job'));
  w.eval("FLAKY_TESTS.push({test:'zz_temp_test',job:'gpu-i',prs:['#4920'],cause:'temp'})");w.renderFlaky();
  w.eval("FLAKY_TESTS.pop()");w.renderFlaky();
  ok('back to 5 rows, no link',fr().length===5&&!fl.querySelector('[data-flmore]'));
  d.querySelector('#date-range').value='7';w.renderFlaky();
  ok('7d: only the two #4920 tests',fr().length===2);
  d.querySelector('#date-range').value='14';w.renderFlaky();
  fr()[1].click();
  ok('moe row click opens the gpu-i job popup',d.querySelector('#wkpop-body').textContent.includes('GPU Tests (gpu-integration)'));
  w.closeWkPop();
  d.querySelector('.fchip[data-hw="tpu"]').click();
  ok('HW=TPU dims GPU test rows to 0.28, keeps the TPU row lit',fr().filter(r=>r.getAttribute('data-ftlane')==='GPU').every(r=>r.getAttribute('style').includes('opacity:0.28'))&&fr().filter(r=>r.getAttribute('data-ftlane')==='TPU').every(r=>r.getAttribute('style').includes('opacity:1')));
  d.querySelector('.fchip[data-hw="all"]').click();
  console.log(`\n${pass} passed, ${fail} failed`);
  process.exit(fail?1:0);
},400);
