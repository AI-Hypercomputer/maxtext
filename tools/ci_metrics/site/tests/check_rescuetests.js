// Flakiest tests: the mapping from the rescue_tests view onto the page's model.
//
// The published view is empty today - the failed attempts' JUnit files had already expired - so
// nothing else on the page exercises this path. These checks feed adaptViews a hand-built view so
// the two ways it can silently drop a row stay caught: a test whose job id does not match the
// rescue group it belongs to, and a test taken from an attempt that was never re-run at all.
const {JSDOM}=require('jsdom');
const dom=new JSDOM(require('./loadpage.js')(),{runScripts:'dangerously',pretendToBeVisual:true});
const w=dom.window;
let pass=0,fail=0;
const ok=(n,c)=>{c?(pass++,console.log('  ok -',n)):(fail++,console.log('  FAIL -',n))};
setTimeout(()=>{
  const rescues=[
    // a rescued worker job: flavor and worker are both present
    {run_id:1,pr:9001,job_name:'CPU Pretrain Tests (cpu-unit) / Execute Tests (1) / cpu-unit',
     lane:'CPU',flavor:'cpu-unit',worker:1,rescued:true,failed_attempt:1,
     final_conclusion:'success',failed_conclusion:'failure',wasted_seconds:120},
    // a rescued job the collector cannot tie to a flavor, so only its name identifies it
    {run_id:2,pr:9002,job_name:'TPU Pathways Unit Tests (1) / tpu-pathways-unit',
     lane:'TPU',flavor:null,worker:null,rescued:true,failed_attempt:1,
     final_conclusion:'success',failed_conclusion:'failure',wasted_seconds:60},
    // failed and never re-run: its tests are not flaky, they just failed
    {run_id:3,pr:9003,job_name:'CPU Pretrain Tests (cpu-unit) / Execute Tests (2) / cpu-unit',
     lane:'CPU',flavor:'cpu-unit',worker:2,rescued:false,failed_attempt:1,
     final_conclusion:'failure',failed_conclusion:'failure',wasted_seconds:300},
  ];
  const rescue_tests=[
    {run_id:1,pr:9001,job_name:rescues[0].job_name,failed_attempt:1,flavor:'cpu-unit',worker:1,
     classname:'tests.cpu',name:'test_rescued_worker',status:'failed',failure_message:'boom'},
    {run_id:2,pr:9002,job_name:rescues[1].job_name,failed_attempt:1,flavor:null,worker:null,
     classname:'tests.pw',name:'test_rescued_noflavor',status:'failed',failure_message:'boom'},
    {run_id:3,pr:9003,job_name:rescues[2].job_name,failed_attempt:1,flavor:'cpu-unit',worker:2,
     classname:'tests.cpu',name:'test_never_rerun',status:'failed',failure_message:'boom'},
  ];
  const m=w.adaptViews({runs:{runs:[],jobs:[]},suites:{suites:[]},queue:{queue:[]},
                        flaky:{rescues,rescue_tests}});
  const names=m.flaky.map(f=>f.test).sort();
  ok('a test from an attempt that was never re-run is left out',!names.includes('test_never_rerun'));
  ok('both rescued tests are kept',names.join()==='test_rescued_noflavor,test_rescued_worker');

  const ids=new Set(m.rescues.map(r=>r.id));
  ok('every test names a job the card can find',m.flaky.every(f=>ids.has(f.job)));
  ok('a flavoured test keeps the flavor#worker id',
     m.flaky.find(f=>f.test==='test_rescued_worker').job==='cpu-unit#1');
  ok('a flavourless test falls back to the job name, like its rescue row does',(()=>{
     const t=m.flaky.find(f=>f.test==='test_rescued_noflavor');
     const g=m.rescues.find(r=>r.events.some(e=>e.pr==='#9002'));
     return !!g&&t.job===g.id&&t.job!==''})());

  // and the split the rescues themselves get, on the same input
  const cpu2=m.rescues.find(r=>r.id==='cpu-unit#2');
  ok('the never-re-run row lands in fails, not events',
     !!cpu2&&cpu2.events.length===0&&cpu2.fails.join()==='#9003');
  ok('the rescued rows land in events',
     m.rescues.find(r=>r.id==='cpu-unit#1').events.length===1);

  console.log(`\n${pass} passed, ${fail} failed`);
  process.exit(fail?1:0);
},400);
