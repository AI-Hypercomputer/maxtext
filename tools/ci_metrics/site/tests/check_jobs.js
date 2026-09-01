// Commit-modal job table sorting (2026-08-31 part B).
const fs=require('fs'),path=require('path');const {JSDOM}=require('jsdom');
const html=fs.readFileSync(path.join(__dirname,'..','index.html'),'utf-8');
const dom=new JSDOM(html,{runScripts:'dangerously',pretendToBeVisual:true});
const w=dom.window,d=w.document;
let pass=0,fail=0;
function ok(name,cond){if(cond){pass++;console.log('  ok -',name)}else{fail++;console.log('  FAIL -',name)}}
const txt=n=>n?n.textContent.trim():'';
setTimeout(()=>{
  w.goToCommit('4920');
  const jid='jt-jobs-4920';
  const tb=d.getElementById(jid+'-body');
  ok('job table registered with 18 units',!!tb&&w.eval('JT')[jid].rows.length===18);
  // direct rows only — the step drawers contain nested step/JUnit tables with their own <tr>s
  const rows=()=>Array.from(tb.children);
  const heads=()=>rows().filter(r=>r.querySelector(':scope > td.cat-head')).length;
  const mains=()=>rows().filter(r=>r.querySelector(':scope > td .xarr'));
  const cell=(r,i)=>txt(r.querySelectorAll(':scope > td')[i]);
  // Sorting keeps the category headings, so "first" is per category, not global.
  const firstOfCat=name=>{
    const rs=rows();const i=rs.findIndex(r=>{const h=r.querySelector(':scope > td.cat-head');return h&&txt(h).startsWith(name)});
    if(i<0)return null;
    return rs.slice(i+1).find(r=>r.querySelector(':scope > td .xarr'))||null;
  };
  ok('default = grouped: 5 category headers + 18 job rows + 18 step drawers',heads()===5&&mains().length===18&&rows().length===41);
  ok('default first job = Analyze Code Changes (infra group)',cell(mains()[0],0).includes('Analyze Code Changes'));
  w.jtSort(jid,'total');
  ok('sort by Total keeps all 5 category headers and 18 job rows',heads()===5&&mains().length===18);
  ok('sorted inside the category: TPU group leads with tpu-integration 37.0m',cell(firstOfCat('TPU Tests'),0).includes('tpu-integration')&&cell(firstOfCat('TPU Tests'),6)==='37.0m');
  ok('categories stay in workflow order when sorted',rows().filter(r=>r.querySelector(':scope > td.cat-head')).map(r=>txt(r).split(' (')[0]).join()==='Infrastructure,Build,TPU Tests,GPU Tests,CPU Tests');
  ok('every job row is followed by its own step drawer',mains().every(r=>r.nextElementSibling&&/xrow-4920-\d/.test(r.nextElementSibling.innerHTML)));
  ok('Total header shows ▼',txt(d.querySelector(`th[data-jt="${jid}"][data-jtkey="total"] .jtarr`))==='▼');
  w.jtSort(jid,'total');
  ok('second click: shortest first inside Infrastructure = Gate Parameters 0.2m',cell(firstOfCat('Infrastructure'),0).includes('Gate and Formalize Parameters')&&cell(firstOfCat('Infrastructure'),6)==='0.2m');
  w.jtSort(jid,'q');
  ok('sort by Queue: the TPU group leads with a 14m wait',cell(firstOfCat('TPU Tests'),2)==='14m'&&cell(firstOfCat('TPU Tests'),1)==='TPU');
  w.jtSort(jid,'r');
  ok('sort by Run: the TPU group leads with tpu-integration at 20m',cell(firstOfCat('TPU Tests'),5)==='20m'&&cell(firstOfCat('TPU Tests'),0).includes('tpu-integration'));
  w.jtSort(jid,'');
  ok('Job header restores grouped order',heads()===5&&cell(mains()[0],0).includes('Analyze Code Changes')&&txt(d.querySelector(`th[data-jt="${jid}"][data-jtkey="total"] .jtarr`))==='');
  ok('legend hint sentence present',d.querySelector('#modal').textContent.includes('Sorting keeps the category headings and orders the jobs inside each one')&&d.querySelector('#modal').textContent.includes('Click Queue, Image, Env, Run, or Total to sort the jobs by that time')&&d.querySelector('#modal').textContent.includes('Click the Job header to restore the grouped order.'));
  // per-test JUnit tables inside the drawers still work after a job sort
  w.jtSort(jid,'total');
  const tt=d.getElementById('jt-xrow-4920-5-body');
  ok('JUnit table inside a drawer still present after job sort',!!tt&&tt.querySelectorAll('tr').length===4);
  w.jtSort('jt-xrow-4920-5','dur');
  ok('JUnit table still sortable after job sort',txt(tt.querySelector('tr td'))==='test_llama3_70b_sharded_convergence');
  w.closeModal();
  w.goToCommit('4908');
  ok('another commit gets its own job table id',!!d.getElementById('jt-jobs-4908-body'));
  console.log(`\n${pass} passed, ${fail} failed`);
  process.exit(fail?1:0);
},300);
