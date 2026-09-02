// The machine each job asks for, shown on all four charts. Chart 1 and 3 name the machine
// type per device; chart 2 names the pool on the runner-wait slice; chart 4 names the type on
// the job row and the type plus pool in the cell tooltip; the commit modal has a full column.
const fs=require('fs');const {JSDOM}=require('jsdom');
const dom=new JSDOM(fs.readFileSync(require('path').join(__dirname,'..','index.html'),'utf-8'),{runScripts:'dangerously',pretendToBeVisual:true});
const w=dom.window,d=w.document;
let p=0,f=0;
const ok=(n,c)=>{if(c){p++;console.log('  ok - '+n)}else{f++;console.log('  FAIL - '+n)}};
const T=n=>n?n.textContent.replace(/\s+/g,' '):'';
const TPU='linux-x86-ct6e-180-4tpu',CPU='linux-x86-n2-32',GPU='linux-x86-a2-48-a100-4gpu';

setTimeout(()=>{
  // the data itself
  const labels=w.eval('RUNNER_LABELS');
  ok('every job in JOBS has a machine label',w.eval('JOBS').every(j=>!!labels[j.id]));
  ok('the four real TPU flavors share the 4-chip label',
     ['tpu-pre-u','tpu-pre-i','tpu-post-u','tpu-post-i'].every(k=>labels[k]===TPU));
  ok('all four CPU flavors share linux-x86-n2-32',
     ['cpu-pre-u','cpu-pre-i','cpu-post-u','cpu-post-i'].every(k=>labels[k]===CPU));
  ok('hosted jobs are ubuntu-latest',['analyze','quality','docs','gate'].every(k=>labels[k]==='ubuntu-latest'));
  ok('runner groups are not shown anywhere',!/ml-east5|ml-central1/.test(d.body.innerHTML));

  // chart 1: bar hover
  const c0=w.eval('COMMITS')[0];
  w.showTip(c0,0,w.eval('COMMITS').map(c=>w.wallClock(c,w.eval('HW'))),32,{clientX:200,clientY:200});
  const tip1=T(d.getElementById('tip'));
  ok('chart 1 hover names the TPU machine',tip1.includes(TPU));
  ok('chart 1 hover names the GPU and CPU machines',tip1.includes(GPU)&&tip1.includes(CPU));

  // chart 2: worker rows carry machine + pool, and the wait slice names the pool
  const ent=w.wkEntities(c0).find(e=>e.jobs&&e.jobs.length);
  const rows=w.wkWorkerRows(c0,ent);
  ok('chart 2 worker rows carry a machine',rows.every(r=>r.machine&&r.machine.length>4));
  // The DEFAULT view is the stacked all-suites chart, whose slices carry data-gid. Test that
  // one first: an earlier version of this test switched to a suite and so never covered the
  // view a reader actually lands on.
  ok('chart 2 default (all suites) hover names the machine',(()=>{
    const s=[...d.querySelectorAll('#worker-chart svg [data-gid]')][0];
    if(!s)return false;
    s.dispatchEvent(new w.MouseEvent('mousemove',{clientX:300,clientY:300,bubbles:true}));
    return /linux-x86|ubuntu-latest/.test(T(d.getElementById('tip')));})());
  ok('chart 2 single-suite hover names the machine',(()=>{
    w.setWKSel('tpu-pre-u');                       // per-worker bars only exist once a suite is picked
    const s=[...d.querySelectorAll('#worker-chart svg [data-wk]')][0];
    if(!s)return false;
    s.dispatchEvent(new w.MouseEvent('mousemove',{clientX:300,clientY:300,bubbles:true}));
    const got=/linux-x86-ct6e-180-4tpu/.test(T(d.getElementById('tip')));
    w.setWKSel('all');
    return got;})());

  // chart 3: device tooltip
  const hit=d.querySelector('#devlines svg rect[cursor="pointer"]')||d.querySelector('#devlines svg rect');
  if(hit){hit.dispatchEvent(new w.MouseEvent('mouseenter',{clientX:300,clientY:300,bubbles:true}));}
  const tip3=T(d.getElementById('tip'));
  ok('chart 3 hover names a machine type',tip3.includes(TPU)||tip3.includes(CPU)||tip3.includes(GPU));

  // chart 4: job row + cell tooltip
  const flaky=T(d.getElementById('flaky'));
  ok('chart 4 job rows name the machine',flaky.includes(GPU)||flaky.includes(TPU));
  const cell=d.querySelector('#flaky [data-cell][data-fljob]');
  w.flakyCellTip({clientX:300,clientY:300},cell.getAttribute('data-fljob'),+cell.getAttribute('data-flidx'));
  const tip4=T(d.getElementById('tip'));
  ok('chart 4 cell tooltip names the machine',tip4.includes(TPU)||tip4.includes(GPU));

  // the modal column
  w.goToCommit('4920');
  const modal=T(d.getElementById('modal'));
  ok('the commit modal has a Machine column with real labels',
     modal.includes('MACHINE')||modal.includes('Machine'));
  ok('modal shows every machine type in use',[TPU,CPU,GPU,'ubuntu-latest','linux-x86-n2-16-buildkit'].every(m=>modal.includes(m)));

  // house rule
  const small=[...d.querySelectorAll('[style*="font-size:11px"],[style*="font-size:10px"]')];
  ok('no machine label uses a font under 12px',!small.some(n=>/linux-x86|ubuntu-latest|ml-(east5|central1)/.test(T(n))));

  console.log(`\n${p} passed, ${f} failed`); process.exit(f?1:0);
},500);
