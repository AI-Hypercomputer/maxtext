const fs=require('fs'),{JSDOM}=require('jsdom');
const html=fs.readFileSync(require('path').join(__dirname,'..','index.html'),'utf-8');
const dom=new JSDOM('<!doctype html><html><head></head><body>'+html+'</body></html>',{runScripts:'dangerously',pretendToBeVisual:true});
let pass=0,fail=0;
const ok=(name,cond)=>{cond?(pass++,console.log('PASS',name)):(fail++,console.log('FAIL',name))};
setTimeout(()=>{
  const w=dom.window,d=w.document;
  const chart=()=>d.querySelector('#worker-chart');
  const n=w.getWindows().current.length;  // PRs in the selected range (20 at the default 14 days)
  const meta=()=>d.querySelector('#wk-meta');

  // all-suites baseline still works (top slice of each column is now a rounded path)
  const allShapes=chart().querySelectorAll('rect,path').length;
  ok('all-suites slices >200',allShapes>200);
  ok('all-suites one rounded cap per PR in range',chart().querySelectorAll('path').length===n&&n===20);

  // cross-highlight: hover one slice dims other suites
  const s0=chart().querySelector('[data-gid]');
  s0.dispatchEvent(new w.MouseEvent('mousemove',{bubbles:true,clientX:200,clientY:200}));
  const dimmed=[...chart().querySelectorAll('[data-gid]')].filter(n=>n.getAttribute('opacity')==='0.22').length;
  ok('hover dims other suites',dimmed>150);
  s0.dispatchEvent(new w.MouseEvent('mouseleave',{bubbles:true}));
  ok('leave restores opacity',[...chart().querySelectorAll('[data-gid]')].every(n=>n.getAttribute('opacity')===n.getAttribute('data-op')));

  // CPU Pretrain single view: stacked, no line
  w.setWKSel('cpu-pre-u');
  const rects=chart().querySelectorAll('rect').length,paths=chart().querySelectorAll('path').length;
  ok('cpu-pre 4 workers: 4n rects + 4n caps',rects===4*n&&paths===4*n);
  ok('cpu-pre no circles (line gone)',chart().querySelectorAll('circle').length===0);
  ok('meta says one bar per worker',meta().textContent.toLowerCase().includes('one bar per worker under each pull request'));
  ok('meta split disclosure',meta().textContent.includes('split estimated'));
  ok('legend wait+setup+tests',meta().textContent.includes('Runner wait')&&meta().textContent.includes('Setup')&&meta().textContent.includes('Worker tests'));
  ok('legend worker shade note',meta().textContent.includes('tests shaded dark to light = worker 1 to 4'));
  const texts=[...chart().querySelectorAll('text')].map(t=>t.textContent);
  ok('per-column totals labeled',texts.filter(t=>/^\d+(\.\d+)?m$/.test(t)).length>=n);
  ok('hash x labels',texts.filter(t=>t.startsWith('#')).length===n);
  ok('back chip present',d.querySelector('#wk-chips').textContent.includes('Back to all suites'));

  // hover a worker slice → tooltip full breakdown
  const slice=chart().querySelectorAll('rect')[5]; // some slice
  slice.dispatchEvent(new w.MouseEvent('mousemove',{bubbles:true,clientX:300,clientY:200}));
  const tip=d.querySelector('#tip');
  ok('tooltip shows breakdown',tip.innerHTML.includes('Runner wait')&&tip.innerHTML.includes('This worker'));
  ok('tooltip click copy',tip.innerHTML.includes("this suite's worker details"));
  // worker-follow highlight active while hovering
  ok('hover follows worker across PRs',[...chart().querySelectorAll('[data-wk]')].some(n=>n.getAttribute('opacity')==='0.3'));
  slice.dispatchEvent(new w.MouseEvent('mouseleave',{bubbles:true}));

  // click a bar -> suite-scoped popup, not the full commit modal
  slice.dispatchEvent(new w.MouseEvent('click',{bubbles:true}));
  const pop=d.querySelector('#wkpop'),popB=d.querySelector('#wkpop-body');
  ok('suite popup opens on click',pop.style.display==='flex');
  ok('popup scoped to suite',popB.innerHTML.includes('CPU Pretrain Tests (cpu-unit')&&!popB.innerHTML.includes('GPU Unit'));
  ok('popup one row per worker',(popB.innerHTML.match(/Execute Tests \(/g)||[]).length===4);
  ok('popup split disclosure',popB.innerHTML.includes('estimated from the suite total'));
  ok('popup escape hatches',popB.innerHTML.includes('Every suite in this run')&&popB.innerHTML.includes('Open full commit details'));
  w.closeWkPop();

  // Pathways (2 real jobs): 13 x (2+2) = 52
  w.setWKSel('tpu-pw-u');
  const pwRects=[...chart().querySelectorAll('rect')];
  const pwPaths=[...chart().querySelectorAll('path')];
  ok('pathways 2 workers: 4n-4 rects + 2n caps (#4940 never got a runner: wait only, no setup slice) (got '+pwRects.length+'+'+pwPaths.length+')',pwRects.length===4*n-4&&pwPaths.length===2*n);
  const pwAll={};[...pwRects,...pwPaths].forEach(r=>{const f=r.getAttribute('fill');pwAll[f]=(pwAll[f]||0)+1});
  ok('pathways per-bar phases: 2n wait + 2(n-1) setup (#4940: no setup) + 2(n-1) test shades',pwAll['var(--q)']===2*n&&pwAll['var(--bld)']===2*n-2&&pwAll['#23548e']===n-1&&pwAll['#4eb9ff']===n-1);
  ok('pathways no split disclosure',!meta().textContent.includes('split estimated'));
  ok('pathways shade note 2 workers',meta().textContent.includes('worker 1 to 2'));

  // single-worker suite (tpu-integration really has 1 worker upstream)
  w.setWKSel('tpu-pre-i');
  const r3=chart().querySelectorAll('rect').length,p3=chart().querySelectorAll('path').length;
  ok('single-worker suite = 2n-2 rects + n caps (got '+r3+'+'+p3+')',r3===2*n-2&&p3===n);

  // tpu-unit now matches upstream: 2 workers -> two bars per PR
  w.setWKSel('tpu-pre-u');
  const r4=chart().querySelectorAll('rect').length,p4=chart().querySelectorAll('path').length;
  ok('tpu-unit shows 2 workers per PR (got '+r4+'+'+p4+')',r4===4*n-4&&p4===2*n);
  ok('tpu-unit shade note 2 workers',meta().textContent.includes('worker 1 to 2'));

  // back to all
  w.setWKSel('all');
  ok('back to all works',chart().querySelectorAll('rect,path').length>200);

  console.log(pass+'/'+(pass+fail)+' passed');
  process.exit(fail?1:0);
},400);
