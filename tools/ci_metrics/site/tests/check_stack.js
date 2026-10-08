// Worker time per merged pull request (rewritten 2026-09-03).
//
// The chart answers two different questions and draws them differently, which is the whole
// point of the rewrite: things that run side by side are drawn side by side under "Wall clock"
// and stacked under "Machine minutes", so the number above a group is always the number the
// eye reads off the axis. These checks pin that down, plus the device-lane grouping that
// replaced one slice per suite, and the per-worker view behind the dropdown.
const fs=require('fs'),{JSDOM}=require('jsdom');
const html=require('./loadpage.js')();
const dom=new JSDOM('<!doctype html><html><head></head><body>'+html+'</body></html>',{runScripts:'dangerously',pretendToBeVisual:true});
let pass=0,fail=0;
const ok=(name,cond)=>{cond?(pass++,console.log('PASS',name)):(fail++,console.log('FAIL',name))};
setTimeout(()=>{
  const w=dom.window,d=w.document;
  const chart=()=>d.querySelector('#worker-chart');
  const svg=()=>d.querySelector('#worker-chart svg');
  const n=w.getWindows().current.length;  // PRs in the selected range (20 at the default 14 days)
  const meta=()=>d.querySelector('#wk-meta');
  const shapes=()=>[...svg().querySelectorAll('rect,path')];
  // x of a bar, whether it was drawn as a rect or as a rounded-top path
  const xOf=e=>e.tagName==='rect'?+e.getAttribute('x'):+(/^M ([\d.]+)/.exec(e.getAttribute('d'))||[0,0])[1];
  const columns=()=>new Set(shapes().map(e=>xOf(e).toFixed(2))).size;
  const fillCount=f=>shapes().filter(e=>e.getAttribute('fill')===f).length;
  // group totals printed above the bars, and the top gridline they are measured against
  const axisTop=()=>Math.max(...[...svg().querySelectorAll('text')].filter(t=>t.getAttribute('text-anchor')==='end').map(t=>parseFloat(t.textContent)));
  const groupLabels=()=>[...svg().querySelectorAll('text')].filter(t=>t.getAttribute('text-anchor')==='middle'&&/^\d+(\.\d+)?m$/.test(t.textContent)).map(t=>parseFloat(t.textContent));
  const set=(sel,meas,ph)=>{w.setWKSel(sel);w.setWKMeasure(meas);w.setWKPhase(ph)};

  // ---------- the view a reader lands on ----------
  ok('opens on all suites, wall clock, tests only',w.eval('WK_SEL')==='all'&&w.eval('WK_MEASURE')==='wall'&&w.eval('WK_PHASE')==='tests');
  ok('both switches are offered',d.querySelector('#wk-chips').textContent.includes('Measure:')&&d.querySelector('#wk-chips').textContent.includes('Wall clock')&&d.querySelector('#wk-chips').textContent.includes('Machine minutes')&&d.querySelector('#wk-chips').textContent.includes('Time shown:')&&d.querySelector('#wk-chips').textContent.includes('Tests only'));

  // ---------- all suites: one bar per device lane, not one slice per suite ----------
  const lanes=()=>{const o={};[...svg().querySelectorAll('[data-gid]')].forEach(e=>{const k=e.getAttribute('data-gid');o[k]=(o[k]||0)+1});return o};
  ok('lanes are the grouping: exactly tpu, gpu, cpu, build, checks',Object.keys(lanes()).sort().join()==='build,cpu,gpu,infra,tpu');
  ok('one bar per lane per PR, at most 5n and far below the old 21 slices',shapes().length<=5*n&&shapes().length>=4*n);
  ok('tests only = one solid capped bar per lane',svg().querySelectorAll('rect').length===0&&svg().querySelectorAll('path').length===shapes().length);
  ok('wall clock puts the lanes side by side, so every bar has its own x',columns()===shapes().length);
  const wallLabels=groupLabels();
  ok('no group total is drawn above the top gridline',wallLabels.every(v=>v<=axisTop()));

  // cross-highlight now follows a device lane across every pull request
  const tpu=[...svg().querySelectorAll('[data-gid="tpu"]')][0];
  tpu.dispatchEvent(new w.MouseEvent('mousemove',{bubbles:true,clientX:200,clientY:200}));
  ok('hover dims the other lanes',[...svg().querySelectorAll('[data-gid]')].filter(e=>e.getAttribute('opacity')==='0.22').length===shapes().length-lanes().tpu);
  ok('lane tooltip names the lane, its job count and both totals',(()=>{const t=d.getElementById('tip').innerHTML;
    return /TPU lane/.test(t)&&/job/.test(t)&&t.includes('This lane')&&t.includes('Slowest lane in this run')})());
  tpu.dispatchEvent(new w.MouseEvent('mouseleave',{bubbles:true}));
  ok('leave restores opacity',[...svg().querySelectorAll('[data-gid]')].every(e=>e.getAttribute('opacity')===e.getAttribute('data-op')));

  // ---------- the phase switch ----------
  set('all','wall','all');
  ok('All time adds runner wait and setup slices',fillCount('var(--q)')>0&&fillCount('var(--bld)')>0);
  ok('adding phases can only make a run longer',groupLabels().every((v,i)=>v>=wallLabels[i]));
  set('all','wall','tests');
  ok('Tests only drops them again',fillCount('var(--q)')===0&&fillCount('var(--bld)')===0);

  // ---------- the measure switch ----------
  set('all','machine','tests');
  ok('machine minutes stack the lanes into one column per PR',columns()===n);
  ok('machine minutes still label every bar under the top gridline',groupLabels().every(v=>v<=axisTop()));
  const machLabels=groupLabels();
  ok('machine minutes are never below wall clock',machLabels.length===wallLabels.length&&machLabels.every((v,i)=>v>=wallLabels[i]));
  ok('meta explains that machine minutes add up',meta().textContent.includes('Machine minutes add up, so the lanes are stacked'));
  set('all','wall','tests');
  ok('meta explains that wall clock is the slowest lane',meta().textContent.includes('the number above a group is the slowest lane'));

  // ---------- one suite: one bar per worker ----------
  set('cpu-pre-u','wall','tests');
  ok('cpu-pre 4 workers side by side, one capped bar each',svg().querySelectorAll('path').length===4*n&&svg().querySelectorAll('rect').length===0&&columns()===4*n);
  ok('four worker shades',new Set(shapes().map(e=>e.getAttribute('fill'))).size===4);
  ok('wall-clock label is the slowest worker, never the sum',groupLabels().every(v=>v<=axisTop()));
  ok('meta says the workers run at the same time',meta().textContent.includes('The workers run at the same time, so they are drawn side by side'));
  ok('meta split disclosure',meta().textContent.includes('estimated from the suite total'));
  ok('legend names the worker shades',meta().textContent.includes('Worker tests')&&meta().textContent.includes('shaded dark to light = worker 1 to 4'));
  ok('back chip present',d.querySelector('#wk-chips').textContent.includes('Back to all suites'));
  const texts=[...svg().querySelectorAll('text')].map(t=>t.textContent);
  ok('hash x labels',texts.filter(t=>t.startsWith('#')).length===n);

  set('cpu-pre-u','machine','tests');
  ok('machine minutes stack the four workers into one column',svg().querySelectorAll('path').length===n&&svg().querySelectorAll('rect').length===3*n&&columns()===n);
  ok('the stacked total still fits under the top gridline',groupLabels().every(v=>v<=axisTop()));
  ok('meta says machine minutes add up across the workers',meta().textContent.includes('the number above a bar is all 4 of them together'));

  set('cpu-pre-u','wall','all');
  ok('All time shows each worker its own wait and setup',fillCount('var(--bld)')>0);

  // hover a worker slice -> tooltip full breakdown
  set('cpu-pre-u','wall','tests');
  const slice=shapes()[5];
  slice.dispatchEvent(new w.MouseEvent('mousemove',{bubbles:true,clientX:300,clientY:200}));
  const tip=d.querySelector('#tip');
  ok('tooltip shows the worker and the suite roll-up',tip.innerHTML.includes('This worker')&&tip.innerHTML.includes('Slowest of the 4 workers'));
  ok('tooltip click copy',tip.innerHTML.includes("this suite's worker details"));
  ok('hover follows the worker across PRs',[...svg().querySelectorAll('[data-wk]')].some(e=>e.getAttribute('opacity')==='0.3'));
  slice.dispatchEvent(new w.MouseEvent('mouseleave',{bubbles:true}));

  // click a bar -> suite-scoped popup, not the full commit modal
  slice.dispatchEvent(new w.MouseEvent('click',{bubbles:true}));
  const pop=d.querySelector('#wkpop'),popB=d.querySelector('#wkpop-body');
  ok('suite popup opens on click',pop.style.display==='flex');
  ok('popup scoped to suite',popB.innerHTML.includes('CPU Pretrain Tests (cpu-unit')&&!popB.innerHTML.includes('GPU Unit'));
  ok('popup one row per worker',(popB.innerHTML.match(/Execute Tests \(/g)||[]).length===4);
  ok('popup states machine time and wall clock, so it cannot disagree with either mode',popB.innerHTML.includes('of machine time')&&popB.innerHTML.includes('the suite added'));
  ok('popup split disclosure',popB.innerHTML.includes('estimated from the suite total'));
  ok('popup escape hatches',popB.innerHTML.includes('Every suite in this run')&&popB.innerHTML.includes('Open full commit details'));
  w.closeWkPop();

  // Pathways: 2 real jobs, so 2 bars per PR
  set('tpu-pw-u','wall','tests');
  ok('pathways draws 2 workers per PR',svg().querySelectorAll('path').length===2*n-2&&columns()===2*n-2);  // #4940 never got a runner, so it ran no tests
  ok('pathways no split disclosure',!meta().textContent.includes('estimated from the suite total'));
  ok('pathways shade note 2 workers',meta().textContent.includes('worker 1 to 2'));

  // a suite that really has one worker upstream
  set('tpu-pre-i','wall','tests');
  ok('single-worker suite draws one bar per PR',svg().querySelectorAll('path').length===n-1&&columns()===n-1);  // #4940 never got a runner, so it ran no tests

  // the all-lanes click path: this popup broke silently once the chart stopped colouring by
  // suite, because nothing here had ever clicked a bar in the default view
  set('all','wall','tests');
  shapes()[0].dispatchEvent(new w.MouseEvent('click',{bubbles:true}));
  const allPop=d.querySelector('#wkpop-body');
  ok('clicking a lane opens the every-suite popup',d.querySelector('#wkpop').style.display==='flex'&&allPop.innerHTML.includes('Every worker suite in this run'));
  ok('every-suite popup rows are coloured, one shade per suite',(()=>{const sw=[...allPop.querySelectorAll('span[style*="border-radius:3px"]')].map(e=>/background:(#[0-9a-f]{6})/i.exec(e.getAttribute('style'))).filter(Boolean);
    return sw.length>=5&&new Set(sw.map(m=>m[1])).size>=5})());
  ok('every-suite popup says which measure its numbers are',allPop.innerHTML.includes("the suite's machine minutes")&&allPop.innerHTML.includes('larger than the bar you clicked'));
  w.closeWkPop();

  // the legend may only name colours the chart is actually drawing
  set('cpu-pre-u','machine','all');
  ok('machine minutes stack whole workers, so the legend drops wait and setup',!meta().textContent.includes('Runner wait')&&meta().textContent.includes('Worker time'));
  set('all','machine','all');
  ok('all-lanes machine legend drops wait and setup too',!meta().textContent.includes('Runner wait'));
  set('all','wall','all');
  ok('wall clock draws the phases, so the legend names them',meta().textContent.includes('Runner wait')&&meta().textContent.includes('Setup'));

  // the red cross over a bar has a key in both views
  ok('all-lanes legend explains the red cross',meta().textContent.includes('this run failed'));
  set('cpu-pre-u','wall','tests');
  ok('single-suite legend explains the red cross',meta().textContent.includes('a worker in this suite failed on that run'));

  set('all','wall','tests');
  ok('back to all suites works',Object.keys(lanes()).length===5);

  console.log(pass+'/'+(pass+fail)+' passed');
  process.exit(fail?1:0);
},400);
