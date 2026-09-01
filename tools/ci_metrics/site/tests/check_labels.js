// Direct labels and card copy: the "This chart shows ..." purpose sentence on every card, the value
// labels the suite chart reveals on hover or pin, the hover card's grid and headers, and the axis
// labels of the device and queue charts. Grew out of the 2026-08-27 review batch.
const fs=require('fs');const {JSDOM}=require('jsdom');
const html=fs.readFileSync(require('path').join(__dirname,'..','index.html'),'utf-8');
const dom=new JSDOM(html,{runScripts:'dangerously',pretendToBeVisual:true});
const w=dom.window,d=w.document;
let pass=0,fail=0;
function ok(name,cond){if(cond){pass++;console.log('  ok -',name)}else{fail++;console.log('  FAIL -',name)}}
setTimeout(()=>{
  const body=d.body.textContent;
  // Purpose sentences
  ok('worker purpose',body.includes("This chart shows where each merged pull request's CI time went, worker suite by worker suite"));
  ok('timeline purpose',body.includes('This chart shows how long CI took on every merged pull request on main'));
  ok('devlines purpose',body.includes("This chart shows how many minutes the TPU, GPU and CPU test jobs spent running their tests on each merged pull request"));
  ok('rerun purpose',body.includes('This chart shows which jobs are flaky'));
  ok('th header purpose',d.getElementById('guide-th').textContent.includes('This section has three parts'));
  ok('combined purpose',body.includes("This chart shows every suite's speed and size in one place"));
  ok('indexed purpose',body.includes('This chart shows whether test count, test duration, and worker count move together'));
  ok('pr list purpose',body.includes('This list shows the merged pull requests inside the selected date range'));
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
  const dlDates=dl?Array.from(dl.querySelectorAll('text')).filter(t=>/^\w{3} \d{2}$/.test(t.textContent)).length:0,dlTimes=dl?Array.from(dl.querySelectorAll('text')).filter(t=>/^\d{2}:\d{2}$/.test(t.textContent)).length:0;
  ok('devlines: one linked hash + date per window commit, no times',dlLabels.length===winN&&winN>0&&dlDates===winN&&dlTimes===0);
  ok('devlines: labels 12px',dlLabels.every(t=>t.getAttribute('font-size')==='12'));
  // Starv: 3 pool charts, each labeling every window commit
  w.showPage&&w.showPage('details');
  const svgs=Array.from(d.querySelectorAll('#starv svg'));
  const starvCounts=svgs.map(s=>Array.from(s.querySelectorAll('text')).filter(t=>/^#\d/.test(t.textContent)&&(t.getAttribute('transform')||'').includes('rotate(-55')).length);
  ok('starv: every pool chart labels every commit',starvCounts.length===3&&starvCounts.every(n=>n===winN));
  w.showPage&&w.showPage('main');
  // Indexed triple chart: 30 rotated labels, no date row
  const thc=d.querySelector('#thCharts');
  const idxSvgTexts=Array.from(thc.querySelectorAll('svg text'));
  const rot12=idxSvgTexts.filter(t=>/^#\d/.test(t.textContent)&&(t.getAttribute('transform')||'').includes('rotate(-55')&&t.getAttribute('font-size')==='12');
  ok('TH section: rotated 12px hashes on both charts (2 x PRs in range)',rot12.length===2*w.flakyCalc().n&&w.flakyCalc().n===20);
  // TH value labels hidden by default
  const labs=Array.from(thc.querySelectorAll('[data-thlab]'));
  ok('value labels exist and start hidden',labs.length>100&&labs.every(n=>n.getAttribute('opacity')==='0'));
  // Isolation shows one suite's labels
  w.thIso('cpu-pretrain');
  const vis=labs.filter(n=>n.getAttribute('opacity')==='1');
  ok('thIso shows only that suite labels',vis.length>0&&vis.every(n=>n.getAttribute('data-thlab')==='cpu-pretrain'));
  const durLabs=vis.filter(n=>/m$/.test(n.textContent));
  const cntLabs=vis.filter(n=>!/m$/.test(n.textContent));
  ok('duration labels on every point in range',durLabs.length===w.flakyCalc().n);
  ok('count labels present',cntLabs.length>0);
  w.thIso(null);
  ok('thIso(null) hides labels again',labs.every(n=>n.getAttribute('opacity')==='0'));
  // Pin also shows labels (pin re-renders the chart now, so re-query the nodes)
  w.thPin('cpu-pretrain');
  const labs2=Array.from(d.querySelectorAll('#thCharts [data-thlab]'));
  ok('pin keeps labels visible',labs2.filter(n=>n.getAttribute('opacity')==='1').every(n=>n.getAttribute('data-thlab')==='cpu-pretrain')&&labs2.some(n=>n.getAttribute('opacity')==='1'));
  w.thPin('cpu-pretrain');
  // Hover card grouped by device with aligned columns
  w.thcTip({clientX:300,clientY:300},10);
  const tip=d.querySelector('#tip');
  ok('hover card has device group headers',tip.textContent.includes('TPU')&&tip.textContent.includes('GPU')&&tip.textContent.includes('CPU'));
  ok('hover card has column headers',tip.textContent.includes('Duration')&&tip.textContent.includes('Tests')&&tip.textContent.includes('Per test'));
  ok('hover card uses a grid',tip.innerHTML.includes('display:grid'));
  w.thcHide();
  console.log(`\n${pass} passed, ${fail} failed`);
  process.exit(fail?1:0);
},400);
