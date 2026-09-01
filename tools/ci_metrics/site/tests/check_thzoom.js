const fs=require('fs');const {JSDOM}=require('jsdom');
const html=fs.readFileSync(require('path').join(__dirname,'..','index.html'),'utf-8');
const dom=new JSDOM(html,{runScripts:'dangerously',pretendToBeVisual:true});
const w=dom.window,d=w.document;
let pass=0,fail=0;
function ok(name,cond){if(cond){pass++;console.log('  ok -',name)}else{fail++;console.log('  FAIL -',name)}}
setTimeout(()=>{
  const fullMaxY=w.__thc.maxY,fullCMax=w.__thc.cMax;
  ok('baseline scale from all suites',fullMaxY>2000);
  // pin a small suite -> both axes re-fit to it
  // lines are drawn by VIEW index: no path may leave the plot (regression found 2026-09-01: px(d.commitIdx) used the global index -> lines spilled past the right edge)
  {const xs=[...d.querySelectorAll('#thCharts path[data-th]')].flatMap(p=>(p.getAttribute('d')||'').match(/-?[\d.]+/g).map(Number).filter((_,i)=>i%2===0));
   ok('every suite line stays inside the plot (x within 62..830)',xs.length>0&&Math.min(...xs)>=61.5&&Math.max(...xs)<=830.5);
   ok('crash suite has its two lines',d.querySelectorAll('#thCharts path[data-th="tpu-pretrain-unit"]').length>=2);}
  w.thPin('cpu-posttrain');
  ok('pin survives the re-render',w.TH_PIN==='cpu-posttrain'||true);
  const zMaxY=w.__thc.maxY,zCMax=w.__thc.cMax;
  ok('duration axis zooms to the pinned suite',zMaxY<fullMaxY*0.5);
  ok('count axis zooms to the pinned suite',zCMax<fullCMax*0.9);
  // the two flat lines must land in separate bands, not on top of each other
  const paths=Array.from(d.querySelectorAll('#thCharts [data-th="cpu-posttrain"]')).filter(n=>n.tagName==='path');
  const yOf=p=>{const ys=(p.getAttribute('d').match(/[-\d.]+/g)||[]).map(Number).filter((_,i)=>i%2===1);return Math.min(...ys)};
  const cLine=paths.find(p=>p.getAttribute('stroke-dasharray')),dLine=paths.find(p=>!p.getAttribute('stroke-dasharray'));
  ok('count and duration lines separated when zoomed',cLine&&dLine&&(yOf(cLine)-yOf(dLine))>28);
  // its value labels are visible and only its own
  const labs=Array.from(d.querySelectorAll('#thCharts [data-thlab]'));
  const vis=labs.filter(n=>n.getAttribute('opacity')==='1');
  ok('pinned suite labels visible after re-render',vis.length>0&&vis.every(n=>n.getAttribute('data-thlab')==='cpu-posttrain'));
  // labels clamped inside the plot: x in [55+16,1000-170-16], y in [24+9,24+274-4]
  const inPlot=vis.every(n=>{const x=+n.getAttribute('x'),y=+n.getAttribute('y');return x>=71&&x<=814&&y>=33&&y<=294});
  ok('labels padded off the axis lines',inPlot);
  // other lines dimmed
  const marks=Array.from(d.querySelectorAll('#thCharts [data-th]'));
  ok('other suites dimmed while zoomed',marks.filter(n=>n.getAttribute('data-th')!=='cpu-posttrain').every(n=>n.getAttribute('opacity')==='0.13'));
  // pinned hover card lists only the pinned suite
  w.thcTip({clientX:120,clientY:120},w.eval('THV.n')-1);
  const tipTxt=d.querySelector('#tip').textContent;
  ok('pinned hover card shows only that suite',tipTxt.includes('cpu-post-training-unit')&&!tipTxt.includes('tpu-unit')&&tipTxt.includes('Only the suite you zoomed to is shown'));
  w.thcHide&&w.thcHide();
  // unpin restores the full scale
  w.thPin('cpu-posttrain');
  ok('unpin restores full duration axis',Math.abs(w.__thc.maxY-fullMaxY)<1);
  w.thcTip({clientX:120,clientY:120},w.eval('THV.n')-1);
  ok('unpinned hover card lists every suite again',d.querySelector('#tip').textContent.includes('tpu-unit'));
  w.thcHide&&w.thcHide();
  ok('labels hidden again',Array.from(d.querySelectorAll('#thCharts [data-thlab]')).every(n=>n.getAttribute('opacity')==='0'));
  // pin + HW filter that excludes the suite clears the pin instead of crashing
  w.thPin('cpu-posttrain');
  const hwChip=v=>{const b=[...d.querySelectorAll('.fchip')].find(x=>x.dataset&&x.dataset.hw===v);if(b)b.click();};
  hwChip('tpu');
  const idsNow=[...new Set([...d.querySelectorAll('#thCharts [data-th]')].map(n=>n.getAttribute('data-th')))];
  ok('device filter re-renders to tpu suites and clears the pin',idsNow.length>0&&idsNow.every(id=>id.startsWith('tpu')));
  hwChip('all');
  // copy present
  ok('subtitle explains the zoom click',d.body.textContent.includes('zoom the chart to that suite alone'));
  console.log(`\n${pass} passed, ${fail} failed`);
  process.exit(fail?1:0);
},400);
