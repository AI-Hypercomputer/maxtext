const fs=require('fs');const {JSDOM}=require('jsdom');
const html=fs.readFileSync(require('path').join(__dirname,'..','index.html'),'utf-8');
const dom=new JSDOM(html,{runScripts:'dangerously',pretendToBeVisual:true});
const w=dom.window,d=w.document;
let pass=0,fail=0;
function ok(name,cond){if(cond){pass++;console.log('  ok -',name)}else{fail++;console.log('  FAIL -',name)}}
function modalText(){return d.querySelector('#modal').textContent}
function splitSums(){
  // parse "tests added/removed +0.0m , workers +0.0m , test speed +1.4m = +1.4m total"
  const t=modalText();
  const g=re=>{const m=t.match(re);if(!m)return null;const v=parseFloat(m[1].replace('−','-'));return m[1].startsWith('−')?-Math.abs(parseFloat(m[1].slice(1))):v};
  const a=g(/tests added\/removed ([+−][\d.]+)m/),b=g(/workers ([+−][\d.]+)m/),c=g(/(?:slower|faster) running ([+−][\d.]+)m/),tot=g(/= ([+−][\d.]+)m total/);
  if([a,b,c,tot].some(v=>v===null))return null;
  return {a,b,c,tot,err:Math.abs(a+b+c-tot)};
}
setTimeout(()=>{
  // 1. crash commit: count collapse label, crit tone
  w.openTHDetail('tpu-pretrain-unit',13);
  ok('pattern box present in modal',!!d.querySelector('#modal [data-thpattern]'));
  ok('crash commit reads as count fall with expected time',modalText().includes('would take about')&&modalText().includes('tests that ran averaged slower per test'));
  ok('crash label carries both counts',/Test count fell [\d,]+ → [\d,]+/.test(modalText()));
  const s1=splitSums();
  ok('split parts sum to the total (crash)',s1&&s1.err<=0.35);
  ok('workers line shows 2 unchanged',modalText().includes('Workers: 2 (unchanged)'));
  // 2. healthy mid-history commit: steady label
  w.openTHDetail('tpu-pretrain-int',12);
  ok('healthy commit reads as no significant change',modalText().includes('No significant change'));
  const s2=splitSums();
  ok('split parts sum to the total (healthy)',s2&&s2.err<=0.35);
  // 3. trimmed drop commit (100 -> 66 at idx 15): count-fall label, non-crash wording still factual
  w.openTHDetail('tpu-posttrain-unit',15);
  ok('trim drop reads as count fall matching old speed',modalText().includes('would take about')&&modalText().includes('matches that'));
  ok('trim label shows 100 and 66',modalText().includes('100')&&modalText().includes('66'));
  // 4. first reported run of the NEW suite: no-previous label, no split line
  let first=-1;
  for(let i=0;i<30&&first<0;i++){d.querySelector('#modal').innerHTML='';w.openTHDetail('decoupled',i);if(modalText().trim())first=i;}
  ok('new suite has a first reported commit',first>=0);
  ok('first run label on the new suite',modalText().includes('no previous commit to compare against'));
  ok('first run has no split line',!/tests added\/removed/.test(modalText()));
  // 5. every pattern box carries the fixed-threshold disclosure
  ok('disclosure sentence present',modalText().includes('fixed thresholds; no judgment is applied'));
  // 6. no advice verbs anywhere in the box
  w.openTHDetail('tpu-pretrain-unit',13);
  const box=d.querySelector('#modal [data-thpattern]').textContent;
  ok('no recommendation copy',!/(should|consider|recommend|fix this|please)/i.test(box));
  // ---- indexed-chart pattern reads ----
  w.closeModal&&w.closeModal();
  ok('pattern reads block on the indexed chart',!!d.querySelector('[data-thipat]'));
  const pat=()=>d.querySelector('[data-thipat]').textContent;
  ok('nested suite adds no worker: no worker-change row (decoupled runs inside cpu-unit worker 1)',!/Workers went/.test(pat())&&!(w.__thi.ev||[]).some(e=>e.kind==='worker'));
  ok('count-fall event states expected time',pat().includes('would take about'));
  ok('rows disclose fixed thresholds',pat().includes('fixed thresholds; no judgment is applied'));
  ok('no advice verbs in pattern reads',!/(should|consider|recommend|please)/i.test(pat()));
  const evi=k2=>w.__thi.ev.findIndex(e2=>e2.kind===k2);
  const rowTexts=[...d.querySelectorAll('[data-thipat] [data-thipatn]')].map(n=>n.parentElement.textContent);
  ok('pattern rows run newest first',rowTexts.length>1&&rowTexts[0].includes('#4930')&&rowTexts[rowTexts.length-1].includes('#4900'));
  w.thiPatPop(evi('fall'));
  ok('row click opens the decomposition popup',modalText().includes('How each suite contributed'));
  ok('popup restates the clicked row',modalText().includes('Test count fell'));
  ok('popup names the biggest mover',modalText().includes('tpu-unit')&&modalText().includes('2,400 → 79'));
  ok('popup totals match the row aggregate',modalText().includes('5,385 → 816'));
  w.closeModal&&w.closeModal();
  const hwChip=v=>{const b=[...d.querySelectorAll('.fchip')].find(x=>x.dataset&&x.dataset.hw===v);if(b)b.click();return !!b};
  ok('cpu hardware chip exists',hwChip('cpu'));
  ok('pattern reads follow the Hardware filter',!!d.querySelector('[data-thipat]')&&!/went 14 → 15/.test(pat()));
  hwChip('all');
  ok('all view still has no worker-change row',!/Workers went/.test(pat()));
  ok('plain tooltip copy present',html.includes('Values at this commit, and how each compares with the first commit in this date range'));
  const mks=[...d.querySelectorAll('[data-thipatmk]')],bads=[...d.querySelectorAll('[data-thipatn]')];
  ok('chart markers exist and match the row badges',mks.length>0&&mks.length===bads.length);
  ok('marker numbers align with badge numbers',mks.map(n=>n.getAttribute('data-thipatmk')).join()===bads.map(n=>n.getAttribute('data-thipatn')).join());
  w.thiPatHl(1,1);
  ok('row hover emphasizes its chart marker',d.querySelector('[data-thipatmk="1"]').getAttribute('opacity')==='1');
  w.thiPatHl(1,0);
  // ---- validation-fix asserts ----
  ok('fmtDuration floors minutes',w.fmtDuration(1532)==='25m 32s'&&w.fmtDuration(119.6)==='2m 0s'&&w.fmtDuration(3599)==='59m 59s');
  ok('sign convention stated in the split',modalText().includes('adds minutes to the run')||(w.openTHDetail('tpu-pretrain-unit',13),modalText().includes('adds minutes to the run')));
  ok('modal tile names the baseline value in raw time',/ (more|less) than the average of the first \d+ (runs in this range|stored runs) \([\dhms ]+\)/.test(modalText())&&!/\d%/.test(modalText().split('Pattern read')[0]));
  ok('modal tile says takes x as long, not speed',modalText().includes('as long as the')||modalText().includes('time per test of'));
  ok('scope sentence on the indexed card',d.body.textContent.includes('totals across suites, not one suite'));
  w.closeModal&&w.closeModal();
  // back-button chain: popup -> suite detail -> back to popup
  w.thiPatPop(evi('fall'));
  w.thiPatDetail(evi('fall'),'tpu-pretrain-unit',13);
  ok('suite detail from popup shows the back button',modalText().includes('Back to pattern event'));
  w.thiPatPop(w.__thiBack);
  ok('back returns to the decomposition popup',modalText().includes('How each suite contributed'));
  w.closeModal&&w.closeModal();
  w.openTHDetail('tpu-pretrain-unit',13);
  ok('direct detail open has no back button',!modalText().includes('Back to pattern event'));
  w.closeModal&&w.closeModal();
  ok('worker band never counts the nested suite',w.__thi.data.every(x=>x.wrk===17));
  const hwc=v=>{const b2=[...d.querySelectorAll('.fchip')].find(x=>x.dataset&&x.dataset.hw===v);if(b2)b2.click();};
  hwc('cpu');
  ok('speed-only rows name the direction, or the alternating runs collapse into one swing row',/swung between/.test(pat())||(pat().includes('averaged slower per test')&&pat().includes('averaged faster per test')));
  hwc('all');
  // ---- matrix + expected-line asserts ----
  const expPaths=[...d.querySelectorAll('svg path')].filter(p=>p.getAttribute('stroke')==='#8B5CF6'&&p.getAttribute('stroke-dasharray')==='2,7');
  const expEls=[...d.querySelectorAll('body *:not(script)')].filter(n=>n.children.length===0&&n.textContent.includes('Expected duration')&&!n.closest('#tip'));
  ok('expected-duration line removed from the chart',expPaths.length===0&&expEls.length===0);
  w.thiTip({clientX:100,clientY:100},w.eval('THV.n')-1);
  ok('hover card has no expected duration either',!d.querySelector('#tip').textContent.includes('Expected duration')&&!d.querySelector('#tip').textContent.includes('real speed change'));
  w.thiHide&&w.thiHide();
  ok('standing summary line present, raw before -> after',/Since the start of this date range \(#\d+, \w+ \d+\) to #\d+: total test time went [\d.]+h → [\d.]+h \([^)]+\), test count [\d,]+ → [\d,]+ \([\d,]+ fewer tests\), workers 17 → 17 \(no change\)/.test(d.body.textContent));
  ok('summary states time per test',/Time per test is now [\d.]+s against [\d.]+s at the start/.test(d.body.textContent));
  // ---- raw-unit chart: no percentages anywhere in the card (2026-08-31 part F) ----
  const thiCard=d.querySelector('[data-thipat]').closest('.chart-card');
  const leaf=root=>[...root.querySelectorAll('*:not(script)')].filter(n=>n.children.length===0).map(n=>n.textContent).join('\n');
  ok('card shows no percent sign anywhere',!leaf(thiCard).includes('%'));
  ok('totals card is parked: hidden on the page, still in source (THI_HIDDEN)',thiCard.hidden===true&&thiCard.style.display==='none'&&thiCard.getAttribute('data-parked')==='thi'&&w.eval('THI_HIDDEN')===true);
  const durTicks=[...thiCard.querySelectorAll('[data-thiax="dur"]')].map(n=>n.textContent),cntTicks=[...thiCard.querySelectorAll('[data-thiax="cnt"]')].map(n=>n.textContent);
  ok('left axis in minutes/hours, right axis in tests',durTicks.length>=3&&durTicks.some(t=>/h$/.test(t))&&durTicks[0]==='0'&&cntTicks.length>=3&&cntTicks[cntTicks.length-1].endsWith(' tests'));
  ok('two lines drawn; workers are not a line',thiCard.querySelectorAll('[data-thiline]').length===2&&![...thiCard.querySelectorAll('svg path')].some(p=>p.getAttribute('stroke')==='#94A3B8'));
  const wk=[...thiCard.querySelectorAll('[data-thiwrk]')].map(n=>n.getAttribute('data-thiwrk'));
  ok('worker band: 17 throughout, no change guide (nested suite excluded)',wk.join()==='17'&&!thiCard.querySelector('[data-thiwrkchg]')&&thiCard.textContent.includes('17 workers'));
  const labs=k=>[...thiCard.querySelectorAll(`[data-thilab="${k}"]`)].map(n=>n.textContent);
  const d0=w.__thi.data[0];
  ok('start and end labels carry raw values (start = first PR in range)',labs('start').includes(d0.cnt.toLocaleString()+' tests')&&labs('start').includes(w.fmtDuration(d0.dur))&&labs('end').includes('762 tests')&&labs('end').some(t=>/^2\.\dh test time$/.test(t)));
  ok('count change labelled at the crash commit',labs('change').includes('816'));
  w.thiTip({clientX:100,clientY:100},w.eval('THV.n')-1);
  const tipT=d.querySelector('#tip').textContent;
  ok('hover card: raw values and raw change since the start, no percent',tipT.includes('762 tests')&&tipT.includes((d0.cnt-762).toLocaleString()+' fewer than the start')&&/2\.\dh/.test(tipT)&&tipT.includes('17 parallel workers')&&!tipT.includes('%'));
  w.thiHide();
  const rowsNow=[...d.querySelectorAll('[data-thipat] [data-thipatn]')].map(n=>n.parentElement.textContent);
  ok('crash row states fewer tests, not a percent',rowsNow.some(t=>/Test count fell [\d,]+ → [\d,]+ \([\d,]+ fewer tests\)/.test(t))&&!rowsNow.join('').includes('%'));
  const pws=[...d.querySelectorAll('[data-thpw]')].map(n=>n.textContent);
  ok('legend rows carry s/test and a pattern word',pws.length>=7&&pws.every(x=>x.includes('s/test')));
  ok('crash suite reads tests vanished',pws.some(x=>x.includes('tests vanished')));
  ok('noisy suite flagged unstable or slower',pws.some(x=>x.includes('unstable run times'))||pws.some(x=>x.includes('tests got slower')));
  const mkIds=[...d.querySelectorAll('[data-thipatmk]')].map(n=>+n.getAttribute('data-thipatmk'));
  const mkx=id=>{const g2=d.querySelector(`[data-thipatmk="${id}"] circle`);return g2?+g2.getAttribute('cx'):-1};
  ok('marker numbers run 1..N left to right',mkIds.includes(1)&&mkx(1)<mkx(Math.max(...mkIds)));
  console.log(`\n${pass} passed, ${fail} failed`);
  process.exit(fail?1:0);
},400);
