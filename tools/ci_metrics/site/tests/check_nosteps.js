// The real data path: the collector publishes queued / setup / run seconds per job and nothing
// finer, so applyModel empties STEPS. Every drawing that used a step template has to degrade to
// the three measured phases rather than keep inventing a breakdown. The other checks all run on
// the sample data, which does carry a template, so this file is the only cover for that path.
const {JSDOM}=require('jsdom');
const html=require('./loadpage.js')();
const dom=new JSDOM(html,{runScripts:'dangerously',pretendToBeVisual:true});
const w=dom.window,d=w.document;
let pass=0,fail=0;
const ok=(n,c)=>{c?(pass++,console.log('  ok -',n)):(fail++,console.log('  FAIL -',n))};
const T=e=>e?e.textContent.replace(/\s+/g,' ').trim():'';
setTimeout(()=>{
  ok('sample data carries a step template',w.stepsKnown());
  w.eval('STEPS={}');            // what applyModel does once real views load
  w.renderAll();
  ok('stepsKnown() is false once the template is gone',!w.stepsKnown());

  // ---- device chart ----
  const chips=[...d.querySelectorAll('#devlines button.fchip')].map(b=>T(b));
  ok('device chart offers the phases it can measure, plus All time',chips.join('|')==='Test run|Queue wait|Setup|All time');
  ok('device chart copy lists only the chips it still offers',
     T(d.querySelector('#devlines-meta')).includes('Test run, Queue wait, Setup, All time')&&
     !T(d.querySelector('#devlines-meta')).includes('Image pull'));
  ok('every chip explains what it counts',[...d.querySelectorAll('#devlines button.fchip')].every(b=>(b.getAttribute('title')||'').length>20));
  ok('All time is every phase added up',(()=>{w.setDevPhase('all');
    const j={q:2,s:3,r:5,cat:'cpu'};return w.devJobVal(j)===10})());
  ok('Setup is the one number the collector records',(()=>{w.setDevPhase('setup');
    return w.devJobVal({q:2,s:3,r:5,cat:'cpu'})===3})());
  w.setDevPhase('run');
  w.setDevPhase('img');
  ok('a phase that no longer exists cannot stay selected',w.eval('DEV_PHASE')==='run');

  // ---- timeline legend and bar ----
  ok('timeline legend shows one Setup key, not two',
     d.querySelector('[data-lg="setup"]').style.display!=='none'&&
     d.querySelector('[data-lg="img"]').style.display==='none'&&
     d.querySelector('[data-lg="env"]').style.display==='none');
  ok('no bar segment is drawn at the invented 0.28 environment opacity',
     [...d.querySelectorAll('#timeline svg rect')].every(r=>r.getAttribute('opacity')!=='0.28'));

  // ---- commit modal ----
  w.goToCommit('4920');
  const heads=[...d.querySelectorAll('#modal .modal-body > table > thead th')].map(T);
  ok('Image and Env merge into one Setup column',heads.includes('Setup')&&!heads.includes('Image')&&!heads.includes('Env')&&heads.length===9);
  ok('the Setup column sorts',(()=>{w.jtSort('jt-jobs-4920','setup');
    const rows=[...d.querySelectorAll('#modal tbody > tr')].filter(r=>r.querySelector(':scope > td .xarr'));
    const v=rows.map(r=>parseFloat(T(r.querySelectorAll(':scope > td')[4])));
    return v.length>1&&v.some(x=>x>0)})());
  w.jtSort('jt-jobs-4920','');
  const rows=[...d.querySelectorAll('#modal tbody > tr')].filter(r=>r.querySelector(':scope > td .xarr'));
  ok('every job row has 9 cells and its drawer spans them',rows.every(r=>r.querySelectorAll(':scope > td').length===9)&&
     rows.every(r=>r.nextElementSibling.querySelector('td').getAttribute('colspan')==='9'));

  // the drawer: three measured phases, widths taken from the durations beside them
  let painted=0,cells=0,mismatch=0;
  rows.forEach(r=>{
    const dr=r.nextElementSibling.querySelector('div[id^="xrow-"]');
    const ws=[...dr.querySelector(':scope > div').children].map(x=>parseFloat(/flex:0 0 ([\d.]+)%/.exec(x.getAttribute('style'))[1]));
    const trs=[...dr.querySelector(':scope > table').querySelectorAll('tr')];
    if(trs.length!==3)mismatch++;
    trs.forEach((tr,k)=>{cells++;if(T(tr.children[1])==='—'&&ws[k]>0.0001)painted++;});
  });
  ok('every drawer lists exactly the three recorded phases',mismatch===0&&cells===rows.length*3);
  ok('no phase that took no time is given a coloured block',painted===0);
  ok('the drawer says why the steps are missing',
     T(rows[0].nextElementSibling).includes('are not published in the data this page reads'));
  ok('no invented per-step log links',!rows[0].nextElementSibling.innerHTML.includes('#step:'));

  // waterfall: three segments, full precision, no inversions
  const tot=rows.map(r=>parseFloat(T(r.querySelectorAll(':scope > td')[6])));
  const drawn=rows.map(r=>[...r.querySelectorAll(':scope > td')[7].querySelectorAll('.wf-seg')]
    .reduce((a,x)=>a+parseFloat(/flex:0 0 ([\d.]+)%/.exec(x.getAttribute('style'))[1]),0));
  ok('waterfall has three segments per row',rows.every(r=>r.querySelectorAll(':scope > td')[7].querySelectorAll('.wf-seg').length===3));
  let inv=0;
  for(let a=0;a<rows.length;a++)for(let b=0;b<rows.length;b++)
    if(tot[a]>tot[b]+1e-9&&drawn[a]<drawn[b]-1e-6)inv++;
  ok('a longer job never draws a shorter bar',inv===0);
  ok('waterfall legend names one Setup colour',T(d.querySelector('#modal')).includes('Setup (image pull and environment)'));

  console.log(`\n${pass} passed, ${fail} failed`);
  process.exit(fail?1:0);
},400);
