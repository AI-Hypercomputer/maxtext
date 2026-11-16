// Checks that a real run shows only the jobs it really had, and that runs which ran nothing do not drag
// medians down. Before 2026-11-16 the page gave every job in the catalogue a value on every merged pull
// request: a job that did not run got that job's usual (median) queue, setup and run minutes. Those filled-in
// minutes set most of the Runner Queue Watchdog's Median Wait, and a run GitHub lists no jobs for still showed
// TPU, GPU and CPU time. The sample data built into the page stores only the jobs that differ from usual, so it
// keeps the fill-in; that is checked here too.
const fs=require('fs');const path=require('path');const {JSDOM}=require('jsdom');
const loadpage=require('./loadpage.js');
const PAGE=process.env.CI_PULSE_TEST_PAGE||loadpage.PAGE;
const VIEWS=process.env.CI_PULSE_TEST_VIEWS||path.join(path.dirname(loadpage.PAGE),'views');

let pass=0,fail=0;
function ok(c,m){if(c){pass++;console.log('  ok   '+m)}else{fail++;console.log('  FAIL '+m)}}
// The page's own median: the middle value, the upper one of the two for an even count.
const median=a=>{if(!a.length)return 0;const s=[...a].sort((x,y)=>x-y);return s[Math.floor(s.length/2)]};

// --- the sample data: a commit that lists no jobs still gets every catalogue job, at its usual minutes ---
{
  const raw=fs.readFileSync(PAGE,'utf8').replace('<script>','<script>window.__CI_PULSE_MOCK__=true;</script>\n<script>');
  const dom=new JSDOM(raw,{runScripts:'dangerously',pretendToBeVisual:true,url:'http://localhost/'});
  const w=dom.window;
  const r=w.eval(`(()=>{const c=COMMITS.find(x=>!Object.keys(x.o).length);if(!c)return null;
    const j=JOBS.find(x=>x.cat==='tpu');const d=jd(c,j.id);
    return{all:allJobs(c).length,jobs:JOBS.length,q:d.q,bq:j.b.q,st:d.status,exact:!!c.exact}})()`);
  ok(r&&!r.exact,'sample commits are not marked exact');
  ok(r&&r.all===r.jobs,'a sample commit with no listed jobs still has every catalogue job ('+(r&&r.all)+' of '+(r&&r.jobs)+')');
  ok(r&&r.q===r.bq&&r.st==='pass','a sample job that is not listed gets its usual queue minutes and passes');
}

if(!fs.existsSync(path.join(VIEWS,'meta.json'))){
  console.log('  skip  no views/ next to the page - the real-data part needs them');
  console.log(`\n${pass} passed, ${fail} failed`);
  process.exit(fail?1:0);
}

const files={};
for(const name of fs.readdirSync(VIEWS)){
  if(name.endsWith('.json'))files['views/'+name]=fs.readFileSync(path.join(VIEWS,name),'utf8');
}
const stub=`<script>
window.__FILES__=${JSON.stringify(files)};
window.fetch=function(url){
  const body=window.__FILES__[String(url)];
  return Promise.resolve({ok:body!==undefined,status:body===undefined?404:200,
    json:function(){return Promise.resolve(JSON.parse(body))}});
};
</script>`;
const html=fs.readFileSync(PAGE,'utf8').replace('<script>',stub+'\n<script>');
const dom=new JSDOM(html,{runScripts:'dangerously',pretendToBeVisual:true,url:'http://localhost/'});
const w=dom.window,d=w.document;

let waited=0;
(function waitReady(){
  const loaded=w.eval('typeof TESTS==="object"&&Object.keys(TESTS).length===0&&COMMITS.length>0');
  if(loaded||waited>=5000){check();return}
  waited+=25;
  setTimeout(waitReady,25);
})();

function check(){
  const commits=w.eval('COMMITS');
  ok(commits.length>0&&commits.every(c=>c.exact===true),'every commit built from the views is marked exact ('+commits.length+')');

  // 1. No filled-in jobs: a commit has exactly the jobs its run listed.
  const counts=w.eval(`COMMITS.map(c=>[allJobs(c).length,Object.keys(c.o).filter(id=>JOBS.some(j=>j.id===id)).length])`);
  const off=counts.filter(([a,b])=>a!==b).length;
  ok(off===0,'allJobs() returns exactly the jobs each run listed ('+off+' of '+counts.length+' runs differ)');
  const absent=w.eval(`(()=>{const c=COMMITS.find(x=>JOBS.some(j=>!x.o[j.id]));if(!c)return null;
    const j=JOBS.find(j=>!c.o[j.id]);const x=jd(c,j.id);return{q:x.q,s:x.s,r:x.r,t:x.total(),st:x.status,ran:x.ran,held:x.held}})()`);
  ok(absent&&absent.q===0&&absent.s===0&&absent.r===0&&absent.t===0,'a job missing from a real run counts as zero minutes, not its usual minutes');
  ok(absent&&absent.st==='skip'&&absent.ran===false&&absent.held===false,'a job missing from a real run has status skip and held=false');

  // 2. Every wall-clock number stays finite, also for a run with no jobs at all.
  const bad=w.eval(`COMMITS.filter(c=>{const wc=wallClock(c,'all');
    return ![wc.build,wc.queue,wc.tests,wc.total,wc.catMax.tpu,wc.catQueue.gpu].every(Number.isFinite)}).map(c=>c.hash)`);
  ok(bad.length===0,'wall-clock numbers are finite for every run'+(bad.length?' (not for #'+bad.slice(0,5).join(', #')+')':''));
  const empty=w.eval(`(()=>{const c=COMMITS.find(x=>!allJobs(x).length);if(!c)return null;const wc=wallClock(c,'all');return{h:c.hash,t:wc.total,q:wc.queue,bn:wc.bottleneck}})()`);
  if(empty)ok(empty.t===0&&empty.q===0&&empty.bn==='','a run GitHub lists no jobs for (#'+empty.h+') shows no time, no wait and no bottleneck');
  else console.log('  note  no run without jobs in this data');

  // 3. Medians exclude runs that ran nothing, so empty/skipped runs never drag the baseline down.
  const sel=d.getElementById('date-range');
  const opts=[...sel.options].map(o=>o.value);
  if(opts.includes('90'))sel.value='90';
  w.eval('setRange()');
  const medInfo=w.eval(`(()=>{
    const cur=getWindows().current;
    const wcs=cur.map(c=>wallClock(c,'all'));
    const nonZero=wcs.filter(w=>w.total>0).map(w=>w.total);
    const withZeros=wcs.map(w=>w.total);
    const svgTxt=[...document.querySelectorAll('#timeline svg text')].map(t=>t.textContent).find(t=>/^median \\d+m · \\d+ runs$/.test(t))||'';
    const rMed=typeof rangeMedian==='function'?rangeMedian(COMMITS.map(c=>wallClock(c,'all'))):null;
    return{allN:cur.length,posN:nonZero.length,posMed:nonZero,zeroMed:withZeros,svgTxt,rMed};
  })()`);
  const wantPosMed=median(medInfo.posMed);
  ok(medInfo.posN<medInfo.allN&&medInfo.svgTxt===`median ${Math.round(wantPosMed)}m · ${medInfo.posN} runs`,
     `the timeline median (${medInfo.svgTxt}) excludes the ${medInfo.allN-medInfo.posN} zero-time runs (with zeros: ${Math.round(median(medInfo.zeroMed))}m)`);
  ok(medInfo.rMed!==null&&Math.abs(medInfo.rMed-wantPosMed)<0.001,
     `rangeMedian (${medInfo.rMed&&medInfo.rMed.toFixed(2)}m) matches the non-zero run median (${wantPosMed.toFixed(2)}m)`);

  // 4. Runner Queue Watchdog: Median Wait over the runs that held a runner on the device, from their own jobs.
  const res=w.eval(`(()=>{const cur=getWindows().current;const rows=computeInsights().queueRows;
    const raw=cur.map(c=>({h:c.hash,o:c.o}));const lanes=JOBS.map(j=>({id:j.id,cat:j.cat}));return{rows,raw,lanes}})()`);
  const catOf=Object.fromEntries(res.lanes.map(j=>[j.id,j.cat]));
  ['tpu','gpu','cpu'].forEach(cat=>{
    const waits=res.raw.map(c=>{
      const entries=Object.entries(c.o).filter(([id])=>catOf[id]===cat);
      if(!entries.some(([,v])=>v.h!==false))return null;
      return Math.max(...entries.map(([,v])=>v.q||0));
    }).filter(v=>v!==null);
    const row=res.rows.find(r=>r.lane===cat.toUpperCase());
    const peak=waits.length?Math.max(...waits):0;
    if(!row){ok(peak<8,`${cat.toUpperCase()}: no watchdog row, and no run waited 8+ minutes (peak ${peak.toFixed(1)})`);return}
    ok(Math.abs(row.maxQ-+peak.toFixed(1))<0.051,`${cat.toUpperCase()} Peak Wait ${row.maxQ}m is the longest real wait (${peak.toFixed(1)}m)`);
    ok(Math.abs(row.medQ-+median(waits).toFixed(1))<0.051,
       `${cat.toUpperCase()} Median Wait ${row.medQ}m is the median over the ${waits.length} runs that held a ${cat.toUpperCase()} runner (${median(waits).toFixed(2)}m)`);
  });

  // 5. Machine labels: range-wide label is the newest run's majority label; per-run label keeps that run's own label.
  const runs=[];const jobs=[];
  Object.keys(files).filter(k=>/views\/runs-/.test(k)).forEach(k=>{
    const v=JSON.parse(files[k]);const fc=w.eval('rowsOf');
    (fc(v.tables.runs)||[]).forEach(r=>runs.push(r));(fc(v.tables.jobs)||[]).forEach(r=>jobs.push(r));
  });
  const at=new Map(runs.map(r=>[r.run_id,String(r.created_at||r.merged_at||'')]));
  const cpuJobs=jobs.filter(r=>r.lane==='CPU'&&r.runner_label);
  const newestAt=cpuJobs.reduce((m,r)=>{const t=at.get(r.run_id)||'';return t>m?t:m},'');
  const tally=new Map();
  cpuJobs.filter(r=>(at.get(r.run_id)||'')===newestAt).forEach(r=>tally.set(r.runner_label,(tally.get(r.runner_label)||0)+1));
  const wantCat=[...tally.entries()].sort((a,b)=>b[1]-a[1]||(a[0]<b[0]?-1:1))[0][0];
  const shown=w.eval(`machineOfCat('cpu')`);
  ok(shown===wantCat,`the CPU machine label is the newest run's majority label (${shown})`);
  const oldN2=w.eval(`(()=>{
    const c=COMMITS.find(x=>allJobs(x).some(j=>j.cat==='cpu'&&/n2-32/.test(j.label||'')));
    if(!c)return null;
    return typeof machineOfRun==='function'?machineOfRun(c,'cpu'):'missing';
  })()`);
  if(oldN2!==null)ok(oldN2==='linux-x86-n2-32',`an older n2-32 run still reports its own CPU label (${oldN2})`);

  // 6. Worker views omit workers and suites that did not run on a real commit.
  const wkCheck=w.eval(`(()=>{
    const ents=wkEntities();
    const multi=ents.find(e=>e.jobs.length>1&&!e.split);
    const partial=multi&&COMMITS.find(c=>{const k=multi.jobs.filter(id=>!!c.o[id]).length;return k>0&&k<multi.jobs.length});
    const skipped=COMMITS.find(c=>ents.some(e=>e.jobs.every(id=>!c.o[id])));
    if(skipped)openWkPop(skipped.hash);
    const popHtml=skipped?(document.getElementById('wkpop-body')||{}).innerHTML||'':'';
    const skippedEnt=skipped&&ents.find(e=>e.jobs.every(id=>!skipped.o[id]));
    return{
      partialOk:!partial||wkWorkerRows(partial,multi).length===multi.jobs.filter(id=>!!partial.o[id]).length,
      skipWorkersOk:!skippedEnt||wkWorkers(skipped,skippedEnt).length===0,
      popOmitsSkipped:!skippedEnt||!popHtml.includes(skippedEnt.name)
    };
  })()`);
  ok(wkCheck.partialOk&&wkCheck.skipWorkersOk&&wkCheck.popOmitsSkipped,'worker views omit workers and suites that did not run');

  console.log(`\n${pass} passed, ${fail} failed`);
  process.exit(fail?1:0);
}
