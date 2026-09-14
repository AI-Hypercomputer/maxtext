// Drives the dashboard down its REAL path: no sample data, a stubbed fetch that
// serves the collector's own view files, and assertions on what actually got
// drawn. The other checks cover the sample data; this one covers the loader,
// the columnar reader and the adapter that turns view rows into the page's
// model.
const fs=require('fs');const path=require('path');const {JSDOM}=require('jsdom');
const PAGE=require('./loadpage.js').PAGE;
const VIEWS=path.join(path.dirname(PAGE),'views');

let pass=0,fail=0;
function ok(c,m){if(c){pass++;console.log('  ok   '+m)}else{fail++;console.log('  FAIL '+m)}}

if(!fs.existsSync(path.join(VIEWS,'meta.json'))){
  console.log('  skip  no views/ next to the page - run tools/ci_metrics/deploy/refresh-data.sh first');
  console.log('\n0 passed, 0 failed');
  process.exit(0);
}

// Every view file, read in node and handed to the page through a fetch stub.
// This is the same content a web server would return, so the page's own
// loadViews() and adaptViews() do all the work.
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

// boot() is async, so let its promise chain drain before looking at the page.
setTimeout(check,120);

function check(){
  const meta=JSON.parse(files['views/meta.json']);

  // --- the loader ran and nothing fell back to sample data -----------------
  // body.textContent includes the page's own <script> source, so look for the
  // rendered card rather than for the words anywhere on the page.
  const titles=[...d.querySelectorAll('.ct')].map(e=>e.textContent);
  ok(!titles.some(x=>/could not load its data/.test(x)),
     'the page did not show the load-failure card');
  ok(w.eval('TESTS')&&Object.keys(w.eval('TESTS')).length===0,
     'the sample per-test detail was cleared, so nothing invented is on screen');

  // --- meta.json drove the clock and the freshness line ---------------------
  const fresh=d.getElementById('freshness');
  ok(fresh&&fresh.textContent.indexOf('Data collected')===0,
     'the freshness line reports when the data was collected: '+(fresh&&fresh.textContent));
  const today=w.eval('TODAY');
  ok(Math.abs(new Date(today)-new Date(meta.generated_at))<86400000,
     'TODAY comes from meta.json generated_at, not the sample date');

  // --- the model was built from the view rows -------------------------------
  // There is one runs file per month, so every month has to be counted.
  const runsFiles=Object.keys(files).filter(k=>/views\/runs-/.test(k)).sort();
  const runsTable={columns:[],rows:[]},jobsTable={columns:[],rows:[]};
  runsFiles.forEach(k=>{
    const f=JSON.parse(files[k]);
    ['runs','jobs'].forEach(name=>{
      const into=name==='runs'?runsTable:jobsTable;
      const from=f.tables[name];
      if(!from)return;
      if(!into.columns.length)into.columns=from.columns;
      into.rows.push(...from.rows);
    });
  });
  const COMMITS=w.eval('COMMITS'),JOBS=w.eval('JOBS');

  // A pull request that was pushed to again has more than one run in the view.
  // Only its last run is the state it merged in, so the chart gets one bar.
  const rc={};runsTable.columns.forEach((c,i)=>{rc[c]=i});
  const distinctPrs=new Set(runsTable.rows.map(r=>String(r[rc.pr])));
  ok(COMMITS.length===distinctPrs.size,
     `one commit per pull request (${COMMITS.length} of ${distinctPrs.size} distinct, from `+
     `${runsTable.rows.length} run row(s) in ${runsFiles.length} file(s))`);
  ok(new Set(COMMITS.map(c=>c.hash)).size===COMMITS.length,
     'no pull request appears twice, so no bar can overwrite another one is numbers');
  ok(COMMITS.every(c=>/^#\d+$/.test(c.pr)),'every commit carries a real pull request number');
  ok(COMMITS.every(c=>['pass','fail','flaky'].includes(c.status)),'every status is one the charts know');

  // NEWEST FIRST is the order the sample data uses and the charts rely on: they
  // reverse the list when drawing, so the newest run lands on the right of the
  // x-axis. An ascending list mirrors every chart without any error.
  const pt=w.eval('prTime');
  const newestFirst=COMMITS.every((c,i)=>i===0||pt(COMMITS[i-1].hash)>=pt(c.hash));
  ok(newestFirst,'commits are ordered newest first, the order the charts reverse when drawing');
  ok(pt(COMMITS[0].hash)>=pt(COMMITS[COMMITS.length-1].hash),
     `the newest run is first (${COMMITS[0].pr} is newer than ${COMMITS[COMMITS.length-1].pr})`);

  ok(JOBS.length>0&&JOBS.length<=jobsTable.rows.length,
     `the job catalogue was discovered from the data (${JOBS.length} distinct job(s))`);
  ok(JOBS.every(j=>['tpu','gpu','cpu','build','infra'].includes(j.cat)),
     'every job has a category the colours know');
  ok(JOBS.every(j=>j.b&&typeof j.b.q==='number'&&typeof j.b.r==='number'),
     'every job has numeric baseline minutes');

  // --- minutes, not seconds -------------------------------------------------
  const ci={};jobsTable.columns.forEach((c,i)=>{ci[c]=i});
  const sample=jobsTable.rows.find(r=>r[ci.run_seconds]>0);
  if(sample){
    const hash=String(sample[ci.pr]);
    const c=COMMITS.find(x=>x.hash===hash);
    const slug=sample[ci.lane]==='Build'?'pkg':
      (sample[ci.worker]==null?sample[ci.flavor]:sample[ci.flavor]+'#'+sample[ci.worker]);
    const got=c&&c.o[slug];
    ok(got&&Math.abs(got.r-sample[ci.run_seconds]/60)<0.02,
       `run_seconds became minutes for ${slug} on #${hash} (${sample[ci.run_seconds]}s -> ${got&&got.r}m)`);
  }else{
    ok(true,'(no job row with a run time to convert)');
  }

  // --- the real machine labels came from the data, not a hard-coded table ---
  const labels=w.eval('RUNNER_LABELS');
  const fromData=new Set(jobsTable.rows.map(r=>r[ci.runner_label]).filter(Boolean));
  ok(Object.values(labels).every(v=>fromData.has(v)),
     'every machine label shown is one the collector actually recorded');

  // --- the charts widen instead of squashing the bars -----------------------
  // The sample data was drawn at 1100 units for about twenty runs. With more
  // runs than that the box has to grow, or every bar becomes a hairline. The
  // card scrolls sideways instead.
  const BOX=w.eval('CHART_BOX'),SLOT=w.eval('CHART_SLOT');
  const widest=sel=>{
    const all=[...d.querySelectorAll(sel)]
      .map(s=>({s,W:+((s.getAttribute('viewBox')||'0 0 0 0').split(' ')[2])}));
    all.sort((a,b)=>b.W-a.W);
    return all[0];
  };
  const runsShown=w.eval('getWindows()').current.length;
  [['#timeline svg','run-time chart'],['#worker-chart svg','worker chart'],
   ['#devlines svg','device chart']].forEach(([sel,name])=>{
    const c=widest(sel);
    if(!c){ok(false,name+' drew no SVG');return}
    const expected=runsShown>21;
    ok(expected?c.W>BOX:c.W===BOX,
       `${name} box is ${c.W} units for ${runsShown} run(s) `+
       `(${expected?'wider than':'the usual'} ${BOX})`);
    ok(!expected||c.s.style.minWidth===c.W+'px',
       `${name} sets a min-width so the card scrolls rather than shrinking the bars`);
  });
  // A bar should still be about as wide as it is in the sample data.
  const tl=widest('#timeline svg');
  if(tl){
    const bars=[...tl.s.querySelectorAll('rect')].map(r=>+(r.getAttribute('width')||0)).filter(x=>x>0.5);
    const max=Math.max(...bars);
    ok(max>SLOT*0.5,`the widest bar is ${max.toFixed(0)} units, not a hairline (slot is ${SLOT})`);
  }

  // --- something was actually drawn ----------------------------------------
  ok(d.querySelectorAll('#timeline svg').length>0,'the run-time chart drew an SVG');
  ok(d.querySelectorAll('#worker-chart svg').length>0,'the worker chart drew an SVG');
  ok(d.querySelectorAll('#devlines svg').length>0,'the device chart drew an SVG');
  ok(d.getElementById('flaky')&&d.getElementById('flaky').innerHTML.length>0,'the flaky card rendered');

  console.log('\n'+pass+' passed, '+fail+' failed');
  process.exit(fail?1:0);
}
