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
  const runsFile=Object.keys(files).find(k=>/views\/runs-/.test(k));
  const runsTable=runsFile?JSON.parse(files[runsFile]).tables.runs:{rows:[]};
  const jobsTable=runsFile?JSON.parse(files[runsFile]).tables.jobs:{rows:[]};
  const COMMITS=w.eval('COMMITS'),JOBS=w.eval('JOBS');

  ok(COMMITS.length===runsTable.rows.length,
     `one commit per row in the runs view (${COMMITS.length} of ${runsTable.rows.length})`);
  ok(COMMITS.every(c=>/^#\d+$/.test(c.pr)),'every commit carries a real pull request number');
  ok(COMMITS.every(c=>['pass','fail','flaky'].includes(c.status)),'every status is one the charts know');

  const ordered=COMMITS.every((c,i)=>i===0||w.eval('prTime')(COMMITS[i-1].hash)<=w.eval('prTime')(c.hash));
  ok(ordered,'commits are ordered oldest first, the order the charts assume');

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

  // --- something was actually drawn ----------------------------------------
  ok(d.querySelectorAll('#timeline svg').length>0,'the run-time chart drew an SVG');
  ok(d.querySelectorAll('#worker-chart svg').length>0,'the worker chart drew an SVG');
  ok(d.querySelectorAll('#devlines svg').length>0,'the device chart drew an SVG');
  ok(d.getElementById('flaky')&&d.getElementById('flaky').innerHTML.length>0,'the flaky card rendered');

  console.log('\n'+pass+' passed, '+fail+' failed');
  process.exit(fail?1:0);
}
