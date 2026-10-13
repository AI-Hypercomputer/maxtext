// Checks that the page works whatever a job, flavor or test is called.
//
// Names are free text. Some of them are spelled like words that every
// JavaScript object already has: "constructor", "toString", "__proto__". The
// page has to treat them like any other name. Each scenario below gives some
// items such a name in the real view files, loads the page through a fetch stub
// (the same way check_realdata.js does), and asks:
//   - did anything throw, while the page loaded and while its pop-ups opened?
//   - do the charts draw the same bars as with the original names?
//   - does the new name show on the page as text?
//   - does the page keep the old numbers under the new name?
// The last block calls the view reader directly, with a few made-up rows for
// the cases the real files do not reach.
const fs=require('fs');const path=require('path');const {JSDOM,VirtualConsole}=require('jsdom');
const PAGE=require('./loadpage.js').PAGE;
const VIEWS=path.join(path.dirname(PAGE),'views');

let pass=0,fail=0;
function ok(c,m){if(c){pass++;console.log('  ok   '+m)}else{fail++;console.log('  FAIL '+m)}}
const wait=ms=>new Promise(r=>setTimeout(r,ms));
const own=(o,k)=>Object.prototype.hasOwnProperty.call(o,k);

if(!fs.existsSync(path.join(VIEWS,'meta.json'))){
  console.log('  skip  no views/ next to the page - run tools/ci_metrics/deploy/refresh-data.sh first');
  console.log('\n0 passed, 0 failed');
  process.exit(0);
}

// The words every object already has, used as names. The scenarios use five of
// them. The lookups and the view reader are asked about all of them.
const NAMES=['constructor','__proto__','toString','hasOwnProperty','valueOf'];
const ALL_NAMES=[...new Set([...NAMES,...Object.getOwnPropertyNames(Object.prototype)])];

// --- the view files ----------------------------------------------------------
// meta.json and the month files, as text. views/pr/*.json are left out: the
// page never asks for them.
const RAW={};
for(const n of fs.readdirSync(VIEWS))if(n.endsWith('.json'))RAW['views/'+n]=fs.readFileSync(path.join(VIEWS,n),'utf8');
const parsed=()=>Object.fromEntries(Object.entries(RAW).map(([k,v])=>[k,JSON.parse(v)]));
const tablesOf=f=>Object.values(f.tables||{});
// The column that holds a job's name: `name` in the jobs table, `job_name` in
// the rescue tables. (In rescue_tests, `name` is the name of a test.)
const jobNameColumn=t=>t.columns.includes('job_name')?'job_name':t.columns.includes('job_id')?'name':null;

// Real examples, picked from the files as they are today, so a data refresh does
// not break the check: a flavor that ran on several workers (its job ids are
// like cpu-unit#2), a flavor with no worker number (its job id is the bare
// flavor), and two jobs that never have a flavor. Each is the busiest of its
// kind in the last 14 days, the range the page opens on, so it is on the page.
function pickExamples(files){
  const all=name=>Object.values(files).flatMap(f=>{
    const t=f.tables&&f.tables[name];
    return t?t.rows.map(r=>Object.fromEntries(t.columns.map((c,i)=>[c,r[i]]))):[];
  });
  const jobs=all('jobs');
  const since=new Date(files['views/meta.json'].generated_at).getTime()-14*86400000;
  const recent=new Map();   // job name -> rescue rows in that range
  all('rescues').forEach(r=>{if(new Date(r.created_at).getTime()>=since)recent.set(r.job_name,(recent.get(r.job_name)||0)+1)});
  const hits=word=>[...recent].reduce((n,[name,c])=>n+((name||'').split(/[^\w-]+/).includes(word)?c:0),0);
  const workers=new Map();  // flavor -> the worker numbers it ran with
  jobs.forEach(j=>{if(j.flavor)workers.set(j.flavor,(workers.get(j.flavor)||new Set()).add(j.worker))});
  const best=list=>list.sort((a,b)=>hits(b)-hits(a)||(a<b?-1:1))[0];
  const flavors=[...workers.keys()];
  const flavored=new Set(jobs.filter(j=>j.flavor).map(j=>j.name));
  const bare=[...new Set(jobs.filter(j=>!j.flavor&&j.lane!=='Build'&&!flavored.has(j.name)).map(j=>j.name))]
    .filter(n=>recent.get(n)>0).sort((a,b)=>recent.get(b)-recent.get(a)||(a<b?-1:1));
  return{multi:best(flavors.filter(f=>workers.get(f).size>1)),
         single:best(flavors.filter(f=>workers.get(f).size===1&&workers.get(f).has(null))),
         jobs:bare.slice(0,2)};
}
const EX=pickExamples(parsed());

// Gives one flavor a new name wherever it is written: in the flavor columns,
// and as a word inside job names such as "CPU Tests (cpu-unit) / cpu-unit".
function renameFlavor(files,from,to){
  const word=new RegExp('(?<![\\w-])'+from.replace(/\W/g,'\\$&')+'(?![\\w-])','g');
  for(const f of Object.values(files))for(const t of tablesOf(f)){
    const nameCol=jobNameColumn(t);
    t.columns.forEach((c,i)=>{
      if(c==='flavor'||c==='suite_id'||c==='nested_in')t.rows.forEach(r=>{if(r[i]===from)r[i]=to});
      else if(c===nameCol)t.rows.forEach(r=>{if(typeof r[i]==='string')r[i]=r[i].replace(word,()=>to)});
    });
  }
}
// Gives one flavorless job a new name, in the jobs table and in its rescue rows.
function renameJob(files,from,to){
  for(const f of Object.values(files))for(const t of tablesOf(f)){
    const i=t.columns.indexOf(jobNameColumn(t));
    if(i>=0)t.rows.forEach(r=>{if(r[i]===from)r[i]=to});
  }
}
// Gives every value of some text columns the same name.
function renameColumns(files,cols,to){
  for(const f of Object.values(files))for(const t of tablesOf(f))
    t.columns.forEach((c,i)=>{if(cols.includes(c))t.rows.forEach(r=>{if(typeof r[i]==='string')r[i]=to})});
}
// The rescue-tests table is empty in today's data. Add one row per name, tied to
// the newest rescue of a merged pull request, so the flaky-test list has
// something to draw. Returns the id of the job the tests belong to.
function addTests(files,names){
  const rowsOf=(f,name)=>{const t=f.tables&&f.tables[name];return t?t.rows.map(r=>Object.fromEntries(t.columns.map((c,i)=>[c,r[i]]))):[]};
  const flaky=Object.keys(files).filter(k=>/^views\/flaky-/.test(k));
  const merged=new Set(Object.keys(files).filter(k=>/^views\/runs-/.test(k)).flatMap(k=>rowsOf(files[k],'runs').map(r=>String(r.pr))));
  let best=null,bestFile=null;
  flaky.forEach(k=>rowsOf(files[k],'rescues').forEach(r=>{
    if(r.rescued&&r.flavor&&merged.has(String(r.pr))&&(!best||r.created_at>best.created_at)){best=r;bestFile=k}
  }));
  const t=files[bestFile].tables.rescue_tests;
  names.forEach(name=>{
    const row={run_id:best.run_id,pr:best.pr,job_name:best.job_name,failed_attempt:best.failed_attempt,
      suite_id:best.flavor,flavor:best.flavor,worker:best.worker,
      classname:name==='test_synthetic_b'?'tests.unit.synthetic_test.SyntheticTest':'',
      name,status:'failed',duration:1.5,failure_message:'AssertionError: synthetic\nsecond line'};
    t.rows.push(t.columns.map(c=>c in row?row[c]:null));
  });
  return best.worker==null?best.flavor:best.flavor+'#'+best.worker;
}

// --- loading the page ---------------------------------------------------------
const STUB=`<script>
window.fetch=function(url){
  const body=window.__FILES__[String(url)];
  return Promise.resolve({ok:body!==undefined,status:body===undefined?404:200,
    json:function(){return Promise.resolve(JSON.parse(body))}});
};
</script>`;
const HTML=fs.readFileSync(PAGE,'utf8').replace('<script>',STUB+'\n<script>');

// Loads the page with these view files and waits until it has drawn, or given up.
// A script error that no handler caught is collected in P.errors. An error the
// page's own loader caught shows up as the load-failure card instead.
async function openPage(f){
  const files={};for(const[k,v]of Object.entries(f))files[k]=JSON.stringify(v);
  const errors=[];
  const vc=new VirtualConsole();
  vc.on('jsdomError',e=>errors.push(String(e&&e.message||e).split('\n')[0]));
  const dom=new JSDOM(HTML,{runScripts:'dangerously',pretendToBeVisual:true,url:'http://localhost/',virtualConsole:vc,
    beforeParse(win){win.__FILES__=files}});
  const w=dom.window,d=w.document;
  const card=()=>[...d.querySelectorAll('.ct')].find(e=>/could not load its data/.test(e.textContent));
  for(let i=0;i<500&&!d.querySelector('#timeline svg')&&!card();i++)await wait(20);
  return{w,d,errors,loadError:()=>{const c=card();return c?c.parentNode.textContent.replace(/\s+/g,' ').trim().slice(0,150):''}};
}

// How many shapes a chart drew: bars, dots and so on, but not the invisible
// squares that catch the mouse.
const shapes=(d,sel)=>[...d.querySelectorAll(`${sel} svg rect, ${sel} svg path, ${sel} svg circle`)]
  .filter(e=>e.getAttribute('fill')!=='transparent').length;
const barsOf=P=>({'run-time chart':shapes(P.d,'#timeline'),'worker chart':shapes(P.d,'#worker-chart'),'device chart':shapes(P.d,'#devlines')});
// The parts of the page's model that carry names, as plain data.
const snapshot=w=>JSON.parse(w.eval(`JSON.stringify({jobs:JOBS,labels:RUNNER_LABELS,counts:TEST_COUNTS,groups:WK_GROUPS,
  rescues:RESCUES,o:COMMITS.map(c=>c.o)})`));

// Every map in the page that is filled from the view files. Asking one of them
// for a name it was never given must come back empty.
const MAPS=['RUNNER_LABELS','TEST_COUNTS','WK_GROUPS','TIMES','TIME_ISO','RUN_IDS','TRIGGERS','PROBE_Q','TESTS','STEPS'];
function inherited(P){
  const bad=[];
  const ask=(label,m)=>{for(const n of ALL_NAMES)if(m[n]!==undefined&&!own(m,n))bad.push(`${label}.${n}`)};
  MAPS.forEach(k=>ask(k,P.w.eval(k)));
  P.w.eval('COMMITS').forEach(c=>ask(`COMMITS[${c.hash}].o`,c.o));
  return bad;
}

// A value read from a map that was never filled shows up in the page as code,
// like "[object Object]" or "function Object() {...}". No real value looks like that.
const LEAK=/\[object |function \w*\(/;

// Opens the pop-ups and detail views the page has, for every job and test, and
// for a few runs: the newest, the oldest and one in between. `focus` lists the
// ids that carry the new names: their one-suite chart is drawn too. Returns what
// threw, and what showed a code value.
function poke(P,focus){
  const {w,d}=P,thrown=[];
  const run=(what,fn)=>{try{fn()}catch(e){thrown.push(`${what} threw: ${String(e&&e.message||e).split('\n')[0]}`)}};
  const look=(what,sel)=>{const m=LEAK.exec(d.querySelector(sel).innerHTML);if(m)thrown.push(`${what} showed a code value: ${m[0]}`)};
  const commits=w.eval('COMMITS'),n=commits.length;
  const sample=[...new Set([0,n>>1,n-1])].filter(i=>i>=0&&i<n).map(i=>commits[i]);
  run('the worker pop-up',()=>sample.forEach(c=>{w.openWkPop(c.hash);look('the worker pop-up','#wkpop-body')}));
  run('the suite pop-ups',()=>{
    const ents=w.eval('wkEntities()');
    ents.forEach(e=>{w.openWkSuitePop(commits[0].hash,e.gid);look('a suite pop-up','#wkpop-body')});
    const one=ents.filter((e,i)=>i===0||(focus||[]).includes(e.gid));
    one.forEach(e=>{w.setWKSel(e.gid);look('the one-suite chart','#worker-chart')});
    w.setWKSel('all');
  });
  run('the run details',()=>{for(const c of sample.slice(0,2)){w.goToCommit(c.hash);look('the run details','#modal');w.closeModal()}});
  run('the flaky pop-ups',()=>{
    const idx=w.eval('flakyCalc().idxs');
    w.eval('RESCUES').forEach((j,i)=>{
      w.openFlakyPop(j.id);look('a flaky pop-up','#wkpop-body');
      const pr=(j.events[0]||(j.fails||[])[0]||{}).pr;
      if(pr&&i<10)w.openFlakyPop(j.id,pr);
      if(idx.length&&i<10){w.flakyCellTip({clientX:10,clientY:10},j.id,idx[0]);look('a flaky tooltip','#tip')}
    });
    w.flakyTipHide();
  });
  run('the flaky test pop-ups',()=>{
    for(const t of w.eval('FLAKY_TESTS')){w.openFlakyTestPop(t.test,t.job);look('a test pop-up','#wkpop-body');w.openFlakyPop(t.job,t.prs[0])}
  });
  run('the page',()=>look('the page','#page-main'));
  w.closeWkPop();
  return thrown;
}

// The page with the new names must hold the same numbers as the page with the
// old names, filed under the new ones. `renames` is a list of [from,to]
// flavors. Returns what differs.
function sameNumbers(before,after,renames){
  const bad=[];
  const entries=o=>new Map(Object.entries(o||{}));
  const cmp=(label,a,b)=>{if(JSON.stringify(a)!==JSON.stringify(b))bad.push(label)};
  // Where an id goes. A flavor's own ids (cpu-unit, cpu-unit#2) follow it. An id
  // cut from a job name that mentions the flavor changes in ways this check does
  // not model, so it has no place here (null) and is left out.
  const mapId=id=>{
    for(const[from,to]of renames){
      if(id===from||id.startsWith(from+'#'))return to+id.slice(from.length);
      if(id.includes(from))return null;
    }
    return id;
  };
  const ren=s=>renames.reduce((t,[from,to])=>t.split(from).join(to),s);
  if(before.jobs.length!==after.jobs.length)bad.push('job count');
  for(const j of before.jobs){
    const id=mapId(j.id);if(id===null)continue;
    const k=after.jobs.find(x=>x.id===id);
    cmp('job '+j.id,k&&[k.cat,k.lane,k.b],[j.cat,j.lane,j.b]);
  }
  for(const key of ['labels','counts']){
    const a=entries(after[key]);
    for(const[id,v]of Object.entries(before[key])){const m=mapId(id);if(m!==null)cmp(`${key} ${id}`,a.get(m),v)}
    if(a.size!==Object.keys(before[key]).length)bad.push(key+' size');
  }
  const g=entries(after.groups);
  for(const[id,v]of Object.entries(before.groups)){const m=mapId(id);if(m!==null)cmp('group '+id,g.get(m),JSON.parse(ren(JSON.stringify(v))))}
  before.o.forEach((o,i)=>{
    const a=entries(after.o[i]);
    for(const[id,v]of Object.entries(o)){const m=mapId(id);if(m!==null)cmp(`run ${i} ${id}`,a.get(m),v)}
  });
  const events=m=>m.rescues.map(r=>[r.events.length,(r.fails||[]).length]);
  cmp('rescue events',events(after),events(before));
  return bad;
}
// The same, for a job that was renamed to a new id: its numbers must not change.
function jobMoved(before,after,oldId,newId,newName){
  const bad=[];
  const was=before.jobs.find(j=>j.id===oldId),now=after.jobs.find(j=>j.id===newId);
  if(!was||!now)return [`job ${now?oldId:newId} missing`];
  if(now.name!==newName)bad.push(`name is ${now.name}`);
  if(JSON.stringify([now.cat,now.lane,now.b])!==JSON.stringify([was.cat,was.lane,was.b]))bad.push('baseline minutes');
  const entries=o=>new Map(Object.entries(o||{}));
  if(entries(after.labels).get(newId)!==entries(before.labels).get(oldId))bad.push('machine label');
  before.o.forEach((o,i)=>{
    if(JSON.stringify(entries(o).get(oldId))!==JSON.stringify(entries(after.o[i]).get(newId)))bad.push(`run ${i} minutes`);
  });
  return bad;
}

// --- one page, all the checks ------------------------------------------------
const BASE={};
async function verify(P,tag,sc){
  const err=P.loadError();
  ok(!err,`${tag}: the page loads`+(err?` - it showed: ${err}`:''));
  ok(!P.errors.length,`${tag}: no script error while loading`+(P.errors.length?`: ${P.errors.slice(0,2).join(' | ')}`:''));
  const bars=barsOf(P);
  for(const k of Object.keys(BASE.bars))
    if(sc.bars===false)ok(bars[k]>0,`${tag}: the ${k} draws its shapes (${bars[k]})`);
    else ok(bars[k]>0&&bars[k]===BASE.bars[k],`${tag}: the ${k} draws ${BASE.bars[k]} shapes, as with the original names (${bars[k]})`);
  for(const[what,test]of sc.shows||[])ok(test(P),`${tag}: ${what}`);
  if(sc.kept){
    const bad=sc.kept(snapshot(P.w));
    ok(!bad.length,`${tag}: the old numbers are filed under the new names`+(bad.length?` - differs: ${bad.slice(0,4).join(', ')}`:''));
  }
  const inh=inherited(P);
  ok(!inh.length,`${tag}: asking a map for a name it never got comes back empty`+(inh.length?` - answered: ${inh.slice(0,4).join(', ')}`:''));
  const thrown=poke(P,sc.focus);
  await wait(20);   // let the page's MutationObserver run
  thrown.push(...P.errors.splice(0));
  ok(!thrown.length,`${tag}: every pop-up and detail view opens without an error`+(thrown.length?` - ${thrown.slice(0,2).join(' | ')}`:''));
  if(sc.after)sc.after(P);
  const dirty=Object.keys(P.w.Object.prototype);
  ok(!dirty.length&&P.w.eval('({}).total')===undefined,`${tag}: nothing was added to Object.prototype`+(dirty.length?` - found: ${dirty.slice(0,4).join(', ')}`:''));
}

// --- the scenarios ----------------------------------------------------------
const scenarios=[];
const text=(sel,n)=>P=>{const e=P.d.querySelector(sel);return !!e&&e.textContent.includes(n)};
const choice=n=>P=>[...P.d.querySelectorAll('#wk-chips option')].some(o=>o.value===n);
// Both flavors are renamed on one page, each to a different name, so five pages
// give each flavor all five names.
const {multi:MULTI,single:SINGLE}=EX;
if(MULTI&&SINGLE)NAMES.forEach((n,i)=>{
  const m=NAMES[(i+1)%NAMES.length];
  scenarios.push({tag:`flavors ${MULTI} -> ${n}, ${SINGLE} -> ${m}`,
    edit:f=>{renameFlavor(f,MULTI,n);renameFlavor(f,SINGLE,m)},
    focus:[n,m],
    shows:[[`the worker dropdown shows ${n} as text`,text('#wk-chips',n)],
           [`the worker dropdown offers ${m} as a choice`,choice(m)],
           [`the flaky card shows ${n} as text`,text('#flaky',n)],
           [`the flaky card shows ${m} as text`,text('#flaky',m)]],
    kept:s=>sameNumbers(BASE.snap,s,[[MULTI,n],[SINGLE,m]])});
});
// A job with no flavor gets its id from its name: "constructor" stays
// "constructor", "__proto__" becomes "proto". Two jobs, one page.
const jobs=[['__proto__','proto'],['constructor','constructor']].slice(0,EX.jobs.length).map(([n,id],i)=>[EX.jobs[i],n,id]);
if(jobs.length)scenarios.push({tag:`job names -> ${jobs.map(j=>j[1]).join(', ')}`,
  edit:f=>jobs.forEach(([from,n])=>renameJob(f,from,n)),
  focus:jobs.map(j=>j[2]),
  shows:jobs.flatMap(([,n])=>[[`the worker dropdown shows ${n} as text`,text('#wk-chips',n)],[`the flaky card shows ${n} as text`,text('#flaky',n)]]),
  // The old job's id is the id of the rescue group that carries its name.
  kept:s=>jobs.flatMap(([from,n,id])=>jobMoved(BASE.snap,s,(BASE.snap.rescues.find(r=>r.name===from)||{}).id,id,n))});
// Columns the collector fills from a fixed list hold a bad value: a lane, a pull
// request number, and a table name, all spelled like words every object has.
// The charts change shape here (a job is in no lane), so only "they draw" is asked.
if(SINGLE){
  const n='constructor';
  scenarios.push({tag:`a lane, a pull request number and a table named ${n}`,
    bars:false,
    edit:f=>{
      const col=(t,c)=>t.columns.indexOf(c);
      for(const[k,file]of Object.entries(f))for(const[tn,t]of Object.entries(file.tables||{})){
        if(tn==='jobs')t.rows.forEach(r=>{if(r[col(t,'flavor')]===SINGLE)r[col(t,'lane')]=n});
        if(tn==='queue')t.rows.forEach(r=>{r[col(t,'lane')]=n});
        if(tn==='suites'&&t.rows.length)t.rows[0][col(t,'pr')]=n;
      }
      const runs=Object.keys(f).find(k=>/^views\/runs-/.test(k));
      f[runs].tables[n]={columns:['a'],rows:[['b']]};
    },
    after:P=>{
      const w=P.w;
      ok(w.eval('JOBS').every(j=>['tpu','gpu','cpu','build','infra'].includes(j.cat)),`a lane named ${n}: every job is still in one of the five categories`);
      ok(w.eval('Object.values(PROBE_Q)').every(v=>Object.keys(v).sort().join()==='cpu,gpu,tpu'),`a lane named ${n}: each queue wait is filed under tpu, gpu and cpu only`);
      ok(w.eval('TH_CATEGORIES').every(c=>c.data.every(d=>typeof d.commitIdx==='number')),`a pull request number named ${n}: every suite point sits at a pull request the page drew`);
    }});
}
// A test with a bad name, and a bad name in every other column the page might
// one day use as a key. The rescue-tests table is empty in today's data, so
// the test is made up.
{
  const n='__proto__';
  let job;
  scenarios.push({tag:`test name and every other name column -> ${n}`,
    edit:f=>{
      renameColumns(f,['title','author','base_ref','workflow_name','workflow_path','runner_label','pool'],n);
      job=addTests(f,[n,'test_synthetic_b']);
    },
    shows:[[`the flaky card shows the test ${n} as text`,text('#flaky',n)]],
    after:P=>{
      const pop=()=>P.d.querySelector('#wkpop-body').textContent;
      P.w.openFlakyTestPop(n,job);
      ok(pop().includes(n),`test name and every other name column -> ${n}: the test's own pop-up shows it as text`);
      P.w.openFlakyTestPop('test_synthetic_b',job);
      ok(pop().includes(n),`test name and every other name column -> ${n}: the other test's pop-up lists it as text`);
      P.w.openWkPop(P.w.eval('COMMITS[0].hash'));
      ok(pop().includes(n),`test name and every other name column -> ${n}: a pull request title shows as text`);
    }});
}

// --- the view reader, called directly ------------------------------------------
// The real files give every flavor a worker number or none, and the rescue-tests
// table is empty. These made-up rows put a name into every table the reader
// reads. With two workers the flavor's ids are n#1 and n#2, so a job with no
// flavor can be named n. With no worker number the flavor's id is the bare n,
// which is the id that name would make for a job with no flavor, so that variant
// has no such job.
function madeUp(n,workers){
  const ws=workers?[1,2]:[null];
  const jobName=`Tests (${n}) / ${n}`;
  const job=k=>({run_id:10,pr:1,attempt:1,name:jobName,lane:'CPU',flavor:n,worker:k,conclusion:'success',
    runner_label:'machine-1',queued_seconds:60,setup_seconds:120,run_seconds:600});
  const rescue=over=>({run_id:10,pr:1,job_name:jobName,lane:'CPU',flavor:n,worker:ws[0],rescued:true,failed_attempt:1,
    failed_conclusion:'failure',wasted_seconds:90,html_url:'https://github.com/x/y',...over});
  const test=over=>({run_id:10,pr:1,job_name:jobName,failed_attempt:1,flavor:n,worker:ws[0],classname:'',name:n,
    failure_message:'boom',...over});
  const rescues=[rescue({})],tests=[test({})];
  if(workers){
    rescues.push(rescue({job_name:n,flavor:null,worker:null,rescued:false}));
    tests.push(test({job_name:n,flavor:null,worker:null,name:'test_b'}));
  }
  return{
    runs:{runs:[{pr:1,title:'a change',run_id:10,created_at:'2026-09-01T10:00:00Z',merged_at:'2026-09-01T11:00:00Z',
      event:'pull_request',conclusion:'success',attempts:2,overlapping_runs:0}],jobs:ws.map(job)},
    suites:{suites:[{pr:1,flavor:n,workers_ran:ws.length,workers_named:ws.length,collected:10,executed:10,failed:1,
      errored:0,skipped:0,duration_seconds:100}]},
    queue:{queue:[{run_id:10,lane:'CPU',median_wait_seconds:120}]},
    flaky:{rescues,rescue_tests:tests}};
}
async function reader(P){
  for(const n of ALL_NAMES)for(const workers of [false,true]){
    const tag=`reader, ${workers?'a flavor on two workers':'a flavor with no worker number'}, name ${n}`;
    const ids=workers?[`${n}#1`,`${n}#2`]:[n];
    const bad=[];
    let m=null;
    try{m=P.w.adaptViews(madeUp(n,workers))}catch(e){bad.push('threw: '+String(e&&e.message||e).split('\n')[0])}
    if(m){
      const has=(map,k,what,v)=>{if(!own(map,k)||(v!==undefined&&JSON.stringify(map[k])!==JSON.stringify(v)))bad.push(what)};
      if(JSON.stringify(m.jobs.map(j=>j.id))!==JSON.stringify(ids))bad.push('job ids');
      ids.forEach(id=>{
        has(m.labels,id,'machine label '+id,'machine-1');
        has(m.counts,id,'test counts '+id);
        has(m.commits[0].o,id,'minutes '+id);
      });
      if(workers)has(m.wkGroups,n,'worker group');else if(own(m.wkGroups,n))bad.push('worker group of one');
      if(!m.thCategories.some(c=>c.id===n))bad.push('suite health');
      const named=m.rescues.find(r=>r.id===ids[0]);
      if(!named||named.events.length!==1)bad.push('rescue of the flavor');
      if(!m.flaky.some(f=>f.test===n&&f.job===ids[0]))bad.push('flaky test of the flavor');
      if(workers){
        const bare=m.rescues.find(r=>r.name===n);
        if(!bare||bare.fails.length!==1)bad.push('failure of the job with no flavor');
        if(!bare||!m.flaky.some(f=>f.test==='test_b'&&f.job===bare.id))bad.push('flaky test of the job with no flavor');
      }
    }
    const dirty=Object.keys(P.w.Object.prototype);
    if(dirty.length)bad.push('Object.prototype got '+dirty.slice(0,3).join(', '));
    ok(!bad.length,`${tag}: every map keeps the rows under their own name`+(bad.length?` - ${bad.slice(0,4).join(', ')}`:''));
  }
}

(async()=>{
  // The page with the original names comes first. It is the yardstick for the
  // others, and it is kept open for the view-reader block at the end.
  ok(!!(MULTI&&SINGLE),'the files have a flavor that ran on several workers and one with no worker number'+(MULTI&&SINGLE?` (${MULTI}, ${SINGLE})`:''));
  ok(jobs.length===2,`the files have two jobs with no flavor that failed lately${jobs.length===2?` (${jobs.map(j=>j[0]).join(', ')})`:''}`);
  const first=await openPage(parsed());
  BASE.bars=barsOf(first);BASE.snap=snapshot(first.w);
  await verify(first,'original names',{});
  for(const sc of scenarios){
    const f=parsed();sc.edit(f);
    const P=await openPage(f);
    try{await verify(P,sc.tag,sc)}
    catch(e){ok(false,`${sc.tag}: the check itself threw: ${String(e&&e.stack||e).split('\n').slice(0,2).join(' | ')}`)}
    finally{P.w.close()}
  }
  await reader(first);
  first.w.close();
  console.log(`\n${pass} passed, ${fail} failed`);
  process.exit(fail?1:0);
})().catch(e=>{
  ok(false,'the check stopped early: '+String(e&&e.stack||e).split('\n').slice(0,3).join(' | '));
  console.log(`\n${pass} passed, ${fail} failed`);
  process.exit(1);
});
