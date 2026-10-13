// Checks that text from the view files is shown as text, never run as HTML.
// It loads the real view files, adds a marker tag to the free-text values,
// opens every tooltip and pop-up it can, and then looks for the marker tag.
// If the page put a value into innerHTML without escaping it, the marker
// becomes a real element and this check fails.
const fs=require('fs');const path=require('path');const {JSDOM}=require('jsdom');
const {VirtualConsole}=require('jsdom');
const PAGE=require('./loadpage.js').PAGE;
const VIEWS=path.join(path.dirname(PAGE),'views');

let pass=0,fail=0;
function ok(c,m){if(c){pass++;console.log('  ok   '+m)}else{fail++;console.log('  FAIL '+m)}}

if(!fs.existsSync(path.join(VIEWS,'meta.json'))){
  console.log('  skip  no views/ next to the page');console.log('\n0 passed, 0 failed');process.exit(0);
}

// A tag, both kinds of quote and a backslash: each one breaks a different unsafe pattern.
const MARK='<b data-xss="1">x</b>"\'\\';
const TAG='<b data-xss="1">x</b>';

// --- which values get the marker ------------------------------------------
// Free text: what a person wrote or a test printed. Job, step and runner
// names are written in workflow files, so they count as free text too.
const FREE=new Set(['title','author','base_ref','name','job_name','classname','failure_message',
  'failed_step','workflow_name','workflow_path','runner_label','runner_group_name','pool']);
// Ids the collector cuts out of a job name: the flavor is its last " / " part.
// They get the same marker, and one value always becomes the same string, so
// the joins between tables still match. Statuses, dates, months, lanes and
// numbers stay as they are, so the page still renders.
const IDS=new Set(['flavor','suite_id','nested_in']);
// The page shows only part of some values: a job name up to its first " / ",
// the first line of a failure message. So the marker goes on both ends.
const poison=v=>typeof v==='string'?MARK+v+MARK:v;
const BAD_URL='javascript:alert(1)';
function poisonValue(col,v){return FREE.has(col)||IDS.has(col)?poison(v):/url$/.test(col)?BAD_URL:v}
function poisonFile(f){
  const out={};
  for(const[k,v]of Object.entries(f))out[k]=k==='tables'?v:poisonValue(k,v);
  if(f.tables){
    out.tables={};
    for(const[name,t]of Object.entries(f.tables))
      out.tables[name]={...t,rows:t.rows.map(r=>r.map((v,i)=>poisonValue(t.columns[i],v)))};
  }
  return out;
}

// --- every view file: meta.json, the month files and pr/*.json -------------
const raw={};
const readDir=(dir,prefix)=>{
  for(const name of fs.readdirSync(dir))
    if(name.endsWith('.json'))raw[prefix+name]=JSON.parse(fs.readFileSync(path.join(dir,name),'utf8'));
};
readDir(VIEWS,'views/');
if(fs.existsSync(path.join(VIEWS,'pr')))readDir(path.join(VIEWS,'pr'),'views/pr/');

const rowsOf=(f,name)=>{
  const t=f.tables&&f.tables[name];
  return t?t.rows.map(r=>Object.fromEntries(t.columns.map((c,i)=>[c,r[i]]))):[];
};
// The rescue-tests table is empty in today's data, so the page never draws a
// flaky test. Add two rows, tied to the newest real rescue of a merged pull
// request, so that path runs too. Two tests in one run also draw the
// "also failed in the same runs" list.
const flakyFiles=Object.keys(raw).filter(k=>/^views\/flaky-/.test(k));
if(flakyFiles.every(k=>!rowsOf(raw[k],'rescue_tests').length)){
  const merged=new Set(Object.keys(raw).filter(k=>/^views\/runs-/.test(k))
    .flatMap(k=>rowsOf(raw[k],'runs').map(r=>String(r.pr))));
  let best=null,bestFile=null;
  flakyFiles.forEach(k=>rowsOf(raw[k],'rescues').forEach(r=>{
    if(r.rescued&&r.flavor&&merged.has(String(r.pr))&&(!best||r.created_at>best.created_at)){best=r;bestFile=k}
  }));
  ok(!!best,'found a rescued job to hang the synthetic flaky tests on');
  if(best){
    const t=raw[bestFile].tables.rescue_tests;
    ['test_synthetic_a','test_synthetic_b'].forEach(name=>{
      const row={run_id:best.run_id,pr:best.pr,job_name:best.job_name,failed_attempt:best.failed_attempt,
        suite_id:best.flavor,flavor:best.flavor,worker:best.worker,
        classname:'tests.unit.synthetic_test.SyntheticTest',name,status:'failed',
        duration:1.5,failure_message:'AssertionError: synthetic\nsecond line'};
      t.rows.push(t.columns.map(c=>c in row?row[c]:null));
    });
  }
}

const files={};
for(const[k,f]of Object.entries(raw))files[k]=JSON.stringify(poisonFile(f));
// Same fetch stub as check_realdata.js. The files go onto window before the
// page is parsed: written into the page as one script, about 25 MB of JSON
// took seconds to parse.
const stub=`<script>
window.fetch=function(url){
  const body=window.__FILES__[String(url)];
  return Promise.resolve({ok:body!==undefined,status:body===undefined?404:200,
    json:function(){return Promise.resolve(JSON.parse(body))}});
};
</script>`;

// A broken inline handler (onclick="...") does not throw in this script. jsdom
// reports it here instead, so collect those errors and check them every step.
const errors=[];
const vc=new VirtualConsole();
vc.on('jsdomError',e=>errors.push(String(e&&e.message||e).split('\n')[0]));

const html=fs.readFileSync(PAGE,'utf8').replace('<script>',stub+'\n<script>');
const dom=new JSDOM(html,{runScripts:'dangerously',pretendToBeVisual:true,url:'http://localhost/',virtualConsole:vc,
  beforeParse(win){win.__FILES__=files}});
const w=dom.window,d=w.document;

const $=s=>d.querySelector(s),$$=s=>[...d.querySelectorAll(s)];
const tip=$('#tip'),pop=$('#wkpop-body'),modal=$('#modal');
const wait=ms=>new Promise(r=>setTimeout(r,ms));
const fire=(el,type)=>el.dispatchEvent(new w.MouseEvent(type,{bubbles:true,cancelable:true,clientX:300,clientY:200}));
const hover=el=>{fire(el,'mouseover');fire(el,'mouseenter');fire(el,'mousemove')};
const shows=el=>el.textContent.includes(TAG);
const popOpen=()=>$('#wkpop').style.display==='flex';
const modalOpen=()=>$('#overlay').classList.contains('on');
const closeModal=()=>fire($('#modal .modal-close'),'click');
const loadFailed=()=>$$('.ct').some(e=>/could not load its data/.test(e.textContent));
// A loop stops at the first bad state in the part of the page it redraws, so
// the checks after the step report it. Looking only at that part keeps a loop
// of a few hundred hovers fast.
const broken=(el=d)=>!!(el.querySelector('[data-xss]')||el.querySelector('a[href^="javascript:"]')||errors.length);
// A value that breaks out of a quoted attribute leaves attributes with odd
// names behind, like 1" from data-xss="1". No real attribute has one.
const oddAttr=()=>$$('*').find(e=>[...e.attributes].some(a=>!/^[a-z_:][-a-z0-9_:.]*$/i.test(a.name)));

// Runs one render step, then looks at the whole page.
async function run(step,fn){
  try{fn()}catch(e){ok(false,step+' threw: '+e.message)}
  await wait(0);   // let the page's MutationObserver run
  ok(d.querySelectorAll('[data-xss]').length===0, 'no marker tag became an element ('+step+')');
  ok(!d.querySelector('a[href^="javascript:"]'), 'no link runs code ('+step+')');
  ok(!oddAttr(),'no value broke out of an attribute ('+step+')');
  ok(!errors.length,'no script error ('+step+')'+(errors.length?': '+errors.splice(0).slice(0,3).join(' | '):''));
}

(async()=>{
  // boot() is async: wait until it has drawn the page, or given up.
  for(let i=0;i<500&&!$('#timeline svg')&&!loadFailed();i++)await wait(20);

  await run('first render',()=>{
    ok(!loadFailed(),'the page loaded the poisoned view files');
    ok(!!$('#timeline svg')&&!!$('#devlines svg')&&!!$('#worker-chart svg'),'the three charts drew');
    ok(shows($('#flaky')),'the flaky card shows a job name as text');
    ok(w.eval('JOBS').some(j=>j.id.includes(TAG)),'job ids carry the marker, so the inline handlers get it too');
  });
  // One device is enough: GPU and CPU redraw the same way TPU does.
  await run('device filter',()=>{
    for(const hw of ['tpu','all']){$(`.fchip[data-hw="${hw}"]`).click();if(broken())break}
  });

  // Every stored run, so every bar and every cell below is drawn.
  await run('90-day range',()=>{const s=$('#date-range');s.value='90';s.dispatchEvent(new w.Event('change'))});
  const runs=w.eval('getWindows().current');
  // One run of each outcome. Every run draws its details the same way, so
  // opening all of them only makes the check slow.
  const sample=['pass','fail','flaky'].map(s=>runs.find(c=>c.status===s)).filter(Boolean);

  // --- run time per merged pull request ------------------------------------
  await run('run-time chart tooltips',()=>{
    const bars=$$('#timeline rect[fill="transparent"][cursor="pointer"]');
    ok(bars.length===runs.length,'the run-time chart has a bar for each of the '+runs.length+' runs');
    for(const b of bars){hover(b);if(broken(tip))break}
    ok(shows(tip),'the run-time tooltip shows the title as text');
  });
  await run('run-time chart marks',()=>{
    for(const m of $$('#timeline circle[cursor="pointer"]')){hover(m);if(broken(tip))break}
  });
  await run('run details',()=>{
    // The bars run oldest to newest, so the newest run is the last bar.
    const bars=$$('#timeline rect[fill="transparent"][cursor="pointer"]');
    let opened=0;
    for(const c of sample){
      fire(bars[runs.length-1-runs.indexOf(c)],'click');
      if(modalOpen()&&shows(modal)&&modal.querySelector('h2').textContent.startsWith(c.pr+' '))opened++;
      if(broken(modal))break;
      closeModal();
    }
    ok(sample.length>0&&opened===sample.length,
       `a bar opened its own run details, with the title as text (${opened} of ${sample.length}: ${sample.map(c=>c.status).join(', ')})`);
  });
  await run('run details rows and sorting',()=>{
    fire($$('#timeline rect[fill="transparent"][cursor="pointer"]')[runs.length-1],'click');
    // The job rows open on an inline onclick that carries the row id.
    const rows=$$('#modal tr[onclick]');
    rows.forEach(r=>fire(r,'click'));
    ok(rows.length>0&&$$('#modal [id^="xrow-"]').every(x=>x.style.display==='block'),
       'every job row of the run details opened its drawer ('+rows.length+')');
    for(const h of $$('#modal .jth')){fire(h,'click');fire(h,'click')}
    ok(shows(modal),'the sorted job table still shows the job names as text');
    closeModal();
  });
  await run('run-time chart zoomed',()=>{w.tlZoomTo(0,3);ok(!!$('#timeline svg'),'the zoomed chart drew')});
  await run('run-time chart reset',()=>{w.tlZoomReset()});

  // --- test run time by device ---------------------------------------------
  await run('device chart tooltips',()=>{
    const cols=$$('#devlines rect[fill="transparent"][cursor="pointer"]');
    ok(cols.length===runs.length,'the device chart has a point for each of the '+runs.length+' runs');
    for(const c of cols){hover(c);if(broken(tip))break}
    ok(shows(tip),'the device tooltip shows the title and job names as text');
    fire(cols[0],'click');
    ok(modalOpen()&&shows(modal),'a click on a point opened its run details');
    closeModal();
  });
  await run('device chart phases',()=>{
    const n=$$('#devlines button.fchip').length;
    for(let i=0;i<n;i++){$$('#devlines button.fchip')[i].click();if(broken())break}
  });
  await run('device summary rows',()=>{
    // Each row opens on an inline onclick that carries the run id.
    let opened=0;const rows=$$('#devlines .sum-row[data-sumrow="dev"]');
    for(const r of rows){r.click();if(modalOpen()&&shows(modal))opened++;closeModal();if(broken(modal))break}
    ok(rows.length>0&&opened===rows.length,`every device summary row opened its run (${opened} of ${rows.length})`);
  });
  await run('device chart zoomed',()=>{w.devZoomTo(0,3);ok(!!$('#devlines svg'),'the zoomed chart drew')});
  await run('device chart reset',()=>{w.devZoomReset()});

  // --- worker time per merged pull request ---------------------------------
  await run('worker chart tooltips',()=>{
    const segs=$$('#worker-chart [cursor="pointer"]');
    ok(segs.length>0,'the worker chart has '+segs.length+' bars to hover');
    for(const s of segs){hover(s);if(broken(tip))break}
    ok(shows(tip),'the lane tooltip shows a job name as text');
  });
  await run('worker pop-up',()=>{
    fire($('#worker-chart [cursor="pointer"]'),'click');
    ok(popOpen()&&shows(pop),'a click on a bar opened the worker pop-up with the title as text');
    $$('#wkpop-body button').find(b=>/full commit details/.test(b.textContent)).click();
    ok(!popOpen()&&modalOpen()&&shows(modal),'its button opened the run details');
    closeModal();
  });
  await run('worker chart, one suite at a time',()=>{
    const ids=$$('#wk-chips select option').map(o=>o.value).filter(v=>v!=='all');
    ok(ids.some(v=>v.includes(TAG)),'some suite ids in the dropdown carry the marker');
    // Inside a <select> the parser drops tags instead of making elements, so
    // check the text itself.
    ok($$('#wk-chips select option').some(o=>shows(o)),'the dropdown shows the suite names as text');
    let picked=0;
    for(const id of ids){
      const sel=$('#wk-chips select');sel.value=id;sel.dispatchEvent(new w.Event('change'));
      if(w.eval('WK_SEL')===id)picked++;
      // The newest run's bars come first, one per worker: enough to show every
      // worker's machine. The older runs draw the same way.
      for(const s of $$('#worker-chart [cursor="pointer"]').slice(0,8)){hover(s);if(broken(tip))break}
      if(broken())break;
    }
    ok(picked===ids.length,`the dropdown handed the page every suite id it showed (${picked} of ${ids.length})`);
    ok(shows(tip),'the worker tooltip shows the suite name as text');
  });
  await run('worker suite pop-up',()=>{
    const id=$$('#wk-chips select option').map(o=>o.value).find(v=>v.includes(TAG));
    const sel=$('#wk-chips select');sel.value=id;sel.dispatchEvent(new w.Event('change'));
    fire($('#worker-chart [cursor="pointer"]'),'click');
    ok(popOpen()&&shows(pop),'a click on a worker bar opened the suite pop-up with the names as text');
    $$('#wkpop-body button').find(b=>/Every suite in this run/.test(b.textContent)).click();
    ok(popOpen()&&/Every worker suite in this run/.test(pop.textContent),'its first button opened every suite of the run');
    fire($('#worker-chart [cursor="pointer"]'),'click');
    $$('#wkpop-body button').find(b=>/full commit details/.test(b.textContent)).click();
    ok(modalOpen()&&shows(modal),'its second button opened the run details');
    closeModal();
    $$('#wk-chips button').find(b=>/Back to all suites/.test(b.textContent)).click();
    ok(w.eval('WK_SEL')==='all','the back button returned to all suites');
  });

  // --- flaky jobs and tests ------------------------------------------------
  await run('flaky cell tooltips',()=>{
    // Every amber or outlined cell, and one plain cell per job: the plain
    // cells all draw the same way. Each opens on an inline onmousemove that
    // carries the job id.
    const marked=c=>/var\(--warn\)|var\(--crit\)/.test(c.getAttribute('style'));
    const cells=$$('#flaky .issue-row[data-flane]').flatMap(r=>{
      const all=[...r.querySelectorAll('[data-cell]')];
      return all.filter(marked).concat(all.filter(c=>!marked(c)).slice(0,1));
    });
    ok(cells.length>0,'the flaky card has '+cells.length+' cells to hover');
    for(const c of cells){fire(c,'mousemove');if(broken(tip))break}
    ok(shows(tip),'the flaky cell tooltip shows the job name as text');
    // The cell of a run with a known failed test lists that test by name.
    const ft=w.eval('FLAKY_TESTS')[0],idx=w.eval('thIdxOf')(ft.prs[0]);
    const cell=$$('#flaky [data-cell]').find(c=>c.getAttribute('data-fljob')===ft.job&&+c.getAttribute('data-flidx')===idx);
    if(cell)fire(cell,'mousemove');
    ok(!!cell&&tip.textContent.includes(ft.test),'the cell tooltip lists the failed test by name, as text');
  });
  await run('flaky cell pop-ups',()=>{
    // Each cell opens on an inline onclick that carries the job id and the pull request.
    let opened=0;const rows=$$('#flaky .issue-row[data-flane]');
    for(const r of rows){
      const c=r.querySelector('[data-cell][style*="var(--warn)"]')||r.querySelector('[data-cell]');
      w.closeWkPop();c.click();if(popOpen()&&shows(pop))opened++;if(broken(pop))break;
    }
    ok(rows.length>0&&opened===rows.length,`a cell in every job row opened its pop-up (${opened} of ${rows.length})`);
  });
  await run('flaky job pop-ups',()=>{
    let opened=0;const rows=$$('#flaky .issue-row[data-flane]');
    for(const r of rows){w.closeWkPop();r.click();if(popOpen()&&shows(pop))opened++;if(broken(pop))break}
    ok(rows.length>0&&opened===rows.length,`every job row opened its pop-up (${opened} of ${rows.length})`);
  });
  await run('flaky test rows',()=>{
    let lit=0,opened=0;const rows=$$('#flaky .sum-row[data-sumrow="flaky"]');
    for(const r of rows){
      fire(r,'mouseenter');   // inline onmouseenter: lights up the cells of that test
      if($$('#flaky [data-cell]').some(c=>c.style.outline))lit++;
      fire(r,'mouseleave');
      w.closeWkPop();r.click();if(popOpen()&&shows(pop))opened++;
      if(broken(pop))break;
    }
    ok(rows.length>0&&lit===rows.length,`every flaky test row lit up its cells (${lit} of ${rows.length})`);
    ok(rows.length>0&&opened===rows.length,`every flaky test row opened its pop-up (${opened} of ${rows.length})`);
    const b=$$('#wkpop-body button').find(x=>/Every re-run of/.test(x.textContent));
    if(b)b.click();
    ok(!!b&&popOpen()&&/lane/.test(pop.textContent)&&shows(pop),'the test pop-up button opened the job pop-up');
    const more=$('#flaky a[data-flmore]');if(more)more.click();
  });

  // --- the global openers, called with ids taken from the data -------------
  await run('openFlakyPop by id',()=>{
    // Each job on its own, then with its first re-run and its first failure.
    let n=0,good=0;
    for(const j of w.eval('RESCUES')){
      const prs=[(j.events[0]||{}).pr,((j.fails||[])[0]||{}).pr].filter(Boolean);
      for(const pr of [undefined,...prs]){n++;w.closeWkPop();w.openFlakyPop(j.id,pr);if(popOpen()&&shows(pop))good++;if(broken(pop))break}
      if(broken(pop))break;
    }
    ok(n>0&&good===n,`openFlakyPop showed the job name as text every time (${good} of ${n})`);
  });
  await run('openFlakyTestPop by id',()=>{
    let n=0,good=0;
    const tests=w.eval('FLAKY_TESTS');
    for(const t of tests){n++;w.closeWkPop();w.openFlakyTestPop(t.test,t.job);if(popOpen()&&shows(pop))good++;if(broken(pop))break}
    ok(n>0&&good===n,`openFlakyTestPop showed the test name as text every time (${good} of ${n})`);
    // Two tests of one job that failed in the same run: each pop-up names the other.
    const a=tests.find(t=>tests.some(u=>u!==t&&u.job===t.job&&u.prs.some(p=>t.prs.includes(p))));
    if(a){
      const b=tests.find(u=>u!==a&&u.job===a.job&&u.prs.some(p=>a.prs.includes(p)));
      w.openFlakyTestPop(a.test,a.job);
      ok(pop.textContent.includes(b.test),'the test pop-up lists the other test of the same run, as text');
    }else ok(true,'(no two flaky tests share a run, so the "also failed" list has nothing to draw)');
  });
  await run('openWkPop by id',()=>{
    let n=0,good=0;
    for(const c of w.eval('COMMITS')){n++;w.closeWkPop();w.openWkPop(c.hash);if(popOpen()&&shows(pop))good++;if(broken(pop))break}
    ok(n>0&&good===n,`openWkPop showed the title as text every time (${good} of ${n})`);
  });
  await run('openWkSuitePop by id',()=>{
    let n=0,good=0;const c=runs[0];
    for(const e of w.eval('wkEntities()')){n++;w.closeWkPop();w.openWkSuitePop(c.hash,e.gid);if(popOpen()&&shows(pop))good++;if(broken(pop))break}
    ok(n>0&&good===n,`openWkSuitePop showed the names as text every time (${good} of ${n})`);
    w.closeWkPop();
  });
  await run('openModal by id',()=>{
    let n=0,good=0;
    for(const c of sample){n++;w.goToCommit(c.hash);if(modalOpen()&&shows(modal))good++;closeModal();if(broken(modal))break}
    ok(n>0&&good===n,`goToCommit showed the title as text every time (${good} of ${n})`);
    w.inspectPR(runs[0].hash);
    ok(modalOpen()&&shows(modal),'inspectPR opened the run details too');
    closeModal();
  });
  await run('guides',()=>{for(const b of $$('.gbtn'))b.click()});

  // Leave the scripts out: their source is not text the page shows.
  $$('script').forEach(s=>s.remove());
  ok(d.body.textContent.includes('<b data-xss="1">x</b>'), 'the marker shows as plain text somewhere');
  console.log(`\n${pass} passed, ${fail} failed`);
  process.exit(fail?1:0);
})();
