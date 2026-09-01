// Guards the 2026-09-01 alignment of the mock with the real ci_pipeline test flavors / jobs / runner labels.
const fs=require('fs');const {JSDOM}=require('jsdom');
const html=fs.readFileSync(require('path').join(__dirname,'..','index.html'),'utf-8');
const dom=new JSDOM(html,{runScripts:'dangerously',pretendToBeVisual:true});
const w=dom.window,d=w.document;
let pass=0,fail=0;
function ok(name,cond){if(cond){pass++;console.log('  ok -',name)}else{fail++;console.log('  FAIL -',name)}}
const vis=root=>[...root.querySelectorAll('*:not(script)')].filter(n=>n.children.length===0).map(n=>n.textContent).join('\n');
setTimeout(()=>{
  const TH=w.eval('TH_CATEGORIES'),JOBS=w.eval('JOBS'),MACHINES=w.eval('MACHINES'),STEPS=w.eval('STEPS'),THIW=w.eval('THI_WORKERS'),COMMITS=w.eval('COMMITS');
  const FLAVORS=['tpu-unit','tpu-integration','tpu-post-training-unit','tpu-post-training-integration','gpu-unit','gpu-integration','cpu-unit','cpu-integration','cpu-post-training-unit','cpu-post-training-integration','decoupled'];
  ok('11 suites named exactly by flavor id',TH.length===11&&FLAVORS.every(f=>TH.some(c=>c.name===f)));
  const wk={'tpu-pretrain-unit':2,'cpu-pretrain':4,'cpu-posttrain':4};
  ok('workers per suite follow the coordinator constants (4/4/2, else 1)',TH.every(c=>(c.workers||1)===(wk[c.id]||1))&&TH.every(c=>THIW[c.id]===(wk[c.id]||1)));
  const dec=TH.find(c=>c.id==='decoupled');
  ok('decoupled is a CPU suite nested in cpu-unit with 50 tests',!!dec&&dec.device==='CPU'&&dec.nested==='cpu-unit'&&dec.tag==='inside cpu-unit'&&/50 tests/.test(dec.note)&&dec.data[29].count===50);
  ok('every suite carries its Actions job group',TH.every(c=>typeof c.group==='string'&&/Tests/.test(c.group)));
  // line colours: first suite of a device = the device hex itself, later suites lighter, never darker (user 2026-09-01: tpu-unit line was too dark)
  const co=w.__thc.colorOf;const lum=h=>[1,3,5].map(v=>parseInt(h.slice(v,v+2),16)).reduce((a,b)=>a+b,0);
  ok('tpu-unit / gpu-unit / cpu-unit lines use the device colours',co['tpu-pretrain-unit'].toLowerCase()==='#3987e5'&&co['gpu-unit'].toLowerCase()==='#f97316'&&co['cpu-pretrain'].toLowerCase()==='#22c55e');
  ok('no suite line is darker than its device colour',TH.every(c=>lum(co[c.id])>=lum({TPU:'#3987E5',GPU:'#F97316',CPU:'#22C55E'}[c.device])));
  // legend + footer + dropdown
  const rows=[...d.querySelectorAll('#thCharts [data-throw]')];
  ok('legend lists 11 rows, decoupled tagged',rows.length===11&&rows.some(r=>r.textContent.includes('decoupled')&&r.textContent.includes('(inside cpu-unit)')));
  const foot=d.querySelector('#thCharts [data-thflavors]');
  ok('flavor footer names the missing flavors and the decoupled rule',!!foot&&/tpu7x-unit/.test(foot.textContent)&&/not started by a pull request/.test(foot.textContent)&&/Pathways/.test(foot.textContent)&&/50 tests/.test(foot.textContent));
  ok('dropdown option shows the tag',[...d.querySelectorAll('#thTabs option')].some(o=>o.textContent.startsWith('decoupled (inside cpu-unit)')));
  const guide=((d.querySelector('#guide-th')||{}).textContent||'')+d.querySelector('[data-guide="th"]').textContent;  // initGuides moves the .g-more text into the #guide-th panel
  ok('section guide lists every flavor + decoupled_target + tpu7x + Pathways',FLAVORS.every(f=>guide.includes(f))&&guide.includes('decoupled_target')&&guide.includes('tpu7x-post-training-unit')&&guide.includes('Pathways'));
  // modal note + classic panel note
  w.openTHDetail('decoupled',29);
  ok('suite modal carries the decoupled note',!!d.querySelector('#modal [data-thnote]')&&/worker 1/.test(d.querySelector('#modal [data-thnote]').textContent));
  w.closeModal&&w.closeModal();
  w.setTHSel('decoupled');
  ok('per-suite panel shows the tag in its title and the note under it',/decoupled/.test(d.querySelector('#thCharts .card-title').textContent)&&!!d.querySelector('#thCharts [data-thnote]'));
  w.setTHSel('all');
  // device groups: nested suite adds time, not tests
  w.setTHGroup(true);
  const grp=w.__thc.cats.find(c=>c.id==='grp-CPU');
  const cpuSuites=TH.filter(c=>c.device==='CPU');
  const i0=w.eval('THV.i0'),last=w.eval('THV.n')-1;
  const expCnt=cpuSuites.filter(c=>!c.nested).reduce((a,c)=>a+(c.data[i0+last].count||0),0);
  const expDur=cpuSuites.reduce((a,c)=>a+(c.data[i0+last].dur||0),0);
  ok('CPU group counts tests once (decoupled excluded) but keeps its time',!!grp&&grp.data[last].count===expCnt&&grp.data[last].dur===expDur&&grp.data[last].commitIdx===i0+last);
  w.setTHGroup(false);
  // jobs, labels, pools, steps
  ok('job names are the real Actions names',JOBS.some(j=>j.name==='Gate and Formalize Parameters')&&JOBS.some(j=>j.name==='CPU Posttrain Tests (cpu-post-training-integration)')&&JOBS.some(j=>j.name==='Code Quality Check / Pre-commit Linters')&&!JOBS.some(j=>/post-training-integ\)/.test(j.name)));
  ok('runner labels are runs-on labels, not image names',MACHINES.gpu==='linux-x86-a2-48-a100-4gpu'&&MACHINES.cpu==='linux-x86-n2-32'&&MACHINES.build==='linux-x86-n2-16-buildkit'&&MACHINES.tpu==='linux-x86-ct6e-180-4tpu');
  const body=vis(d.body);
  ok('queue pools use the real labels',body.includes('a2-48-a100-4gpu')&&body.includes('n2-32')&&!body.includes('a100-2x')&&!body.includes('cpu-16'));
  const cpu1=STEPS['cpu-pre-u'];
  ok('cpu-unit worker 1 steps include the decoupled pass; test pcts sum to the cpu template',Array.isArray(cpu1)&&cpu1.some(s=>s.name==='Run Targeted Decoupled Tests'&&s.isTest)&&Math.abs(cpu1.filter(s=>s.isTest).reduce((a,s)=>a+s.pct,0)-STEPS.cpu.filter(s=>s.isTest).reduce((a,s)=>a+s.pct,0))<1e-9);
  ok('step names are the real ones',STEPS.tpu.some(s=>s.name==='Initialize containers'&&s.img==='maxtext-unit-test-tpu:py312')&&STEPS.cpu.some(s=>s.img==='maxtext-unit-test-tpu:py312')&&STEPS.gpu.some(s=>s.img==='maxtext-unit-test-cuda12:py312')&&!JSON.stringify(STEPS).includes('Run pytest'));
  // modal drawer: the two test steps share the job's test minutes
  w.goToCommit(COMMITS[0].hash);
  const row=[...d.querySelectorAll('#modal tr')].find(r=>r.textContent.includes('CPU Pretrain Tests (cpu-unit)'));
  row&&row.click();
  const drawer=row&&row.nextElementSibling;const dt=drawer?drawer.textContent:'';
  const m1=dt.match(/Run Tests\s*([\d.]+)m/),m2=dt.match(/Run Targeted Decoupled Tests\s*([\d.]+)m/);
  const jr=(COMMITS[0].o&&COMMITS[0].o['cpu-pre-u']&&COMMITS[0].o['cpu-pre-u'].r)||JOBS.find(j=>j.id==='cpu-pre-u').b.r;
  ok('drawer: Run Tests + decoupled minutes = the job\'s test minutes',!!m1&&!!m2&&Math.abs(parseFloat(m1[1])+parseFloat(m2[1])-jr)<0.11);
  ok('modal legend discloses the seconds-long jobs the mock omits',d.querySelector('#modal').textContent.includes('Setup Parameters')&&d.querySelector('#modal').textContent.includes('All Required Tests Passed'));
  w.closeModal&&w.closeModal();
  // Pathways jobs publish no test results
  const TC=w.eval('TEST_COUNTS');
  ok('no test counts for the Pathways jobs',!TC['tpu-pw-u1']&&!TC['tpu-pw-u2']&&!TC['tpu-pw-i']&&!!TC['cpu-pre-u']);
  w.goToCommit(COMMITS[0].hash);
  const pw=[...d.querySelectorAll('#modal tr')].find(r=>r.textContent.includes('TPU Pathways Integration Tests'));pw&&pw.click();
  ok('Pathways drawer says there is no test-result file',!!pw&&/No test-result file/.test(pw.nextElementSibling.textContent)&&!/tests total/.test(pw.nextElementSibling.textContent));
  w.closeModal&&w.closeModal();
  // the queue-timeout story is gone
  const c4940=COMMITS.find(c=>c.pr==='#4940');
  w.goToCommit(c4940.hash);
  const mt=d.querySelector('#modal').textContent;
  ok('#4940 reads as cancelled while waiting, never "timed out"',/No runner became free|Cancelled while waiting/.test(mt)&&!/timed out waiting|Queue timeout/.test(mt));
  w.closeModal&&w.closeModal();
  ok('no "timed out waiting" / "Queue timeout" copy anywhere',!/timed out waiting|Queue timeout/.test(vis(d.body)));
  // ---- pass 4 (audit-confirmed facts) ----
  ok('#4940 TPU jobs have no setup time (never got a runner)',Object.values(c4940.o).every(o=>o.s===0&&o.r===0));
  ok('hosted / build jobs are not CPU-lane jobs',JOBS.filter(j=>j.cat==='infra').every(j=>j.lane==='Hosted')&&JOBS.find(j=>j.id==='pkg').lane==='Build'&&w.eval('LANE_COLORS').Hosted&&w.eval('LANE_COLORS').Build);
  w.goToCommit(COMMITS.find(c=>c.pr==='#4908').hash);
  const mh=[...d.querySelectorAll('#modal thead th')].map(x=>x.textContent);
  ok('modal job table has no Docker cache HIT/MISS column',!mh.some(h=>/Docker cache/.test(h))&&!/HIT|MISS/.test(d.querySelector('#modal').textContent)&&d.querySelectorAll('#modal td.cat-head[colspan="9"]').length>=3);
  w.closeModal&&w.closeModal();
  ok('queue pools carry the full runs-on labels',/linux-x86-ct6e-180-4tpu/.test(vis(d.body))&&/linux-x86-a2-48-a100-4gpu/.test(vis(d.body))&&/linux-x86-n2-32/.test(vis(d.body)));
  const rsc=w.eval('RESCUES');
  ok('Pathways rescue cause is the failed step name, no log-derived sentence; no dead runs field',rsc.every(r=>r.runs===undefined)&&rsc.find(r=>r.id==='tpu-pw-i').events.every(e=>/failed at step "/.test(e.cause)));
  ok('legend delta cell has a real title attribute',!!d.querySelector('[data-thdelta][title], span[title^="Change in test count"]'));
  const dec4=TH.find(c=>c.id==='decoupled');
  ok('decoupled is a ~20 s pass, not minutes',dec4.data.filter(x=>x.dur!=null).every(x=>x.dur>=15&&x.dur<=25)&&dec4.maxY===60);
  w.openTHDetail('decoupled',29);
  const dm=d.querySelector('#modal').textContent;
  ok('suite modal names the GitHub job and the nested worker rule',/GitHub job: step "Run Targeted Decoupled Tests" inside CPU Pretrain Tests \(cpu-unit\) \/ Execute Tests \(1\) \/ cpu-unit/.test(dm)&&/Workers: none of its own/.test(dm));
  w.closeModal&&w.closeModal();
  w.openTHDetail('cpu-pretrain',29);
  ok('4-worker suite modal shows Execute Tests (1..4)',/GitHub job: CPU Pretrain Tests \(cpu-unit\) \/ Execute Tests \(1\.\.4\) \/ cpu-unit/.test(d.querySelector('#modal').textContent));
  w.closeModal&&w.closeModal();
  ok('the PR that added decoupled is a cpu-unit change',COMMITS.some(c=>c.pr==='#4902'&&/cpu-unit pass/.test(c.title)&&c.o['cpu-pre-u'].r===9));
  ok('worker guide defines "worker suite"',/A worker suite is one test flavor of the CI workflow/.test(d.querySelector('#guide-worker').textContent));
  ok('TH section guide: one file per worker, only when selected',/one test-result file per parallel worker on every run where the suite was selected/.test(d.querySelector('#guide-th').textContent));
  // ---- audit M1 / M3 ----
  const RID=w.eval('RUN_IDS'),TM=w.eval('TIMES');
  const rids=Object.values(RID);
  ok('run ids look like real GitHub run ids (11 digits), none in the range GitHub has issued',
     rids.length===30&&rids.every(v=>String(v).length===11&&v>34e9));
  const MO={Jul:7,Aug:8},stmp=s=>{const p2=s.split(' '),hm=p2[2].split(':');return ((MO[p2[0]]*100+ +p2[1])*100+ +hm[0])*100+ +hm[1]};
  const realOrder=Object.keys(TM).sort((a,b)=>stmp(TM[a])-stmp(TM[b]));
  ok('run ids rise with merge time (the newest merged pull request has the largest id)',
     realOrder.every((h,i)=>i===0||RID[h]>RID[realOrder[i-1]]));
  ok('every stored commit has a run id below the oldest real one or above, never a 5-digit placeholder',
     COMMITS.every(c=>String(RID[c.hash]).length===11));
  ok('no undefined "concurrent" jargon left in visible copy',!/concurrent runs|runs at once|High concurrent merge/.test(vis(d.body).replace(/More runs at once means[^.]*\./g,'')));
  w.goToCommit(COMMITS[0].hash);
  const cm=d.querySelector('#modal').textContent;
  ok('the modal says how many runs overlapped and prints an 11-digit run id',/\d+ pipeline runs overlapped it/.test(cm)&&/run #\d{11}/.test(cm));
  w.closeModal&&w.closeModal();
  w.initGuides&&w.initGuides();
  ok('the timeline guide defines overlapping runs',/Overlapping runs, in the hover card, counts how many other runs of this pipeline/.test(d.querySelector('#guide-timeline').textContent));
  console.log(`${pass} passed, ${fail} failed`);
},60);
