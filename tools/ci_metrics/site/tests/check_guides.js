// Card guides: visible lead line + "How to read" panel (2026-08-31 part G).
const fs=require('fs');const {JSDOM}=require('jsdom');
const html=fs.readFileSync(require('path').join(__dirname,'..','index.html'),'utf-8');
const dom=new JSDOM(html,{runScripts:'dangerously',pretendToBeVisual:true});
const w=dom.window,d=w.document;
let pass=0,fail=0;
function ok(name,cond){if(cond){pass++;console.log('  ok -',name)}else{fail++;console.log('  FAIL -',name)}}
const txt=n=>n?n.textContent.replace(/\s+/g,' ').trim():'';
const panel=k=>d.getElementById('guide-'+k),btn=k=>d.querySelector(`[data-gbtn="${k}"]`),lead=k=>d.querySelector(`[data-guide="${k}"]`);
setTimeout(()=>{
  const keys=[...d.querySelectorAll('[data-guide]')].map(n=>n.getAttribute('data-guide'));
  ok('main-page guides registered',['timeline','worker','flaky','th','prlist','th-combined','thi'].every(k=>keys.includes(k)));
  ok('every guide has a panel and a button',keys.every(k=>panel(k)&&btn(k)));
  ok('all panels start closed',keys.every(k=>panel(k).hidden===true&&btn(k).getAttribute('aria-expanded')==='false'));
  ok('lead lines open with the purpose sentence and stay under 300 chars (boxes with data-guide-kind may show a fact line or nothing)',keys.every(k=>(lead(k).hasAttribute('data-guide-kind')&&txt(lead(k)).length<=300)||(/^This (chart|section|table|list|card) shows/.test(txt(lead(k)))&&txt(lead(k)).length<=300)));
  ok('no .g-more span left inline',d.querySelectorAll('.g-more').length===0);
  ok('button sits in the title row / title line',btn('thi').parentElement.classList.contains('card-title-row')&&btn('timeline').parentElement.classList.contains('ct'));
  ok('button says How to read and points at its panel',btn('thi').textContent.includes('How to read')&&btn('thi').getAttribute('aria-controls')==='guide-thi'&&btn('thi').tagName==='BUTTON');
  // caveats stay visible in the lead
  ok('bar-card leads drop the newest-left claim; panels state oldest -> newest',!txt(lead('timeline')).includes('right to left')&&!txt(lead('worker')).includes('right to left')&&txt(panel('timeline')).includes('oldest merged pull request on the left to the newest on the right')&&txt(panel('worker')).includes('oldest merged pull request on the left to the newest on the right'));
  ok('totals caveat stays visible on the totals card',txt(lead('thi')).includes('totals across suites, not one suite'));
  ok('flaky definition stays visible',txt(lead('flaky')).includes('fail on a first attempt and pass when re-run'));
  ok('final-commit-only caveat stays visible on the PR list',txt(lead('prlist')).includes('earlier pushes are not tracked'));
  // moved text lives in the panels
  ok('merge-date note lives in the timeline panel, no UTC anywhere',txt(panel('timeline')).includes('its merge date is under it')&&!d.body.textContent.includes('UTC'));
  ok('hover/click hints moved into the worker panel',txt(panel('worker')).includes('Hover a slice to follow that suite')&&txt(panel('worker')).includes('Pick one suite to see one bar per worker'));
  ok('panel heading names the kind',txt(panel('thi')).startsWith('How to read this chart')&&txt(panel('th')).startsWith('How to read this section')&&txt(panel('prlist')).startsWith('How to read this list'));
  ok('flaky guide sits on the Re-runs per job sub-heading, not the card title',btn('flaky').parentElement.classList.contains('gtitle')&&btn('flaky').parentElement.textContent.includes('Re-runs per job')&&![...d.querySelectorAll('.ct')].find(n=>n.textContent.includes('Flaky jobs and tests')).querySelector('.gbtn'));
  ok('flaky lead + panel moved intact, panel explains the dated axis',txt(lead('flaky')).startsWith('This chart shows which jobs are flaky')&&txt(panel('flaky')).includes('Hover a cell to see which tests failed in that run')&&txt(panel('flaky')).includes('merge date'));
  ok('worker panel keeps the sentence order otherwise',txt(panel('worker')).indexOf('Every bar is one merged pull request')<txt(panel('worker')).indexOf('The chart above splits the same runs by phase'));
  // toggle
  btn('thi').click();
  ok('click opens the panel and relabels the button',panel('thi').hidden===false&&btn('thi').textContent.includes('Hide guide')&&btn('thi').getAttribute('aria-expanded')==='true');
  ok('other panels stay closed',panel('timeline').hidden===true&&panel('th-combined').hidden===true);
  const chip=v=>{const b=[...d.querySelectorAll('.fchip')].find(x=>x.dataset&&x.dataset.hw===v);if(b)b.click();return !!b};
  ok('tpu chip exists',chip('tpu'));
  setTimeout(()=>{
    ok('open state survives the re-render a filter click triggers',!!panel('thi')&&panel('thi').hidden===false&&btn('thi').textContent.includes('Hide guide'));
    ok('exactly one button per guide after the re-render',d.querySelectorAll('[data-gbtn="thi"]').length===1&&d.querySelectorAll('#guide-thi').length===1);
    btn('thi').click();
    ok('second click closes it',panel('thi').hidden===true&&btn('thi').textContent.includes('How to read'));
    chip('all');
    w.inspectPR('4920');
    setTimeout(()=>{
      ok('PR page guides registered (tests, jobs, errors)',['pr-tests','pr-jobs','pr-errors'].every(k=>panel(k)&&btn(k)));
      ok('tests lead keeps grouped-by-job; panel keeps the sort rules',txt(lead('pr-tests')).includes('grouped by job')&&txt(panel('pr-tests')).includes('Click the Test header to restore the default order'));
      ok('jobs lead keeps the attempt-1 caveat; panel keeps the column meanings',txt(lead('pr-jobs')).includes('Timing columns come from attempt 1')&&txt(panel('pr-jobs')).includes('Wait is time spent holding no runner'));
      w.setTHSel('tpu-pretrain-unit');
      setTimeout(()=>{
        ok('per-suite chart gets its own guide',!!panel('th-suite')&&txt(panel('th-suite')).includes('The solid line is total duration'));
        w.setTHSel('all');
        const noBtn=['Issues detected','CI phase breakdown','Runner queue latency by pool'].every(t=>{const ct=[...d.querySelectorAll('.ct')].find(n=>n.textContent.includes(t));return ct&&!ct.querySelector('.gbtn')});
        ok('one-sentence descriptions keep no button',noBtn);
        ok('fourteen guides stay mounted once the per-suite card is dismissed',new Set([...d.querySelectorAll('[data-guide]')].map(n=>n.getAttribute('data-guide'))).size===14);
        w.initGuides();  // setTHSel('all') just re-rendered; guides re-attach on a microtask
        ok('TH section: no lead, guide describes the parts',txt(lead('th'))===''&&txt(panel('th')).startsWith('How to read this section')&&txt(panel('th')).includes('This section has three parts')&&!txt(panel('th')).includes('Hover a commit'));
        ok('TH summary: counts visible, rules in its own guide, no hover claim',txt(lead('th-sum')).startsWith('Of the 10 suites in this view (plus the decoupled pass inside cpu-unit)')&&txt(panel('th-sum')).includes('Click a suite to open its details')&&!txt(panel('th-sum')).toLowerCase().includes('hover')&&btn('th-sum').parentElement.classList.contains('sum-h'));
        ok('chart guide says the legend is above the chart',txt(panel('th-combined')).includes('In the legend above the chart')&&txt(panel('th-combined')).includes('zoom the chart to that suite alone'));
        ok('Flakiest tests: lead moved into a How to read this table panel on the box title',btn('flaky-sum').parentElement.classList.contains('sum-h')&&btn('flaky-sum').parentElement.textContent.includes('Flakiest tests')&&txt(lead('flaky-sum'))===''&&txt(panel('flaky-sum')).startsWith('How to read this table')&&panel('flaky-sum').hidden===true);
        ok('device summary: whole lead moved into a How to read this table panel on the block title',btn('dev-sum').parentElement.classList.contains('sum-h')&&btn('dev-sum').parentElement.textContent.includes('Largest changes by device')&&txt(lead('dev-sum'))===''&&txt(panel('dev-sum')).startsWith('How to read this table')&&txt(panel('dev-sum')).includes('One row per device: TPU, GPU and CPU.')&&panel('dev-sum').hidden===true);
        ok('device card guide button sits on its .ct title (parent-sibling lookup)',btn('devlines').parentElement.classList.contains('ct')&&btn('devlines').parentElement.textContent.includes('Test run time by device'));
        w.setDevPhase('queue');
        ok('device lead follows the phase chip',txt(lead('devlines')).includes('spent waiting for a free runner on each merged pull request'));
        w.setDevPhase('run');
        ok('no percent sign leaked into any panel',[...d.querySelectorAll('.card-guide')].every(p=>!p.textContent.includes('%')));
        console.log(`\n${pass} passed, ${fail} failed`);
        process.exit(fail?1:0);
      },60);
    },60);
  },60);
},400);
