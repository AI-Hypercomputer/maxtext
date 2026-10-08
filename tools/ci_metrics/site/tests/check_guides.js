// The "How to read" mechanism on the four charts: every description keeps a visible lead and
// moves its detail into a panel behind a button. Checks registration, the closed default, the
// toggle and its aria state, that an open panel survives the re-render a filter click causes,
// and that the caveats which must stay visible were not swept into a panel.
const fs=require('fs');const {JSDOM}=require('jsdom');
const dom=new JSDOM(require('./loadpage.js')(),{runScripts:'dangerously',pretendToBeVisual:true});
const w=dom.window,d=w.document;
let pass=0,fail=0;
const ok=(n,c)=>{if(c){pass++;console.log('  ok - '+n)}else{fail++;console.log('  FAIL - '+n)}};
const txt=n=>n?n.textContent.replace(/\s+/g,' ').trim():'';
const panel=k=>d.getElementById('guide-'+k);
const btn=k=>d.querySelector(`[data-gbtn="${k}"]`);
const lead=k=>d.querySelector(`[data-guide="${k}"]`);
const KEYS=['timeline','worker','devlines','flaky'];

setTimeout(()=>{
  ok('a guide is registered for each of the four charts',KEYS.every(k=>!!lead(k)));
  ok('every guide has a panel and a button',KEYS.every(k=>panel(k)&&btn(k)));
  ok('all panels start closed',KEYS.every(k=>panel(k).hidden===true&&btn(k).getAttribute('aria-expanded')==='false'));
  ok('no .g-more span left inline',d.querySelectorAll('.g-more').length===0);
  ok('every lead opens with a purpose sentence',KEYS.every(k=>/^This (chart|section|list|table)/.test(txt(lead(k)))));
  ok('leads stay short',KEYS.every(k=>txt(lead(k)).length<420));
  ok('panel heading names the kind',KEYS.every(k=>txt(panel(k)).startsWith('How to read this')));
  ok('button says How to read and points at its panel',
     btn('timeline').textContent.includes('How to read')&&btn('timeline').getAttribute('aria-controls')==='guide-timeline');

  // caveats that must stay where a reader sees them without opening anything
  ok('flaky definition stays visible',txt(lead('flaky')).includes('fail on a first attempt'));
  ok('bar-card leads make no newest-left claim; the panels state the direction',
     !txt(lead('timeline')).includes('newest merged pull request is on the left')&&
     txt(panel('timeline')).includes('oldest merged pull request on the left'));
  ok('merge-date note lives in the timeline panel',txt(panel('timeline')).includes('pull request number under each bar'));
  ok('hover and click hints moved into the worker panel',txt(panel('worker')).includes('Hover a lane'));
  ok('flaky guide sits on the Re-runs per job sub-heading, not the card title',
     btn('flaky').parentElement.classList.contains('gtitle'));

  // toggle
  btn('worker').click();
  ok('click opens the panel and relabels the button',
     panel('worker').hidden===false&&btn('worker').textContent.includes('Hide guide')&&
     btn('worker').getAttribute('aria-expanded')==='true');
  ok('other panels stay closed',panel('timeline').hidden===true&&panel('flaky').hidden===true);

  // a filter click re-renders every card; open panels must come back open
  d.querySelector('.fchip[data-hw="TPU"]').click();
  setTimeout(()=>{
    ok('open state survives the re-render a filter click triggers',!!panel('worker')&&panel('worker').hidden===false);
    ok('exactly one button per guide after the re-render',KEYS.every(k=>d.querySelectorAll(`[data-gbtn="${k}"]`).length===1));
    btn('worker').click();
    ok('second click closes it',panel('worker').hidden===true&&btn('worker').textContent.includes('How to read'));
    d.querySelector('.fchip[data-hw="all"]').click();
    console.log(`\n${pass} passed, ${fail} failed`);
    process.exit(fail?1:0);
  },250);
},400);
