// Linked, dated x-axis labels on both bar charts; the card order they depend on
// (timeline first, worker second) and the copy that points between them; and the
// sortable per-test tables in the commit modal. Grew out of the 2026-08-31 session.
const fs=require('fs');const {JSDOM}=require('jsdom');
const html=fs.readFileSync(require('path').join(__dirname,'..','index.html'),'utf-8');
const dom=new JSDOM(html,{runScripts:'dangerously',pretendToBeVisual:true});
const w=dom.window,d=w.document;
let pass=0,fail=0;
function ok(name,cond){if(cond){pass++;console.log('  ok -',name)}else{fail++;console.log('  FAIL -',name)}}
const txt=n=>n?n.textContent.trim():'';
setTimeout(()=>{
  // ---------- A. axis labels ----------
  const wsvg=d.querySelector('#worker-chart svg'),tsvg=d.querySelector('#timeline svg');
  ok('both bar charts mounted',!!wsvg&&!!tsvg);
  const wl=wsvg.querySelectorAll('a.axlink'),tl=tsvg.querySelectorAll('a.axlink');
  const cur=w.getWindows().current,n=cur.length,oldest=cur[cur.length-1].hash,newest=cur[0].hash;
  ok('worker chart: one hash link per PR in range ('+n+')',wl.length===n&&n===20);
  ok('timeline: one hash link per PR in range',tl.length===n);
  ok('link -> GitHub PR, new tab',wl[0].getAttribute('href')==='https://github.com/AI-Hypercomputer/maxtext/pull/4908'&&wl[0].getAttribute('target')==='_blank');
  ok('link has hover title',txt(wl[0].querySelector('title'))==='Open pull request #4908 on GitHub');
  ok('hash text under bar is the bare pull request number',txt(wl[0].querySelector('text'))==='#4908'&&!wl[0].querySelector('text tspan'));
  ok('date line follows the hash, no time line',txt(wl[0].nextSibling)==='Aug 20'&&!/^\d\d:\d\d$/.test(txt(wl[0].nextSibling.nextSibling)));
  ok('last slot = oldest in range (#4894 Aug 06)',txt(wl[n-1].querySelector('text'))==='#'+oldest&&oldest==='4894'&&txt(wl[n-1].nextSibling)==='Aug 06');
  ok('hash uses accent ink (inline style beats #timeline css)',wl[0].querySelector('text').style.fill==='var(--accent)'&&tl[0].querySelector('text').style.fill==='var(--accent)');
  ok('worker viewBox = 26+274+44 with two-line labels',wsvg.getAttribute('viewBox')==='0 0 1100 344');
  ok('timeline viewBox = 70+228+44 with two-line labels',tsvg.getAttribute('viewBox')==='0 0 1100 342');
  // single-suite worker view uses the same labels
  w.setWKSel('cpu-pre-u');
  const ssvg=d.querySelector('#worker-chart svg');
  ok('single-suite view: one link per PR in range, viewBox 314',ssvg.querySelectorAll('a.axlink').length===n&&ssvg.getAttribute('viewBox')==='0 0 1100 314');
  w.setWKSel('all');
  // ---------- C. card order + copy ----------
  const heroes=d.querySelectorAll('#page-main .card.hero');
  ok('timeline card is first, worker card second',txt(heroes[0].querySelector('.ct')).startsWith('Main branch · run time per merged pull request')&&txt(heroes[1].querySelector('.ct')).startsWith('Worker time per merged pull request'));
  ok('worker copy says "chart above"',heroes[1].textContent.includes('The chart above splits the same runs by phase.')&&!d.body.textContent.includes('The chart below splits'));
  ok('timeline copy points down to the worker chart',heroes[0].textContent.includes('The chart below breaks the same runs down by worker suite.'));
  // ---------- B1. commit modal sort ----------
  w.goToCommit('4916');
  const bodies=Array.from(d.querySelectorAll('#modal tbody[id^="jt-xrow-4916-"]'));
  ok('4916 modal: 2 per-test tables registered',bodies.length===2&&bodies.every(b=>w.eval('JT')[b.id.replace(/-body$/,'')]));
  const tb=bodies.find(b=>b.querySelectorAll('tr').length===11);
  const tid=tb.id.replace(/-body$/,'');
  const names=()=>Array.from(tb.querySelectorAll('tr')).map(r=>txt(r.querySelector('td')));
  const durs=()=>Array.from(tb.querySelectorAll('tr')).map(r=>txt(r.querySelectorAll('td')[1]));
  ok('default order: biggest slow-down first, smallest last',names()[0]==='test_llama3_70b_sharded_convergence'&&names()[10]==='test_attention_flash_v2');
  w.jtSort(tid,'dur');
  ok('sort by duration: longest first (45.2s) ... shortest last (2.1s)',durs()[0]==='45.2s'&&durs()[10]==='2.1s'&&names()[5]==='test_convergence_gemma_7b');
  const thDur=d.querySelector(`#modal th[data-jt="${tid}"][data-jtkey="dur"] .jtarr`);
  ok('active header shows ▼',txt(thDur)==='▼');
  w.jtSort(tid,'dur');
  ok('second click flips to shortest first',durs()[0]==='2.1s'&&durs()[10]==='45.2s'&&txt(thDur)==='▲');
  w.jtSort(tid,'');
  ok('Test header restores default order',names()[10]==='test_attention_flash_v2'&&txt(thDur)==='');
  ok('modal hint sentence present',d.querySelector('#modal').textContent.includes('Click Duration, Baseline, or Δ to sort by that column'));
  w.closeModal();
  w.goToCommit('4920');
  const tb2=Array.from(d.querySelectorAll('#modal tbody[id^="jt-xrow-4920-"]')).find(b=>b.querySelectorAll('tr').length===4);
  const tid2=tb2.id.replace(/-body$/,'');
  const n2=()=>Array.from(tb2.querySelectorAll('tr')).map(r=>txt(r.querySelector('td')));
  ok('4920 default: failed test first',n2()[0]==='test_ring_attention_multihost');
  w.jtSort(tid2,'dur');
  ok('4920 sort by duration: failed test (no duration) sinks to bottom',n2()[0]==='test_llama3_70b_sharded_convergence'&&n2()[3]==='test_ring_attention_multihost');
  w.jtSort(tid2,'delta');
  ok('4920 sort by Δ: +9.1s (moe) first, no-Δ last',n2()[0]==='test_mixture_of_experts_routing'&&n2()[3]==='test_ring_attention_multihost');
  w.closeModal();
  // ---------- D. review-fix asserts (wf_38aa44ab-975) ----------
  w.tlZoomReset();w.setWKSel('all');
  const h0=d.querySelectorAll('#page-main .card.hero');
  ok('both bar cards state the direction and the GitHub link',[0,1].every(i=>h0[i].textContent.includes('Bars run from the oldest merged pull request on the left to the newest on the right.')&&!h0[i].textContent.includes('right to left')&&h0[i].textContent.includes('opens that pull request on GitHub in a new tab')&&h0[i].textContent.includes('its merge date is under it')));
  ok('zoombar says oldest left, newest right',d.querySelector('#tl-zoombar').textContent.includes('oldest on the left and newest on the right'));
  const tsvg2=d.querySelector('#timeline svg');
  const medLine=Array.from(tsvg2.querySelectorAll('line')).find(l=>l.getAttribute('stroke-dasharray')==='6,4');
  const medText=Array.from(tsvg2.querySelectorAll('text')).find(t=>t.textContent.startsWith('median '));
  ok('median line uses neutral ink, label 12px, no cyan left on non-link text',!!medLine&&medLine.style.stroke==='var(--ink2)'&&!medLine.getAttribute('stroke')&&!!medText&&medText.getAttribute('font-size')==='12'&&!medText.getAttribute('fill'));
  w.tlZoomTo(4,7);
  const zt=Array.from(d.querySelectorAll('#timeline svg text')).map(t=>t.textContent);
  ok('trailing median now uses OLDER commits: #4920 pill reads +11m slower / usually 32m',zt.includes('+11m slower')&&zt.some(t=>t.includes('usually 32m'))&&!zt.some(t=>t.includes('usually 33m')));
  ok('#4916 (36m vs 33m older median = +9%) stays unflagged, #4918 flagged',zt.includes('+9m slower')&&!zt.includes('+4m slower'));
  w.tlZoomReset();
  w.goToCommit('4916');
  const mtxt=d.querySelector('#modal').textContent;
  ok('modal hint describes the real default order and the blank-cell rule',mtxt.includes('Failed tests come first. Then come the tests that added the most time')&&mtxt.includes('drop to the bottom: failed tests have no duration, and new tests have no baseline yet')&&mtxt.includes('Click the Test header to restore the default order')&&!mtxt.includes('Failures come first'));
  ok('JUnit header style is 12px (sort arrow inherits it)',(d.querySelector('#modal th[data-jtkey="dur"]').getAttribute('style')||'').includes('font-size:12px'));
  w.closeModal();
  // ---------- flip: oldest left, newest right on every commit-driven chart (2026-08-31 part I) ----------
  const byX=nodes=>[...nodes].map(n=>({n,x:+n.getAttribute('x')})).sort((a,b)=>a.x-b.x).map(o=>o.n.textContent.trim());
  const tlH=byX(d.querySelectorAll('#timeline svg a.axlink text'));
  ok('timeline: oldest in range leftmost, newest rightmost',tlH[0]==='#'+oldest&&tlH[tlH.length-1]==='#'+newest&&newest==='4908');
  const wkH=byX(d.querySelectorAll('#worker-chart svg a.axlink text'));
  ok('worker: oldest leftmost, newest rightmost',wkH[0]==='#'+oldest&&wkH[wkH.length-1]==='#'+newest);
  const devH=byX(d.querySelectorAll('#devlines svg a.axlink text'));
  // device + queue charts draw the 10-PR w.current window (#4940 .. #4914), not all 13 COMMITS
  ok('device lines: every PR in range, oldest leftmost, newest rightmost',devH.length===n&&devH[0]==='#'+oldest&&devH[devH.length-1]==='#'+newest);
  const devEnd=[...d.querySelectorAll('#devlines svg text')].filter(t=>/^(TPU|GPU|CPU) [\d.]+m$/.test(t.textContent));
  ok('device endpoint labels sit at the right edge',devEnd.length===3&&devEnd.every(t=>+t.getAttribute('x')>1000));
  ok('timeline guide states oldest -> newest',d.getElementById('guide-timeline').textContent.includes('oldest merged pull request on the left to the newest on the right'));
  ok('legend draws the three badge marks',['slow','fail','flaky'].every(k=>!!d.querySelector(`.leg [data-lg="${k}"] svg circle`))&&d.querySelectorAll('.leg [data-lg="fail"] svg line').length===2);
  ok('legend labels name the states',txt(d.querySelector('.leg [data-lg="slow"]')).includes('slower than recent runs')&&txt(d.querySelector('.leg [data-lg="fail"]'))==='failed'&&txt(d.querySelector('.leg [data-lg="flaky"]')).includes('passed on retry'));
  ok('timeline guide explains the marks',d.getElementById('guide-timeline').textContent.includes('▲ means the run was slower than recent runs'));
  // worker meta: sentence and legend on separate lines (2026-08-31 part K)
  const wkMeta=d.querySelector('#wk-meta');
  ok('worker meta = sentence line + legend line',wkMeta.children.length===2&&wkMeta.children[0].tagName==='DIV'&&wkMeta.children[1].hasAttribute('data-wklegend')&&wkMeta.children[1].textContent.includes('TPU')&&wkMeta.children[1].textContent.includes('Checks')&&!wkMeta.children[0].textContent.includes('Checks'));
  w.setWKSel('tpu-pre-u');
  const wkMeta2=d.querySelector('#wk-meta');
  ok('single-suite meta keeps the same two-line shape',wkMeta2.children.length===2&&wkMeta2.children[1].hasAttribute('data-wklegend')&&wkMeta2.children[1].textContent.includes('Runner wait'));
  w.setWKSel('all');
  console.log(`\n${pass} passed, ${fail} failed`);
  process.exit(fail?1:0);
},300);
