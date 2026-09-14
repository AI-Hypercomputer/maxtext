// Checks the CI Pulse tab that dev/bench/index.html uses to open the dashboard:
// which tab opens by default, that the iframe is not downloaded until the tab is
// used, that the URL remembers the tab, and that the iframe height is measured
// from the real header instead of a hard-coded number.
const fs=require('fs');const path=require('path');const {JSDOM}=require('jsdom');
const BENCH=path.join(__dirname,'..','..','..','..','dev','bench');
let pass=0,fail=0;
function ok(c,m){if(c){pass++;console.log('  ok   '+m)}else{fail++;console.log('  FAIL '+m)}}

let html=fs.readFileSync(path.join(BENCH,'index.html'),'utf8');
// Chart.js is loaded from a CDN that jsdom will not fetch, and data.js is a
// sibling script. Stub the first and inline the second so the page's own
// start-up code actually runs.
html=html.replace(/<script src="https:\/\/cdn[^"]*"><\/script>/,'<script>window.Chart=function(){};</script>');
html=html.replace('<script src="data.js"></script>','<script>'+fs.readFileSync(path.join(BENCH,'data.js'),'utf8')+'</script>');

const load=hash=>new JSDOM(html,{runScripts:'dangerously',pretendToBeVisual:true,
  url:'http://localhost/dev/bench/'+hash});

// --- the page as a reader first sees it -------------------------------------
const dom=load('');const d=dom.window.document;
ok(d.querySelector('#benchmarks-page').classList.contains('active'),'Benchmarks is the default tab');
ok(!d.querySelector('#ci-page').classList.contains('active'),'the CI pane starts hidden');

const frame=d.getElementById('ci-metrics-frame');
ok(frame!==null,'the CI Pulse iframe is in the page');
ok(frame.getAttribute('src')===null,'the iframe has no src until the tab is opened');
ok(frame.dataset.src==='ci-pulse/index.html','the iframe points at the sibling ci-pulse folder');
ok(!html.includes('tools/ci_metrics/site/index.html'),'nothing still points at the old tools path');
ok(d.documentElement.style.getPropertyValue('--ci-chrome')!=='','the header height is measured at start-up');

// --- opening and closing the tab --------------------------------------------
const btn=v=>[...d.querySelectorAll('.toggle-button')].find(b=>b.dataset.view===v);
ok(btn('ci').textContent.trim()==='CI Pulse','the second button reads CI Pulse');
btn('ci').click();
ok(d.querySelector('#ci-page').classList.contains('active'),'clicking it shows the CI pane');
ok(!d.querySelector('#benchmarks-page').classList.contains('active'),'and hides the benchmark pane');
ok(btn('ci').classList.contains('active'),'the CI button is marked as the active one');
ok(frame.getAttribute('src')==='ci-pulse/index.html','the iframe loads on first open');
ok(dom.window.location.hash==='#ci','the URL remembers the CI tab');
btn('benchmarks').click();
ok(d.querySelector('#benchmarks-page').classList.contains('active'),'clicking back returns to the charts');
ok(dom.window.location.hash==='#benchmarks','the URL remembers that too');

// --- a link straight to the dashboard ---------------------------------------
const deep=load('#ci');const dd=deep.window.document;
ok(dd.querySelector('#ci-page').classList.contains('active'),'a #ci link opens on the CI tab');
ok(dd.getElementById('ci-metrics-frame').getAttribute('src')==='ci-pulse/index.html','and loads the iframe right away');

// --- the height rule --------------------------------------------------------
ok(/height:\s*calc\(100vh - var\(--ci-chrome, 88px\)\)/.test(html),'the iframe height reads the measured value');
ok(!/height:\s*calc\(100vh - 88px\);/.test(html),'the hard-coded 88px height is gone');

// --- the file the tab opens actually exists ---------------------------------
ok(fs.existsSync(path.join(BENCH,'ci-pulse','index.html')),'dev/bench/ci-pulse/index.html is there to open');

console.log('\n'+pass+' passed, '+fail+' failed');
process.exit(fail?1:0);
