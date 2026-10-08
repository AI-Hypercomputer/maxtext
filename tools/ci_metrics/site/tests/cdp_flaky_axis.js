// Real-Chrome check of the Re-runs per job axis: every date label must sit under its own cell.
// jsdom has no layout, so the grid track that decides this can only be measured here.
// Manual check (needs Chrome and the page served over http):
//   python3 -m http.server 8899 --directory dev/bench/ci-pulse
//   node cdp_flaky_axis.js       -- set URL= or CHROME= to override.
const {spawn}=require('child_process'),path=require('path'),os=require('os');
const CH=process.env.CHROME||'/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const port=9500+Math.floor(Math.random()*100);
const url=process.env.URL||'http://127.0.0.1:8899/index.html';
const chrome=spawn(CH,['--headless=new','--disable-gpu','--hide-scrollbars','--window-size=1500,2400',
  '--remote-debugging-port='+port,'--user-data-dir='+path.join(os.tmpdir(),'cip'+port),'about:blank'],{stdio:'ignore'});
const sleep=ms=>new Promise(r=>setTimeout(r,ms));
(async()=>{
  let targets;for(let i=0;i<40;i++){await sleep(250);try{targets=await (await fetch(`http://127.0.0.1:${port}/json`)).json();if(targets.some(t=>t.type==='page'))break}catch(e){}}
  const pg=targets.find(t=>t.type==='page');const ws=new WebSocket(pg.webSocketDebuggerUrl);
  await new Promise(r=>ws.onopen=r);
  let id=0;const pending={};ws.onmessage=m=>{const j=JSON.parse(m.data);if(j.id&&pending[j.id]){pending[j.id](j);delete pending[j.id]}};
  const send=(method,params={})=>new Promise(r=>{const i=++id;pending[i]=r;ws.send(JSON.stringify({id:i,method,params}))});
  const ev=async e=>{const r=await send('Runtime.evaluate',{expression:e,returnByValue:true});return r.result&&r.result.result?r.result.result.value:JSON.stringify(r)};
  await send('Page.enable');await send('Page.navigate',{url});await sleep(4000);
  const out=JSON.parse(await ev(`(()=>{
    const row=document.querySelector('#flaky .issue-row');
    const ax=document.querySelector('#flaky [data-flaxis]');
    if(!row||!ax)return JSON.stringify({err:'not found'});
    const cells=[...row.querySelectorAll('[data-cell]')].map(c=>{const b=c.getBoundingClientRect();return b.left+b.width/2});
    const ticks=[...ax.children[1].children].map(c=>{const b=c.getBoundingClientRect();return b.left+b.width/2});
    const rr=row.getBoundingClientRect(), ar=ax.getBoundingClientRect();
    const mid=(el)=>{const b=el.getBoundingClientRect();return {l:b.left,w:b.width}};
    return JSON.stringify({
      rowBox:mid(row), axBox:mid(ax),
      rowMid:mid(row.children[1]), axMid:mid(ax.children[1]),
      n:cells.length, m:ticks.length,
      first:{cell:cells[0],tick:ticks[0]}, last:{cell:cells[cells.length-1],tick:ticks[ticks.length-1]},
      maxDelta:Math.max(...cells.map((c,i)=>Math.abs(c-(ticks[i]??c))))
    });
  })()`));
  console.log('row cells  left',out.rowMid.l,'width',out.rowMid.w);
  console.log('axis ticks left',out.axMid.l,'width',out.axMid.w);
  console.log('cells',out.n,'ticks',out.m,'| worst misalignment',out.maxDelta,'px');
  const bad=out.n!==out.m||out.maxDelta>0.5;
  console.log(bad?'  FAIL - a date label does not sit under its own cell':'  ok - every date label sits under its own cell');
  ws.close();chrome.kill();process.exit(bad?1:0);
})();
