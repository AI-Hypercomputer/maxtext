// Real-Chrome check of the flipped axes: label order by screen x, and the brush zoom's slot->index inversion.
const {spawn}=require('child_process');
// Manual check (needs Chrome): node cdp_flip.js   -- set CHROME=/path/to/chrome to override the binary.
const path=require('path'),os=require('os');
const CH=process.env.CHROME||'/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const port=9400+Math.floor(Math.random()*100);
const file='file://'+path.join(__dirname,'..','index.html');
const chrome=spawn(CH,['--headless=new','--disable-gpu','--hide-scrollbars','--window-size=1400,1700','--remote-debugging-port='+port,'--user-data-dir='+path.join(os.tmpdir(),'ci-pulse-prof'+port),'about:blank'],{stdio:'ignore'});
const sleep=ms=>new Promise(r=>setTimeout(r,ms));
(async()=>{
  let targets;for(let i=0;i<40;i++){await sleep(250);try{targets=await (await fetch(`http://127.0.0.1:${port}/json`)).json();if(targets.some(t=>t.type==='page'))break}catch(e){}}
  const pg=targets.find(t=>t.type==='page');const ws=new WebSocket(pg.webSocketDebuggerUrl);
  await new Promise(r=>ws.onopen=r);
  let id=0;const pending={};ws.onmessage=m=>{const j=JSON.parse(m.data);if(j.id&&pending[j.id]){pending[j.id](j);delete pending[j.id]}};
  const send=(method,params={})=>new Promise(r=>{const i=++id;pending[i]=r;ws.send(JSON.stringify({id:i,method,params}))});
  const ev=async expr=>{const r=await send('Runtime.evaluate',{expression:expr,returnByValue:true});return r.result&&r.result.result?r.result.result.value:JSON.stringify(r)};
  await send('Page.enable');await send('Page.navigate',{url:file});await sleep(3500);
  let fails=0;const ok=(n,c)=>{console.log((c?'  ok - ':'  FAIL - ')+n);if(!c)fails++};
  const drag=async(xa,xb,y)=>{
    await send('Input.dispatchMouseEvent',{type:'mouseMoved',x:xa,y});
    await send('Input.dispatchMouseEvent',{type:'mousePressed',x:xa,y,button:'left',buttons:1,clickCount:1});
    for(let k=1;k<=10;k++){await send('Input.dispatchMouseEvent',{type:'mouseMoved',x:xa+(xb-xa)*k/10,y,button:'left',buttons:1});await sleep(20)}
    await send('Input.dispatchMouseEvent',{type:'mouseReleased',x:xb,y,button:'left',buttons:0,clickCount:1});
    await sleep(400);
  };
  // timeline
  const box=JSON.parse(await ev(`(()=>{const s=document.querySelector('#timeline svg');const r=s.getBoundingClientRect();const links=[...s.querySelectorAll('a.axlink text')].map(t=>{const b=t.getBoundingClientRect();return{h:t.textContent.trim(),x:b.left+b.width/2}}).sort((a,b)=>a.x-b.x);return JSON.stringify({top:r.top,bottom:r.bottom,links})})()`));
  ok('timeline leftmost label is the oldest in range (#4894): '+box.links[0].h,box.links[0].h.startsWith('#4894'));
  ok('timeline rightmost label is the newest (#4908): '+box.links[box.links.length-1].h,box.links[box.links.length-1].h.startsWith('#4908'));
  const y=(box.top+box.bottom)/2-30;
  await drag(box.links[box.links.length-2].x-10,box.links[box.links.length-1].x+10,y);
  const z=await ev('JSON.stringify(TL_ZOOM)');
  ok('dragging over the two rightmost bars zooms to the two NEWEST commits {a:0,b:1} -> '+z,z==='{"a":0,"b":1}');
  const zbText=await ev(`document.querySelector('#tl-zoombar').textContent`);
  ok('zoomed zoombar reads left -> right (#4910 to #4908)',zbText.includes('(#4910 to #4908)'));
  const zoomedOrder=JSON.parse(await ev(`JSON.stringify([...document.querySelectorAll('#timeline svg a.axlink text')].map(t=>{const b=t.getBoundingClientRect();return{h:t.textContent.trim(),x:b.left}}).sort((a,b)=>a.x-b.x).map(o=>o.h))`));
  ok('zoomed view keeps oldest left: '+zoomedOrder.join(' | '),zoomedOrder.length===2&&zoomedOrder[0].startsWith('#4910')&&zoomedOrder[1].startsWith('#4908'));
  await ev('tlZoomTo(4,7)');await sleep(400);
  const pills=JSON.parse(await ev(`JSON.stringify([...document.querySelectorAll('#timeline svg rect[filter]')].map(r=>({x:+r.getAttribute('x'),w:+r.getAttribute('width'),y:+r.getAttribute('y')})))`));
  ok('zoomed onto #4916-#4922: annotation pills present and none overlap on a row ('+pills.length+' pills)',pills.length>=3&&pills.every((p,i)=>pills.every((q,j)=>i===j||p.y!==q.y||p.x+p.w<=q.x+0.01||q.x+q.w<=p.x+0.01)));
  const pillOrder=JSON.parse(await ev(`JSON.stringify([...document.querySelectorAll('#timeline svg a.axlink text')].map(t=>({h:t.textContent.trim(),x:t.getBoundingClientRect().left})).sort((a,b)=>a.x-b.x).map(o=>o.h))`));
  ok('zoomed 4-bar view runs #4922 -> #4916 left to right: '+pillOrder.join(' | '),pillOrder[0].startsWith('#4922')&&pillOrder[3].startsWith('#4916'));
  await ev('tlZoomReset()');await sleep(300);
  // device lines: drag over the right end -> DEV_ZOOM.a must be 0 (newest)
  await ev(`document.querySelectorAll('#page-main > .card').forEach((c,i)=>{if(i<2)c.style.display='none'});window.scrollTo(0,0)`);await sleep(300);
  const dv=JSON.parse(await ev(`(()=>{const s=[...document.querySelectorAll('#devlines svg')].find(x=>x.querySelector('a.axlink'));const r=s.getBoundingClientRect();return JSON.stringify({l:r.left,r:r.right,t:r.top,b:r.bottom})})()`));
  const dy=(dv.t+dv.b)/2;
  await drag(dv.l+(dv.r-dv.l)*0.72,dv.r-75,dy);
  const dz=await ev('JSON.stringify(DEV_ZOOM)');
  ok('device-lines drag over the right end zooms to the newest commits (a:0) -> '+dz,dz!=='null'&&JSON.parse(dz).a===0);
  const devOrder=JSON.parse(await ev(`JSON.stringify([...document.querySelectorAll('#devlines svg a.axlink text')].map(t=>{const b=t.getBoundingClientRect();return{h:t.textContent.trim(),x:b.left}}).sort((a,b)=>a.x-b.x).map(o=>o.h))`));
  ok('zoomed device view: newest (#4908) is rightmost -> '+devOrder.join(' | '),devOrder[devOrder.length-1].startsWith('#4908'));
  await ev('devZoomReset()');await sleep(300);
  console.log(fails?`\n${fails} FAILED`:'\nall CDP checks passed');
  chrome.kill();process.exit(fails?1:0);
})().catch(e=>{console.error(e);chrome.kill();process.exit(2)});
