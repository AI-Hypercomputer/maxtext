// Checks that the run-time chart opens on the newest merged pull requests (2026-10-19).
// The chart draws the oldest pull request on the left and the newest on the right, and it is wider
// than its card once there are more than about twenty bars. It used to open scrolled to the far
// left, on the oldest pull requests, so the latest runs were off screen.
//
// jsdom does no layout, so this check gives the chart box a width (1100) and a scroll width (5000)
// and records where the page scrolls it.
const {JSDOM}=require('jsdom');
const pageHtml=require('./loadpage.js');

let pass=0,fail=0;
function ok(c,m){if(c){pass++;console.log('  ok   '+m)}else{fail++;console.log('  FAIL '+m)}}
function finish(){console.log('\n'+pass+' passed, '+fail+' failed');process.exit(fail?1:0)}

const layout=`<script>
(function(){
  const box=el=>el&&el.id==='timeline';
  Object.defineProperty(HTMLElement.prototype,'clientWidth',{configurable:true,get(){return box(this)?1100:0}});
  Object.defineProperty(HTMLElement.prototype,'scrollWidth',{configurable:true,get(){return box(this)?5000:0}});
  Object.defineProperty(Element.prototype,'scrollLeft',{configurable:true,
    get(){return this.__scrollLeft||0},
    set(v){this.__scrollLeft=Math.max(0,Math.min(Number(v)||0,(box(this)?5000-1100:0)))}});
})();
</script>`;
const html=pageHtml().replace('<script>',layout+'\n<script>');
const dom=new JSDOM(html,{runScripts:'dangerously',pretendToBeVisual:true,url:'http://localhost/'});
const w=dom.window,d=w.document;
const tl=()=>d.getElementById('timeline');

(function wait(n){
  if(d.querySelector('#timeline svg')||n>250)return check();
  setTimeout(()=>wait(n+1),20);
})(0);

function check(){
  ok(!!d.querySelector('#timeline svg'),'the run-time chart is drawn');
  if(!d.querySelector('#timeline svg'))finish();
  ok(tl().scrollLeft===3900,`the chart opens scrolled to the right end, on the newest pull requests (scrollLeft ${tl().scrollLeft} of 3900)`);

  // A new date range draws a new chart, and it opens on the newest pull requests again.
  tl().scrollLeft=0;
  const sel=d.getElementById('date-range');
  sel.value=sel.value==='30'?'14':'30';
  sel.dispatchEvent(new w.Event('change',{bubbles:true}));
  ok(tl().scrollLeft===3900,`after changing the date range it is at the right end again (scrollLeft ${tl().scrollLeft})`);
  finish();
}
