// Runs every check_*.js against ../index.html and exits non-zero if any assertion fails.
// Usage: cd tools/ci_metrics/site/tests && npm install && npm test   (jsdom is the only dependency)
const {spawnSync}=require('child_process');const fs=require('fs');const path=require('path');
const files=fs.readdirSync(__dirname).filter(f=>/^check_.*\.js$/.test(f)).sort();
let failed=0,total=0;
for(const f of files){
  const r=spawnSync(process.execPath,[path.join(__dirname,f)],{encoding:'utf-8'});
  const out=(r.stdout||'')+(r.stderr||'');
  const m=out.match(/(\d+) passed, (\d+) failed|(\d+)\/(\d+) passed/);
  const passed=m?Number(m[1]||m[3]):0,fails=m?Number(m[2]||(Number(m[4])-Number(m[3]))):1;
  total+=passed;failed+=fails;
  console.log(`${fails?'FAIL':'ok  '} ${f}: ${passed} passed, ${fails} failed`);
  if(fails)console.log(out.split('\n').filter(l=>/FAIL|Error/.test(l)).slice(0,10).map(l=>'     '+l).join('\n'));
  if(r.status!==0&&!m){failed++;console.log('     (process exited with status '+r.status+')');}
}
console.log(`\n${files.length} files, ${total} assertions passed, ${failed} failed`);
process.exit(failed?1:0);
