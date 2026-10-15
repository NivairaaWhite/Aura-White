"""Single-page web UI + dependency-free WebGL viewer (served by aura_white.server)."""

INDEX_HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Aura White</title>
<style>
:root{--bg:#0e0f13;--panel:#16181f;--line:#262a35;--fg:#eef0f6;--dim:#8b91a3;--acc:#f4f6ff}
*{box-sizing:border-box}html,body{margin:0;height:100%;background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,Segoe UI,Roboto,sans-serif}
.app{display:grid;grid-template-columns:340px 1fr;height:100%}
@media(max-width:820px){.app{grid-template-columns:1fr;grid-template-rows:auto 60vh}}
aside{background:var(--panel);border-right:1px solid var(--line);padding:18px;overflow:auto}
h1{font-size:20px;margin:0 0 2px;letter-spacing:.5px}.sub{color:var(--dim);margin:0 0 16px}
#drop{border:1.5px dashed #3a4052;border-radius:12px;min-height:150px;display:flex;align-items:center;justify-content:center;text-align:center;color:var(--dim);cursor:pointer;overflow:hidden;position:relative}
#drop.over{border-color:var(--acc);color:var(--fg)}#drop img{max-width:100%;max-height:220px;display:block}
label{display:block;margin:12px 0 4px;color:var(--dim);font-size:12px;text-transform:uppercase;letter-spacing:.6px}
input[type=range]{width:100%}select,input[type=number]{width:100%;background:#0f1117;color:var(--fg);border:1px solid var(--line);border-radius:8px;padding:7px}
.row{display:flex;gap:8px;align-items:center}.row>*{flex:1}.chk{display:flex;gap:8px;align-items:center;margin-top:10px;color:var(--fg);text-transform:none;font-size:14px;letter-spacing:0}
button{width:100%;margin-top:16px;background:var(--acc);color:#0b0c10;font-weight:700;border:0;border-radius:10px;padding:12px;font-size:15px;cursor:pointer}button:disabled{opacity:.45;cursor:default}
#bar{height:6px;background:#232734;border-radius:4px;margin-top:14px;overflow:hidden;display:none}#bar i{display:block;height:100%;width:0;background:var(--acc);transition:width .3s}
#msg{margin-top:8px;color:var(--dim);min-height:20px}#msg.err{color:#ff8c8c}
#dl{margin-top:12px;display:flex;flex-wrap:wrap;gap:8px}#dl a{color:var(--fg);border:1px solid var(--line);padding:6px 12px;border-radius:8px;text-decoration:none;background:#0f1117}
#stats{color:var(--dim);font-size:12px;margin-top:10px;white-space:pre-line}
main{position:relative;min-height:0}canvas{width:100%;height:100%;display:block;touch-action:none;cursor:grab}
#hint{position:absolute;left:14px;bottom:12px;color:var(--dim);font-size:12px;pointer-events:none}
</style></head><body><div class="app"><aside>
<h1>Aura White</h1><p class="sub">Image to 3D, locally.</p>
<div id="drop"><span id="dropText">Drop an image here, or tap to choose</span></div>
<input id="file" type="file" accept="image/*" hidden>
<label>Mode</label>
<select id="mode"><option value="studio">Sharp textured (UV texture + normal map)</option><option value="fast">Fast (vertex colours)</option></select>
<div id="studioOpts"><label>Quality</label><select id="qual"><option value="draft">Draft (1K texture)</option><option value="standard" selected>Standard (2K texture)</option><option value="high">High (4K texture)</option></select></div>
<div id="fastOpts"><label>Mesh detail <span id="resv">256</span></label>
<input id="res" type="range" min="96" max="384" step="32" value="256">
<label>Smoothing <span id="smv">2</span></label>
<input id="sm" type="range" min="0" max="8" step="1" value="2"></div>
<label>Formats</label>
<div class="row"><label class="chk"><input type="checkbox" class="fmt" value="glb" checked>GLB</label><label class="chk"><input type="checkbox" class="fmt" value="obj" checked>OBJ</label><label class="chk"><input type="checkbox" class="fmt" value="ply">PLY</label><label class="chk"><input type="checkbox" class="fmt" value="stl">STL</label></div>
<label class="chk"><input id="rmbg" type="checkbox" checked>Remove background</label>
<button id="go" disabled>Generate 3D</button>
<div id="bar"><i></i></div><div id="msg"></div><div id="dl"></div><div id="stats"></div>
</aside><main><canvas id="gl"></canvas><div id="hint">drag to rotate - scroll or pinch to zoom</div></main></div>
<script>
"use strict";
const $=s=>document.querySelector(s);
let file=null;
function setFile(f){if(!f)return;file=f;const url=URL.createObjectURL(f);$("#drop").innerHTML='<img alt="input">';$("#drop img").src=url;$("#go").disabled=false;msg("");}
$("#drop").onclick=()=>$("#file").click();
$("#file").onchange=e=>setFile(e.target.files[0]);
["dragenter","dragover"].forEach(t=>$("#drop").addEventListener(t,e=>{e.preventDefault();$("#drop").classList.add("over")}));
["dragleave","drop"].forEach(t=>$("#drop").addEventListener(t,e=>{e.preventDefault();$("#drop").classList.remove("over")}));
$("#drop").addEventListener("drop",e=>setFile(e.dataTransfer.files[0]));
function syncMode(){const st=$("#mode").value==="studio";$("#studioOpts").style.display=st?"block":"none";$("#fastOpts").style.display=st?"none":"block"}
$("#mode").onchange=syncMode;syncMode();
$("#res").oninput=e=>$("#resv").textContent=e.target.value;$("#sm").oninput=e=>$("#smv").textContent=e.target.value;
function msg(t,err){const m=$("#msg");m.textContent=t;m.className=err?"err":""}
function bar(f){const b=$("#bar");b.style.display=f==null?"none":"block";if(f!=null)b.firstElementChild.style.width=Math.round(f*100)+"%"}
$("#go").onclick=async()=>{
  if(!file)return;$("#go").disabled=true;$("#dl").innerHTML="";$("#stats").textContent="";bar(0);msg("uploading...");
  const fm=[...document.querySelectorAll(".fmt:checked")].map(x=>x.value).join(",")||"glb";
  const q=new URLSearchParams({resolution:$("#res").value,smooth:$("#sm").value,remove_bg:$("#rmbg").checked?1:0,formats:fm,mode:$("#mode").value,quality:$("#qual").value});
  try{
    const r=await fetch("/api/jobs?"+q,{method:"POST",body:file});
    const j=await r.json();if(!r.ok)throw new Error(j.error||r.statusText);
    await poll(j.id);
  }catch(e){msg(String(e.message||e),true);bar(null);}
  $("#go").disabled=false;
};
async function poll(id){
  for(;;){
    const j=await (await fetch("/api/jobs/"+id)).json();
    if(j.status==="error"){throw new Error(j.error);}
    msg(j.stage||j.status);bar(j.progress||0);
    if(j.status==="done"){bar(null);finish(j.result);return;}
    await new Promise(r=>setTimeout(r,600));
  }
}
function finish(res){
  const names={glb:"GLB",obj:"OBJ",ply:"PLY",stl:"STL",albedo:"Texture",normal:"Normal map",preview:"Preview",report:"Report"};
  $("#dl").innerHTML=Object.entries(res.files).filter(([k])=>names[k]).map(([k,u])=>'<a href="'+u+'" download>'+names[k]+'</a>').join("");
  const s=res.stats,t=res.timings;
  if(res.mode==="studio"){
    $("#stats").textContent=s.vertices.toLocaleString()+" vertices, "+s.faces.toLocaleString()+" faces, "+s.texture_size+"px texture\n"+
      "geometry: "+s.geometry+(s.side_views?" + side views":"")+" | artwork match "+Math.round(s.iou*100)+"%\ntotal "+t.total.toFixed(1)+"s"+(s.warnings.length?"\n! "+s.warnings.join("\n! "):"");
    msg("done");loadMesh(res.files.view,res.files.texture);return;
  }
  $("#stats").textContent=s.vertices.toLocaleString()+" vertices, "+s.faces.toLocaleString()+" faces\n"+
    "total "+t.total.toFixed(1)+"s (encode "+t.encode.toFixed(1)+"s, density "+t.density.toFixed(1)+"s)"+(s.warnings.length?"\n! "+s.warnings.join("\n! "):"");
  msg("done");loadMesh(res.files.view);
}
/* ---------------- minimal WebGL viewer ---------------- */
const canvas=$("#gl"),gl=canvas.getContext("webgl",{antialias:true});
let prog,buf={},count=0,yaw=0.6,pitch=0.25,dist=3.4,center=[0,0,0],scale=1,auto=true;
function mul(a,b){const o=new Array(16).fill(0);for(let c=0;c<4;c++)for(let r=0;r<4;r++){let s=0;for(let k=0;k<4;k++)s+=a[k*4+r]*b[c*4+k];o[c*4+r]=s}return o}
function persp(fovy,asp,n,f){const t=1/Math.tan(fovy/2),nf=1/(n-f);return[t/asp,0,0,0,0,t,0,0,0,0,(f+n)*nf,-1,0,0,2*f*n*nf,0]}
function rotX(a){const c=Math.cos(a),s=Math.sin(a);return[1,0,0,0,0,c,s,0,0,-s,c,0,0,0,0,1]}
function rotY(a){const c=Math.cos(a),s=Math.sin(a);return[c,0,-s,0,0,1,0,0,s,0,c,0,0,0,0,1]}
function trans(x,y,z){return[1,0,0,0,0,1,0,0,0,0,1,0,x,y,z,1]}
function scl(s){return[s,0,0,0,0,s,0,0,0,0,s,0,0,0,0,1]}
function sh(type,src){const s=gl.createShader(type);gl.shaderSource(s,src);gl.compileShader(s);if(!gl.getShaderParameter(s,gl.COMPILE_STATUS))throw new Error(gl.getShaderInfoLog(s));return s}
function initGL(){
  if(!gl){$("#hint").textContent="WebGL is not available in this browser - use the download links.";return false}
  if(!gl.getExtension("OES_element_index_uint")){$("#hint").textContent="This browser lacks 32-bit indices - use the download links.";return false}
  prog=gl.createProgram();
  gl.attachShader(prog,sh(gl.VERTEX_SHADER,"attribute vec3 aP;attribute vec3 aN;attribute vec4 aC;attribute vec2 aT;uniform mat4 uM;uniform mat4 uR;varying vec3 vN;varying vec3 vC;varying vec2 vT;void main(){vN=(uR*vec4(aN,0.)).xyz;vC=aC.rgb;vT=aT;gl_Position=uM*vec4(aP,1.);}"));
  gl.attachShader(prog,sh(gl.FRAGMENT_SHADER,"precision mediump float;varying vec3 vN;varying vec3 vC;varying vec2 vT;uniform sampler2D uTex;uniform float uUseTex;void main(){vec3 n=normalize(vN);vec3 base=uUseTex>.5?texture2D(uTex,vT).rgb:vC;float d=max(dot(n,normalize(vec3(.4,.7,.6))),0.)*.75+max(dot(n,normalize(vec3(-.6,-.2,.4))),0.)*.3;float h=.5+.5*n.y;vec3 lin=pow(base,vec3(2.2))*(.3*h+d);gl_FragColor=vec4(pow(lin,vec3(1./2.2)),1.);}"));
  gl.linkProgram(prog);if(!gl.getProgramParameter(prog,gl.LINK_STATUS))throw new Error(gl.getProgramInfoLog(prog));
  gl.enable(gl.DEPTH_TEST);["p","n","c","t","i"].forEach(k=>buf[k]=gl.createBuffer());return true}
const ok=initGL();
let tex=null,useTex=0;
function loadTexture(url){return new Promise((res,rej)=>{const im=new Image();im.onload=()=>{
  const t=gl.createTexture();gl.bindTexture(gl.TEXTURE_2D,t);gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA,gl.RGBA,gl.UNSIGNED_BYTE,im);
  const pot=x=>(x&(x-1))===0;
  if(pot(im.width)&&pot(im.height)){gl.generateMipmap(gl.TEXTURE_2D);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MIN_FILTER,gl.LINEAR_MIPMAP_LINEAR)}else gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MIN_FILTER,gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MAG_FILTER,gl.LINEAR);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_S,gl.CLAMP_TO_EDGE);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_T,gl.CLAMP_TO_EDGE);
  const ext=gl.getExtension("EXT_texture_filter_anisotropic");if(ext)gl.texParameterf(gl.TEXTURE_2D,ext.TEXTURE_MAX_ANISOTROPY_EXT,4);
  res(t)};im.onerror=rej;im.src=url})}
async function loadMesh(url,texUrl){
  if(!ok)return;
  const ab=await (await fetch(url)).arrayBuffer(),dv=new DataView(ab);
  const magic=String.fromCharCode(dv.getUint8(0),dv.getUint8(1),dv.getUint8(2),dv.getUint8(3));
  if(magic!=="AWV1"&&magic!=="AWV2")throw new Error("bad mesh blob");
  const nv=dv.getUint32(4,true),nf=dv.getUint32(8,true);
  const P=new Float32Array(ab,12,nv*3),N=new Float32Array(ab,12+12*nv,nv*3);
  let C=null,T=null,I;
  if(magic==="AWV1"){C=new Uint8Array(ab,12+24*nv,nv*4);I=new Uint32Array(ab,12+28*nv,nf*3);useTex=0}
  else{T=new Float32Array(ab,12+24*nv,nv*2);I=new Uint32Array(ab,12+32*nv,nf*3);tex=await loadTexture(texUrl);useTex=1}
  const lo=[1e9,1e9,1e9],hi=[-1e9,-1e9,-1e9];
  for(let i=0;i<nv;i++)for(let k=0;k<3;k++){const v=P[i*3+k];if(v<lo[k])lo[k]=v;if(v>hi[k])hi[k]=v}
  center=lo.map((v,k)=>(v+hi[k])/2);scale=2/Math.max(hi[0]-lo[0],hi[1]-lo[1],hi[2]-lo[2],1e-6);
  gl.bindBuffer(gl.ARRAY_BUFFER,buf.p);gl.bufferData(gl.ARRAY_BUFFER,P,gl.STATIC_DRAW);
  gl.bindBuffer(gl.ARRAY_BUFFER,buf.n);gl.bufferData(gl.ARRAY_BUFFER,N,gl.STATIC_DRAW);
  if(C){gl.bindBuffer(gl.ARRAY_BUFFER,buf.c);gl.bufferData(gl.ARRAY_BUFFER,C,gl.STATIC_DRAW)}
  if(T){gl.bindBuffer(gl.ARRAY_BUFFER,buf.t);gl.bufferData(gl.ARRAY_BUFFER,T,gl.STATIC_DRAW)}
  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER,buf.i);gl.bufferData(gl.ELEMENT_ARRAY_BUFFER,I,gl.STATIC_DRAW);
  count=nf*3;yaw=0.6;pitch=0.25;dist=3.4;auto=true;
}
function draw(){
  const dpr=window.devicePixelRatio||1,w=Math.max(1,Math.round(canvas.clientWidth*dpr)),h=Math.max(1,Math.round(canvas.clientHeight*dpr));
  if(canvas.width!==w||canvas.height!==h){canvas.width=w;canvas.height=h}
  gl.viewport(0,0,w,h);gl.clearColor(.055,.06,.075,1);gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);
  if(count){
    if(auto)yaw+=0.004;
    const rot=mul(rotX(pitch),rotY(yaw)),model=mul(rot,mul(scl(scale),trans(-center[0],-center[1],-center[2])));
    const mvp=mul(persp(0.75,w/h,0.05,50),mul(trans(0,0,-dist),model));
    gl.useProgram(prog);
    gl.uniformMatrix4fv(gl.getUniformLocation(prog,"uM"),false,new Float32Array(mvp));
    gl.uniformMatrix4fv(gl.getUniformLocation(prog,"uR"),false,new Float32Array(rot));
    const attr=(n,b,sz,ty,nm)=>{const l=gl.getAttribLocation(prog,n);gl.bindBuffer(gl.ARRAY_BUFFER,buf[b]);gl.enableVertexAttribArray(l);gl.vertexAttribPointer(l,sz,ty,nm,0,0)};
    const off=(n,v)=>{const l=gl.getAttribLocation(prog,n);gl.disableVertexAttribArray(l);gl.vertexAttrib4f(l,v[0],v[1],v[2],v[3])};
    attr("aP","p",3,gl.FLOAT,false);attr("aN","n",3,gl.FLOAT,false);
    if(useTex){attr("aT","t",2,gl.FLOAT,false);off("aC",[.8,.8,.8,1]);gl.activeTexture(gl.TEXTURE0);gl.bindTexture(gl.TEXTURE_2D,tex);gl.uniform1i(gl.getUniformLocation(prog,"uTex"),0)}
    else{attr("aC","c",4,gl.UNSIGNED_BYTE,true);off("aT",[0,0,0,0])}
    gl.uniform1f(gl.getUniformLocation(prog,"uUseTex"),useTex);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER,buf.i);gl.drawElements(gl.TRIANGLES,count,gl.UNSIGNED_INT,0);
  }
  requestAnimationFrame(draw)}
if(ok)draw();
const ptr=new Map();let lastPinch=0;
canvas.addEventListener("pointerdown",e=>{canvas.setPointerCapture(e.pointerId);ptr.set(e.pointerId,[e.clientX,e.clientY]);auto=false;lastPinch=0});
canvas.addEventListener("pointerup",e=>ptr.delete(e.pointerId));canvas.addEventListener("pointercancel",e=>ptr.delete(e.pointerId));
canvas.addEventListener("pointermove",e=>{
  if(!ptr.has(e.pointerId))return;const p=ptr.get(e.pointerId);
  if(ptr.size===1){yaw+=(e.clientX-p[0])*0.008;pitch=Math.max(-1.5,Math.min(1.5,pitch+(e.clientY-p[1])*0.008))}
  ptr.set(e.pointerId,[e.clientX,e.clientY]);
  if(ptr.size===2){const [a,b]=[...ptr.values()],d=Math.hypot(a[0]-b[0],a[1]-b[1]);if(lastPinch)dist=Math.max(1,Math.min(10,dist*lastPinch/d));lastPinch=d}});
canvas.addEventListener("wheel",e=>{e.preventDefault();dist=Math.max(1,Math.min(10,dist*Math.exp(e.deltaY*0.001)));auto=false},{passive:false});
</script></body></html>
"""
