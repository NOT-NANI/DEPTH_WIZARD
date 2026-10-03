#!/usr/bin/env python3
"""Generate a small self-contained data HTML viewer (Three.js loads from CDN)."""
import argparse, base64, io, json, sys
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument("--height",type=Path,default=ROOT/"outputs/gamus_aligned_height.npy")
parser.add_argument("--rgb",type=Path,default=ROOT/"gamus_rgb.png")
parser.add_argument("--output",type=Path,default=ROOT/"outputs/terrain_viewer.html")
parser.add_argument("--max-size",type=int,default=256)
args=parser.parse_args()
def abs_path(p): return p if p.is_absolute() else (Path.cwd()/p).resolve()
try:
    h=np.load(abs_path(args.height),allow_pickle=False).astype(np.float32)
    h=Image.fromarray(h).resize((args.max_size,args.max_size),Image.Resampling.BILINEAR)
    h=np.asarray(h,dtype=np.float32)
    image=Image.open(abs_path(args.rgb)).convert("RGB").resize((args.max_size,args.max_size),Image.Resampling.BILINEAR)
    buf=io.BytesIO(); image.save(buf,format="PNG"); texture=base64.b64encode(buf.getvalue()).decode()
    payload=json.dumps(h.flatten().tolist(),separators=(",",":"))
    html=f'''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>DepthWizard terrain preview</title><style>body{{margin:0;background:#111;color:#eee;font:14px system-ui}}#info{{position:fixed;z-index:2;top:12px;left:14px;background:#111c;padding:12px;line-height:1.5}}canvas{{display:block}}input{{width:130px}}</style></head><body><div id="info"><b>DepthWizard terrain preview</b><br>Sample-fitted GAMUS surface — not deployment-calibrated elevation<br>Drag: rotate · wheel: zoom · W/A/S/D: fly · click: inspect height<br>Height exaggeration <input id="ex" type="range" min="0.2" max="5" value="1" step="0.1"><span id="readout"></span></div><script type="module">
import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js';import {{OrbitControls}} from 'https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/controls/OrbitControls.js';
const N={args.max_size},H={payload},im=new Image();im.src='data:image/png;base64,{texture}';await im.decode();const c=document.createElement('canvas');c.width=c.height=N;const cx=c.getContext('2d');cx.drawImage(im,0,0);const pix=cx.getImageData(0,0,N,N).data;
const scene=new THREE.Scene();scene.background=new THREE.Color(0x101820);const camera=new THREE.PerspectiveCamera(55,innerWidth/innerHeight,.1,2000);camera.position.set(N*.8,N*.7,N*.8);const renderer=new THREE.WebGLRenderer({{antialias:true}});renderer.setSize(innerWidth,innerHeight);document.body.appendChild(renderer.domElement);const controls=new OrbitControls(camera,renderer.domElement);controls.target.set(0,0,0);controls.enableDamping=true;
const geom=new THREE.PlaneGeometry(N,N,N-1,N-1);geom.rotateX(-Math.PI/2);const pos=geom.attributes.position,col=[];let lo=H.reduce((a,b)=>Math.min(a,b),Infinity),hi=H.reduce((a,b)=>Math.max(a,b),-Infinity);for(let i=0;i<H.length;i++){{pos.setY(i,(H[i]-lo)*.2);col.push(pix[i*4]/255,pix[i*4+1]/255,pix[i*4+2]/255)}}geom.setAttribute('color',new THREE.Float32BufferAttribute(col,3));geom.computeVertexNormals();const mesh=new THREE.Mesh(geom,new THREE.MeshStandardMaterial({{vertexColors:true,side:THREE.DoubleSide,roughness:1}}));scene.add(mesh);scene.add(new THREE.HemisphereLight(0xffffff,0x334455,2));const light=new THREE.DirectionalLight(0xffffff,1.5);light.position.set(0,200,100);scene.add(light);
const ray=new THREE.Raycaster(),mouse=new THREE.Vector2(),read=document.querySelector('#readout');renderer.domElement.onclick=e=>{{mouse.x=e.clientX/innerWidth*2-1;mouse.y=-(e.clientY/innerHeight)*2+1;ray.setFromCamera(mouse,camera);const hit=ray.intersectObject(mesh)[0];if(hit){{const x=Math.floor(hit.uv.x*(N-1)),y=Math.floor((1-hit.uv.y)*(N-1)),z=H[y*N+x];read.textContent=' · '+x+','+y+': '+z.toFixed(2)+' m'}}}};
document.querySelector('#ex').oninput=e=>{{const k=Number(e.target.value);for(let i=0;i<H.length;i++)pos.setY(i,(H[i]-lo)*.2*k);pos.needsUpdate=true;geom.computeVertexNormals()}};const keys={{}};onkeydown=e=>keys[e.key.toLowerCase()]=true;onkeyup=e=>keys[e.key.toLowerCase()]=false;function animate(){{requestAnimationFrame(animate);const v=.8;if(keys.w)camera.translateZ(-v);if(keys.s)camera.translateZ(v);if(keys.a)camera.translateX(-v);if(keys.d)camera.translateX(v);controls.update();renderer.render(scene,camera)}}animate();onresize=()=>{{camera.aspect=innerWidth/innerHeight;camera.updateProjectionMatrix();renderer.setSize(innerWidth,innerHeight)}};
</script></body></html>'''
    out=abs_path(args.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(html)
    print(f"Viewer: {out}\nGrid: {args.max_size} × {args.max_size}; surface range {float(h.min()):.3f}–{float(h.max()):.3f}")
except Exception as exc: parser.exit(1,f"Error: {exc}\n")
