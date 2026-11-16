import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.160.1/build/three.module.js';
import jsQR from 'https://cdn.jsdelivr.net/npm/jsqr@1.4.0/+esm';

// WebSpark AR V2: QR detection + optional OpenCV Lucas-Kanade optical flow,
// PnP pose estimation, adaptive smoothing, explicit tracking loss.
// QR must be a flat, 60 mm wide print. This is experimental, not production tracking.
const $ = (id) => document.getElementById(id);
const video = $('video'), start = $('start'), status = $('status');
const metrics = $('metrics'), errorBox = $('error'), debug = $('debug');
const debugCtx = debug.getContext('2d');
const MARKER_M = 0.060;
const DETECT_MS = 160;
const LOST_MS = 170;
const MAX_REPROJECTION_PX = 12;
const scan = document.createElement('canvas');
const ctx = scan.getContext('2d', {willReadFrequently:true});
const renderer = new THREE.WebGLRenderer({alpha:true,antialias:true});
renderer.setPixelRatio(Math.min(devicePixelRatio,2));
renderer.domElement.id = 'gl';
document.body.insertBefore(renderer.domElement,$('toolbar'));
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(60,1,0.01,30);
const cube = new THREE.Mesh(new THREE.BoxGeometry(.032,.032,.032),new THREE.MeshNormalMaterial());
cube.visible = false;
scene.add(cube);
const axes = new THREE.AxesHelper(.04);axes.visible=false;scene.add(axes);
const poseTarget = {position:new THREE.Vector3(),quaternion:new THREE.Quaternion()};
let cv = null, stream = null, active = false, showDebug=false;
let prevGray = null, tracked = null, corners = null, lastGood=0,lastDetect=0;
let frames=0, fps=0, lastFps=0, lastFrame=0, scanW=0,scanH=0;
let source='NONE', pointsCount=0, detectionCount=0, lastPoseTime=0;
let smoothed=false;

function setError(message){errorBox.textContent=message;console.error(message);}
window.addEventListener('error', e => setError(`JavaScript error: ${e.message}`));
window.addEventListener('unhandledrejection', e => setError(`Unhandled error: ${e.reason}`));
$('toggle').onclick=()=>{showDebug=!showDebug;debug.classList.toggle('hidden',!showDebug);$('toggle').textContent=`Debug: ${showDebug?'on':'off'}`;};

// Load OpenCV independently: failure falls back to QR-only tracking.
async function loadCV(){
  try {
    const script=document.createElement('script');
    script.src='https://docs.opencv.org/4.10.0/opencv.js';
    script.async=true;
    await new Promise((resolve,reject)=>{script.onload=resolve;script.onerror=()=>reject(new Error('OpenCV CDN unavailable'));document.head.appendChild(script);});
    const candidate=window.cv;
    cv = candidate && typeof candidate.then==='function' ? await candidate : candidate;
    if (!cv || !cv.Mat || !cv.calcOpticalFlowPyrLK || !cv.solvePnP) throw new Error('OpenCV API unavailable');
    status.textContent='Ready — QR + optical flow';
  } catch(e){cv=null;status.textContent='Ready — QR only (OpenCV unavailable)';console.warn(e);}
}
const cvPromise=loadCV();

function makeCameraProjection(w,h){
  // Camera intrinsics approximate actual lens: calibrated intrinsics would improve pose.
  const f=Math.max(w,h)*0.95, cx=w/2,cy=h/2;
  // Three camera FOV derived from SAME intrinsics used in solvePnP.
  camera.fov=THREE.MathUtils.radToDeg(2*Math.atan(h/(2*f)));
  camera.aspect=w/h;camera.updateProjectionMatrix();
  return {f,cx,cy};
}
function videoGeometry(){
  const w=video.videoWidth,h=video.videoHeight;
  const screenW=innerWidth,screenH=innerHeight;
  // object-fit:cover cropping offsets for both video and WebGL.
  const scale=Math.max(screenW/w,screenH/h);
  const drawW=w*scale,drawH=h*scale;
  const left=(screenW-drawW)/2,top=(screenH-drawH)/2;
  return {left,top,scale,drawW,drawH};
}
function updateRenderer(){
  if(video.videoWidth){
    const g=videoGeometry();
    renderer.setSize(Math.round(g.drawW),Math.round(g.drawH),false);
    renderer.domElement.style.width=`${g.drawW}px`;
    renderer.domElement.style.height=`${g.drawH}px`;
    renderer.domElement.style.left=`${g.left}px`;
    renderer.domElement.style.top=`${g.top}px`;
    // Three uses full source aspect ratio, with same CSS cover crop as video.
    makeCameraProjection(scanW,scanH);
  }else renderer.setSize(innerWidth,innerHeight,false);
  debug.width=innerWidth;debug.height=innerHeight;
}
function cvMatPoints(pts){return cv.matFromArray(pts.length,1,cv.CV_32FC2,pts.flatMap(p=>[p.x,p.y]));}
function clearTracking(){
  if(prevGray){prevGray.delete();prevGray=null;}
  if(tracked){tracked.delete();tracked=null;}
  corners=null;pointsCount=0;source='NONE';smoothed=false;cube.visible=false;
}
function qrCorners(result){
  const l=result.location;
  return [l.topLeftCorner,l.topRightCorner,l.bottomRightCorner,l.bottomLeftCorner].map(p=>({x:p.x,y:p.y}));
}
function quadValid(pts){
  if(!pts || pts.length!==4 || pts.some(p=>!Number.isFinite(p.x)||!Number.isFinite(p.y)))return false;
  const area=Math.abs(pts.reduce((sum,p,i)=>sum+p.x*pts[(i+1)%4].y-pts[(i+1)%4].x*p.y,0))/2;
  return area>150 && pts.every(p=>p.x>=-4&&p.y>=-4&&p.x<scanW+4&&p.y<scanH+4);
}
function poseFromCorners(pts){
  if(!cv || !quadValid(pts))return null;
  const s=MARKER_M/2;
  // Marker local axes: x right, y up, z outward from printed face.
  const object=cv.matFromArray(4,1,cv.CV_64FC3,[-s,s,0, s,s,0, s,-s,0, -s,-s,0]);
  const image=cv.matFromArray(4,1,cv.CV_64FC2,pts.flatMap(p=>[p.x,p.y]));
  const {f,cx,cy}=makeCameraProjection(scanW,scanH);
  const intrinsic=cv.matFromArray(3,3,cv.CV_64F,[f,0,cx, 0,f,cy, 0,0,1]);
  const distortion=cv.Mat.zeros(4,1,cv.CV_64F);
  const rvec=new cv.Mat(),tvec=new cv.Mat(),rot=new cv.Mat();
  let out=null;
  try{
    if(cv.solvePnP(object,image,intrinsic,distortion,rvec,tvec,false,cv.SOLVEPNP_ITERATIVE)){
      cv.Rodrigues(rvec,rot);
      const R=rot.data64F,T=tvec.data64F;
      // OpenCV (x right,y down,z forward) -> Three camera (x right,y up,z back).
      const m=new THREE.Matrix4().set(R[0],R[1],R[2],0,-R[3],-R[4],-R[5],0,-R[6],-R[7],-R[8],0,0,0,0,1);
      const position=new THREE.Vector3(T[0],-T[1],-T[2]);
      const quaternion=new THREE.Quaternion().setFromRotationMatrix(m).normalize();
      // Reject implausible distance and reprojection outliers.
      const dist=T[2];
      if(dist>0.035 && dist<4){
        const proj=new cv.Mat(),jac=new cv.Mat();
        try{
          cv.projectPoints(object,rvec,tvec,intrinsic,distortion,proj,jac);
          const arr=proj.data64F;
          let err=0;for(let i=0;i<4;i++)err+=Math.hypot(arr[2*i]-pts[i].x,arr[2*i+1]-pts[i].y);
          err/=4;
          if(err<MAX_REPROJECTION_PX)out={position,quaternion,err};
        }finally{proj.delete();jac.delete();}
      }
    }
  }catch(e){console.warn('PnP pose failed',e);}finally{
    object.delete();image.delete();intrinsic.delete();distortion.delete();rvec.delete();tvec.delete();rot.delete();
  }
  return out;
}
// No-OpenCV fallback: rough translation + scale; does NOT claim full 6DoF.
function fallbackPose(pts){
  if(!quadValid(pts))return null;
  const {f,cx,cy}=makeCameraProjection(scanW,scanH);
  const x=pts.reduce((a,p)=>a+p.x,0)/4,y=pts.reduce((a,p)=>a+p.y,0)/4;
  const side=(Math.hypot(pts[1].x-pts[0].x,pts[1].y-pts[0].y)+Math.hypot(pts[2].x-pts[3].x,pts[2].y-pts[3].y))/2;
  const z=MARKER_M*f/Math.max(side,1);
  return {position:new THREE.Vector3((x-cx)*z/f,-(y-cy)*z/f,-z),quaternion:new THREE.Quaternion(),err:0};
}
function acceptCorners(pts,now,kind){
  const pose=cv?poseFromCorners(pts):fallbackPose(pts);
  if(!pose)return false;
  corners=pts;poseTarget.position.copy(pose.position);poseTarget.quaternion.copy(pose.quaternion);
  lastGood=now;source=kind;
  if(!smoothed){cube.position.copy(pose.position);cube.quaternion.copy(pose.quaternion);smoothed=true;}
  return true;
}
function detectQR(data,now){
  const result=jsQR(data.data,scanW,scanH,{inversionAttempts:'attemptBoth'});
  if(!result)return false;
  const pts=qrCorners(result);
  if(!acceptCorners(pts,now,'QR'))return false;
  detectionCount++;
  return true;
}
function initOpticalFlow(gray){
  if(!cv||!corners)return;
  // Seed with corners plus Shi-Tomasi features inside QR polygon.
  let mask=cv.Mat.zeros(scanH,scanW,cv.CV_8UC1);
  const polygon=cv.matFromArray(4,1,cv.CV_32SC2,corners.flatMap(p=>[Math.round(p.x),Math.round(p.y)]));
  const contour=new cv.MatVector();contour.push_back(polygon);
  const features=new cv.Mat();
  try{
    cv.fillPoly(mask,contour,new cv.Scalar(255));
    cv.goodFeaturesToTrack(gray,features,60,0.01,5,mask,3,false,0.04);
    const pts=corners.map(p=>({...p}));
    for(let i=0;i<features.rows;i++)pts.push({x:features.data32F[2*i],y:features.data32F[2*i+1]});
    if(tracked)tracked.delete();tracked=cvMatPoints(pts);pointsCount=pts.length;
    if(prevGray)prevGray.delete();prevGray=gray.clone();
  }finally{mask.delete();polygon.delete();contour.delete();features.delete();}
}
function flowStep(gray,now){
  if(!cv||!prevGray||!tracked||tracked.rows<8||!corners)return false;
  const next=new cv.Mat(),valid=new cv.Mat(),errors=new cv.Mat();
  let homography=null,mask=null,src=null,dst=null,cornerMat=null,transformed=null;
  try{
    cv.calcOpticalFlowPyrLK(prevGray,gray,tracked,next,valid,errors,new cv.Size(21,21),3,
      new cv.TermCriteria(cv.TermCriteria_COUNT+cv.TermCriteria_EPS,25,0.02));
    const old=[],newer=[];
    for(let i=0;i<tracked.rows;i++){
      if(!valid.data[i]||errors.data32F[i]>30)continue;
      const x=next.data32F[2*i],y=next.data32F[2*i+1];
      if(x<0||y<0||x>=scanW||y>=scanH)continue;
      old.push({x:tracked.data32F[2*i],y:tracked.data32F[2*i+1]});
      newer.push({x,y});
    }
    pointsCount=newer.length;
    if(newer.length<8)return false;
    src=cvMatPoints(old);dst=cvMatPoints(newer);mask=new cv.Mat();
    homography=cv.findHomography(src,dst,cv.RANSAC,3,mask);
    if(homography.empty())return false;
    const inliers=Array.from(mask.data).filter(Boolean).length;
    if(inliers<8||inliers/newer.length<0.55)return false;
    cornerMat=cvMatPoints(corners);transformed=new cv.Mat();
    cv.perspectiveTransform(cornerMat,transformed,homography);
    const newCorners=[];
    for(let i=0;i<4;i++)newCorners.push({x:transformed.data32F[2*i],y:transformed.data32F[2*i+1]});
    if(!acceptCorners(newCorners,now,'FLOW'))return false;
    tracked.delete();tracked=cvMatPoints(newer);
    prevGray.delete();prevGray=gray.clone();
    return true;
  }catch(e){console.warn('Optical flow failed',e);return false;}
  finally{
    next.delete();valid.delete();errors.delete();
    for(const m of [homography,mask,src,dst,cornerMat,transformed])if(m)m.delete();
  }
}
function oneEuroLike(alphaDt){
  // Adaptive response: quicker for larger motion; calmer when nearly stationary.
  const translation=cube.position.distanceTo(poseTarget.position);
  const angle=cube.quaternion.angleTo(poseTarget.quaternion);
  const speed=Math.max(translation/.02,angle/.4);
  const cutoff=3+Math.min(18,speed*14);
  return 1-Math.exp(-2*Math.PI*cutoff*alphaDt);
}
function drawDebug(){
  if(!showDebug||!corners||!video.videoWidth)return;
  debugCtx.clearRect(0,0,debug.width,debug.height);
  const g=videoGeometry(),sx=g.drawW/scanW,sy=g.drawH/scanH;
  debugCtx.strokeStyle=source==='QR'?'#2dff81':'#ffd54a';debugCtx.lineWidth=3;
  debugCtx.beginPath();corners.forEach((p,i)=>{const x=g.left+p.x*sx,y=g.top+p.y*sy;if(i===0)debugCtx.moveTo(x,y);else debugCtx.lineTo(x,y);});debugCtx.closePath();debugCtx.stroke();
  debugCtx.fillStyle='#2dff81';for(const p of corners){debugCtx.beginPath();debugCtx.arc(g.left+p.x*sx,g.top+p.y*sy,5,0,Math.PI*2);debugCtx.fill();}
}
function frame(now){
  if(!active)return;
  requestAnimationFrame(frame);
  const dt=Math.min((now-lastFrame)/1000||.016,.1);lastFrame=now;
  frames++;if(now-lastFps>1000){fps=Math.round(frames*1000/(now-lastFps));frames=0;lastFps=now;}
  if(video.readyState>=2){
    ctx.drawImage(video,0,0,scanW,scanH);
    const pixels=ctx.getImageData(0,0,scanW,scanH);
    let gray=null;
    if(cv){
      const rgba=cv.matFromImageData(pixels);gray=new cv.Mat();
      cv.cvtColor(rgba,gray,cv.COLOR_RGBA2GRAY);rgba.delete();
    }
    try{
      let detected=false;
      if(now-lastDetect>=DETECT_MS){lastDetect=now;detected=detectQR(pixels,now);if(detected&&gray)initOpticalFlow(gray);}
      if(!detected&&gray&&corners)flowStep(gray,now);
    }catch(e){setError(`Tracking error: ${e.message}`);clearTracking();}
    finally{if(gray)gray.delete();}
  }
  const good=smoothed&&(now-lastGood<LOST_MS);
  cube.visible=good;
  if(good){
    const a=oneEuroLike(dt);
    cube.position.lerp(poseTarget.position,a);
    cube.quaternion.slerp(poseTarget.quaternion,a);
    // Offset is applied to the geometry ONCE in local marker coordinates, not to cube.position per frame.
    status.textContent=`TRACKING: ${source}`;
  }else{
    status.textContent='Searching for QR…';
    if(now-lastGood>=LOST_MS){clearTracking();}
  }
  metrics.textContent=`FPS: ${fps}\nTracker: ${cv?'OpenCV + jsQR':'jsQR fallback'}\nState: ${good?source:'LOST'}\nFlow points: ${pointsCount}\nQR detections: ${detectionCount}\nLast seen: ${lastGood?Math.round(now-lastGood)+' ms':'never'}`;
  if(showDebug){debugCtx.clearRect(0,0,debug.width,debug.height);drawDebug();}
  renderer.render(scene,camera);
}
async function startCamera(){
  try{
    start.disabled=true;errorBox.textContent='';status.textContent='Requesting camera…';
    if(!navigator.mediaDevices?.getUserMedia)throw new Error('Camera API unavailable. Use HTTPS or localhost.');
    stream=await navigator.mediaDevices.getUserMedia({audio:false,video:{facingMode:{ideal:'environment'},width:{ideal:1280},height:{ideal:720},frameRate:{ideal:30}}});
    video.srcObject=stream;await video.play();
    scanW=640;scanH=Math.round(scanW*video.videoHeight/video.videoWidth);
    scan.width=scanW;scan.height=scanH;
    updateRenderer();window.addEventListener('resize',updateRenderer);
    // Load OpenCV asynchronously; QR-only mode runs while it loads.
    start.style.display='none';active=true;lastFps=performance.now();requestAnimationFrame(frame);
  }catch(e){start.disabled=false;status.textContent='Camera failed';setError(`Camera: ${e.message}`);}
}
start.onclick=startCamera;
status.textContent='Ready — click Start camera';
// Cube base is at marker surface, its centre is 16 mm above it.
cube.geometry.translate(0,0,.016);
