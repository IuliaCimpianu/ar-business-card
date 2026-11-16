import * as THREE from './vendor/three.module.js';
import {MarkerTracker,CornerFilter,BUILD,EXPECTED_QR} from './tracker.js?v=2.2';
import {estimatePose,intrinsics} from './pose.js?v=2.2';
import {verifyMarker} from './marker-pattern.js?v=2.2';
import {QrDecoder} from './qr-decoder.js?v=2.2';
const $=id=>document.getElementById(id);
const video=$('video'),status=$('status'),metrics=$('metrics'),error=$('error'),start=$('start');
const scan=document.createElement('canvas'),ctx=scan.getContext('2d',{willReadFrequently:true});
const debug=$('debug'),dc=debug.getContext('2d');
const renderer=new THREE.WebGLRenderer({alpha:true,antialias:true});
renderer.setPixelRatio(Math.min(devicePixelRatio,1.5));renderer.domElement.id='gl';
// CSS uses exactly the same object-fit cover transform for camera and renderer.
document.body.insertBefore(renderer.domElement,$('toolbar'));
const scene=new THREE.Scene(),camera=new THREE.PerspectiveCamera(60,1,.005,20);
const anchor=new THREE.Group();scene.add(anchor);anchor.visible=false;
const cube=new THREE.Mesh(new THREE.BoxGeometry(.026,.026,.026),new THREE.MeshNormalMaterial());
cube.position.z=.013;anchor.add(cube); // Fixed local offset, applied exactly once.
const edges=new THREE.LineSegments(new THREE.EdgesGeometry(cube.geometry),new THREE.LineBasicMaterial({color:0x10182a}));
edges.position.copy(cube.position);anchor.add(edges);
const cornerFilter=new CornerFilter();
const decoder=new QrDecoder();let decodeFrame=null;
let cv,tracker,stream,active=false,showDebug=false,scanW=0,scanH=0;
let generation=0,videoFrameHandle=null,raf=null,lastVideoTime=-1;
let lastDetect=-Infinity,lastQr=-Infinity,lastVerified=-Infinity,lastMeasured=-Infinity,lastRender=-Infinity;
let frameCount=0,fps=0,fpsStart=0,source='SEARCHING',reason='Press Start camera';
let qrCount=0,poseError=0,elapsed=0,lastMetric=0,lossReason='—',flowFailures=0;
let rawCorners=null,displayCorners=null;
// The old 60 mm print includes four quiet-zone modules on each side. jsQR
// corners enclose the symbol only. Version is updated from the decoded marker.
let version=4;
const symbolMetres=()=>.060*(17+4*version)/(25+4*version);
const setError=e=>{error.textContent=String(e);console.error(e);};
function resetMeasurement(message) {
  anchor.visible=false;cornerFilter.reset();displayCorners=null;source='SEARCHING';reason=message;
}
function readCv() {
  return new Promise((resolve,reject)=>{
    let done=false,timer,poll;
    const finish=(e)=>{
      if(done)return;done=true;clearTimeout(timer);clearInterval(poll);
      if(e)reject(e);else resolve({cv:window.cv}); // Never assimilate OpenCV's self-thenable.
    };
    const ready=()=>{
      const c=window.cv;
      if(!c?.Mat)return;
      const required=['calcOpticalFlowPyrLK','findHomography','solvePnP','goodFeaturesToTrack','getPerspectiveTransform','findTransformECC','warpPerspective','ORB','BFMatcher'];
      const missing=required.filter(n=>typeof c[n]!=='function');
      finish(missing.length?new Error(`OpenCV missing: ${missing.join(', ')}`):null);
    };
    timer=setTimeout(()=>finish(new Error('OpenCV did not initialise within 30 seconds. Reload the page.')),30000);
    poll=setInterval(ready,50);
    const script=document.createElement('script');script.src='./vendor/opencv.js';
    script.onload=()=>{
      ready();
      if(!done&&window.cv) {
        const previous=window.cv.onRuntimeInitialized;
        window.cv.onRuntimeInitialized=()=>{previous?.();ready();};
      }
    };
    script.onerror=()=>finish(new Error('Cannot load v2/vendor/opencv.js. Upload the complete v2 folder.'));
    document.head.appendChild(script);
  });
}
let cvReady=readCv();
// Attach immediately so a failed preload is handled before the user presses Start.
cvReady.catch(e=>{reason='OpenCV unavailable';setError(e.message);});
function geometry() {
  const w=video.videoWidth||1,h=video.videoHeight||1,s=Math.max(innerWidth/w,innerHeight/h);
  return {width:w*s,height:h*s,left:(innerWidth-w*s)/2,top:(innerHeight-h*s)/2};
}
function resize() {
  const g=geometry();renderer.setSize(Math.round(g.width),Math.round(g.height),false);
  Object.assign(renderer.domElement.style,{width:`${g.width}px`,height:`${g.height}px`,left:`${g.left}px`,top:`${g.top}px`});
  debug.width=innerWidth;debug.height=innerHeight;
  if(scanW){const {f}=intrinsics(scanW,scanH);camera.fov=THREE.MathUtils.radToDeg(2*Math.atan(scanH/(2*f)));camera.aspect=scanW/scanH;camera.updateProjectionMatrix();}
}
function applyDecoded(result,gray,now,current) {
  const job=decodeFrame;decodeFrame=null;
  if(!job)return null;
  try {
    const q=result.q;
    if(!q||job.generation!==generation||now-job.time>220)return null;
    if(q.data!==EXPECTED_QR){
      // A result from an earlier frame never overrides a currently verified QR.
      if(current&&tracker.pattern?.ok)return {verifiedOnly:true};
      reason='Different QR: show the original printed WebSpark test marker';return {wrong:true};
    }
    const l=q.location,pts=[l.topLeftCorner,l.topRightCorner,l.bottomRightCorner,l.bottomLeftCorner].map(p=>({x:p.x,y:p.y}));
    version=q.version;lastQr=job.time;qrCount++;
    if(current&&tracker.pattern?.ok)return {verifiedOnly:true};
    // Reproject a delayed decoder result onto this CURRENT camera frame using
    // optical flow. No stale decoder coordinates are displayed directly.
    if(!tracker.seed(job.gray,pts,job.time))return null;
    const aligned=job.time===now?pts:tracker.flow(gray,now);
    if(!aligned)return null;
    const check=verifyMarker(gray,aligned,scanW,scanH,cv);
    if(!check.ok)return null;
    tracker.pattern=check;lastVerified=now;return aligned;
  } finally {job.gray.delete();}
}
function scanQR(pixels,gray,now) {
  if(decoder.busy||decodeFrame)return;
  const snapshot=gray.clone();
  if(decoder.scan(pixels,scanW,scanH,now)) {
    lastDetect=now;decodeFrame={gray:snapshot,time:now,generation};
  } else snapshot.delete();
}
function clearDecoder() {decoder.cancel();decodeFrame?.gray.delete();decodeFrame=null;}

function processFrame(now) {
  if(!active||video.readyState<2||document.hidden)return;
  const t=performance.now();
  if(video.videoWidth&&Math.abs(video.videoWidth/video.videoHeight-scanW/scanH)>.01){configureFrames();return;}
  ctx.drawImage(video,0,0,scanW,scanH);
  const pixels=ctx.getImageData(0,0,scanW,scanH),rgba=cv.matFromImageData(pixels),gray=new cv.Mat();
  let pts=null,kind='FLOW';
  try {
    cv.cvtColor(rgba,gray,cv.COLOR_RGBA2GRAY);
    // Flow always runs first on this new frame. A missed QR decode never resets it.
    if(tracker.corners){pts=tracker.flow(gray,now);if(!pts){lossReason=tracker.reason;flowFailures++;}}
    if(!pts){const recovered=tracker.recover(gray,now);if(recovered){pts=recovered;kind='FEATURES';}}
    const consume=result=>{
      const detected=applyDecoded(result,gray,now,pts);
      if(detected?.wrong){pts=null;lossReason=reason;tracker.reset(reason);}
      else if(detected&&!detected.verifiedOnly){pts=detected;kind='QR';}
    };
    const completed=decoder.take();if(completed)consume(completed);
    const interval=pts?(tracker.stats.inliers<50||tracker.pattern?.score<.9?120:450):100;
    if(now-lastDetect>=interval||(!pts&&tracker.corners)) {
      scanQR(pixels,gray,now);
      // Compatibility fallback can return synchronously; the worker never blocks.
      const immediate=decoder.take();if(immediate)consume(immediate);
    }
    // The exact printed data pattern is checked in the CURRENT frame. This can
    // verify identity at oblique/partial views where the decoder cannot read it.
    if(pts&&tracker.pattern?.ok)lastVerified=now;
    if(pts&&now-lastVerified>250){pts=null;lossReason='Current QR appearance could not be verified';tracker.reset(lossReason);}
    if(pts) {
      if(!anchor.visible||now-lastMeasured>150)cornerFilter.reset();
      let filtered=cornerFilter.apply(pts,now);
      let pose=estimatePose(cv,filtered,scanW,scanH,symbolMetres());
      // Independent corner smoothing can temporarily distort a tilted square.
      // Fall back to the measured, verified quad rather than hiding a valid pose.
      if(!pose.position){const raw=estimatePose(cv,pts,scanW,scanH,symbolMetres());if(raw.position){pose=raw;filtered=pts;cornerFilter.reset();}}
      if(pose.position) {
        anchor.position.copy(pose.position);anchor.quaternion.copy(pose.quaternion);anchor.visible=true;
        rawCorners=pts;displayCorners=filtered;lastMeasured=now;source=kind;
        reason=kind==='QR'?'QR decoded and verified':'Measured movement from QR features';poseError=pose.err;
      } else {lossReason=pose.reason;resetMeasurement(pose.reason);}
    } else {
      resetMeasurement(reason.startsWith('Different QR')?reason:lossReason==='—'?'Searching for the printed QR':lossReason);
      if(now-tracker.lastGood>220&&tracker.corners)tracker.reset(reason);
    }
  } catch(e) {lossReason=`Tracking error: ${e.message||e}`;tracker.reset(lossReason);resetMeasurement(lossReason);setError(lossReason);}
  finally {rgba.delete();gray.delete();}
  elapsed=performance.now()-t;frameCount++;
  if(now-fpsStart>1000){fps=Math.round(frameCount*1000/(now-fpsStart));frameCount=0;fpsStart=now;}
  lastRender=now;renderer.render(scene,camera);paintDebug();updateMetrics(now);
  // Exposes read-only diagnostic snapshots for reproducible tests, with no camera data.
  window.websparkDiagnostics={build:BUILD,state:source,reason,lossReason,flowFailures,qrCount,fps,processingMs:elapsed,decoder:decoder.mode,poseError,pattern:tracker.pattern,points:{...tracker.stats},rawCorners,displayCorners,visible:anchor.visible};
}
function updateMetrics(now,force=false) {
  if(!force&&now-lastMetric<150)return;lastMetric=now;
  status.textContent=anchor.visible?`TRACKING: ${source}`:'SEARCHING';
  const s=tracker?.stats||{input:0,forward:0,fb:0,inliers:0,rejected:0};
  metrics.textContent=`${BUILD}\nOpenCV: ${cv?'READY — optical flow enabled':'loading'}\nQR decoder: ${decoder.mode}\nState: ${source} | ${fps} camera FPS | ${elapsed.toFixed(0)} ms/frame\nPoints: ${s.input} → LK ${s.forward} → FB ${s.fb} → RANSAC ${s.inliers}\nRejected: ${s.rejected} | QR reads: ${qrCount}\nPose error: ${poseError.toFixed(2)} px | QR pattern: ${Math.round((tracker?.pattern?.score||0)*100)}%\n${reason}\nLast flow loss: ${lossReason}`;
}
function paintDebug() {
  dc.clearRect(0,0,debug.width,debug.height);
  if(!showDebug||!anchor.visible||!rawCorners)return;
  const g=geometry(),sx=g.width/scanW,sy=g.height/scanH;
  const path=(pts,color,width)=>{dc.strokeStyle=color;dc.lineWidth=width;dc.beginPath();pts.forEach((p,i)=>{const x=g.left+p.x*sx,y=g.top+p.y*sy;i?dc.lineTo(x,y):dc.moveTo(x,y);});dc.closePath();dc.stroke();};
  path(rawCorners,'#ffce55',1);path(displayCorners,'#3bff9b',2);
  if(tracker.tracked)for(let i=0;i<tracker.tracked.rows;i++) {
    const a=tracker.tracked.data32F;dc.fillStyle='#3bff9b';dc.fillRect(g.left+a[2*i]*sx-2,g.top+a[2*i+1]*sy-2,4,4);
  }
}
function configureFrames() {
  const scale=Math.min(1,720/Math.max(video.videoWidth,video.videoHeight));
  scanW=Math.round(video.videoWidth*scale);scanH=Math.round(video.videoHeight*scale);
  scan.width=scanW;scan.height=scanH;clearDecoder();tracker?.dispose();tracker=new MarkerTracker(cv,scanW,scanH);
  lastDetect=lastQr=lastVerified=-Infinity;resetMeasurement('Show the printed QR');resize();
}
function scheduleFrame(token) {
  if(!active||token!==generation)return;
  if(video.requestVideoFrameCallback)videoFrameHandle=video.requestVideoFrameCallback(()=>{
    if(!active||token!==generation)return;
    processFrame(performance.now());scheduleFrame(token);
  });
  else raf=requestAnimationFrame(()=>{
    if(!active||token!==generation)return;
    if(video.currentTime!==lastVideoTime){lastVideoTime=video.currentTime;processFrame(performance.now());}
    scheduleFrame(token);
  });
}
async function startCamera() {
  const token=++generation;
  try {
    start.disabled=true;error.textContent='';status.textContent='Loading OpenCV…';
    ({cv}=await cvReady);
    if(token!==generation)return;
    if(typeof window.jsQR!=='function')throw new Error('Cannot load v2/vendor/jsQR.js. Upload the complete v2 folder.');
    if(!navigator.mediaDevices?.getUserMedia)throw new Error('Camera requires HTTPS or localhost.');
    status.textContent='Allow access to camera…';
    const candidate=await navigator.mediaDevices.getUserMedia({audio:false,video:{facingMode:{ideal:'environment'},width:{ideal:1280},height:{ideal:720},frameRate:{ideal:30}}});
    if(token!==generation){candidate.getTracks().forEach(t=>t.stop());return;}
    stream=candidate;video.srcObject=stream;await video.play();
    configureFrames();active=true;fpsStart=performance.now();frameCount=0;lastRender=fpsStart;
    start.hidden=true;$('stop').hidden=false;scheduleFrame(token);updateMetrics(performance.now(),true);
  } catch(e) {stopCamera();status.textContent='Start failed';setError(`${e.name||'Error'}: ${e.message||e}`);}
  finally {start.disabled=false;}
}
function stopCamera() {
  active=false;generation++;
  if(videoFrameHandle!==null)video.cancelVideoFrameCallback?.(videoFrameHandle);
  if(raf!==null)cancelAnimationFrame(raf);
  stream?.getTracks().forEach(t=>t.stop());stream=null;video.srcObject=null;
  clearDecoder();tracker?.reset();resetMeasurement('Camera stopped');renderer.render(scene,camera);dc.clearRect(0,0,debug.width,debug.height);
  start.hidden=false;$('stop').hidden=true;start.disabled=false;status.textContent='Camera stopped';
}
start.onclick=startCamera;$('stop').onclick=stopCamera;
$('toggle').onclick=()=>{showDebug=!showDebug;debug.classList.toggle('hidden',!showDebug);$('toggle').textContent=`Debug: ${showDebug?'on':'off'}`;paintDebug();};
$('copy').onclick=async()=>{
  try{await navigator.clipboard.writeText(JSON.stringify(window.websparkDiagnostics||{build:BUILD,reason},null,2));$('copy').textContent='Copied';setTimeout(()=>$('copy').textContent='Copy diagnostics',1800);}
  catch{setError('Could not copy. The diagnostics are also visible on screen.');}
};
window.addEventListener('resize',resize);window.addEventListener('pagehide',stopCamera);
document.addEventListener('visibilitychange',()=>{
  if(document.hidden){clearDecoder();tracker?.reset('Tab hidden');resetMeasurement('Tab hidden');renderer.render(scene,camera);}
});
// Hide the last rendered pose if camera frames stop; no frozen anchor is held.
setInterval(()=>{if(active&&performance.now()-lastRender>180){resetMeasurement('No new camera frames');renderer.render(scene,camera);updateMetrics(performance.now(),true);
    if(window.websparkDiagnostics)Object.assign(window.websparkDiagnostics,{visible:false,state:source,reason});}},80);
status.textContent='Ready — Start camera';
