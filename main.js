import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.160.1/build/three.module.js';
import jsQR from 'https://cdn.jsdelivr.net/npm/jsqr@1.4.0/+esm';
const video=document.querySelector('#video'), status=document.querySelector('#status'), start=document.querySelector('#start');
const renderer=new THREE.WebGLRenderer({alpha:true,antialias:true});renderer.setPixelRatio(Math.min(devicePixelRatio,2));renderer.domElement.style.objectFit='cover';document.body.insertBefore(renderer.domElement,document.querySelector('#ui'));
const scene=new THREE.Scene(),camera=new THREE.PerspectiveCamera(60,1,0.01,100);camera.position.z=0;
const cube=new THREE.Mesh(new THREE.BoxGeometry(.032,.032,.032),new THREE.MeshNormalMaterial());cube.visible=false;scene.add(cube);
const scanCanvas=document.createElement('canvas'),ctx=scanCanvas.getContext('2d',{willReadFrequently:true});
const QR_SIZE=.06; // 60 mm printed outer QR square
let running=false,lastSeen=0,lastScan=0,detectCount=0;
// This v1 uses planar homography decomposition with approximate camera intrinsics.
// It is an experimental baseline, NOT calibrated production-grade pose tracking.
function poseFromCorners(loc,w,h){
 const pts=[loc.topLeftCorner,loc.topRightCorner,loc.bottomRightCorner,loc.bottomLeftCorner];
 const s=QR_SIZE/2,world=[[-s,-s],[s,-s],[s,s],[-s,s]];
 // Solve 8x8 homography: [u,v,1] ~ H*[X,Y,1]
 const A=[],b=[];for(let i=0;i<4;i++){let [x,y]=world[i],u=pts[i].x,v=pts[i].y;A.push([x,y,1,0,0,0,-u*x,-u*y]);b.push(u);A.push([0,0,0,x,y,1,-v*x,-v*y]);b.push(v)}
 for(let k=0;k<8;k++){let pivot=k;for(let j=k+1;j<8;j++)if(Math.abs(A[j][k])>Math.abs(A[pivot][k]))pivot=j;[A[k],A[pivot]]=[A[pivot],A[k]];[b[k],b[pivot]]=[b[pivot],b[k]];if(Math.abs(A[k][k])<1e-12)return null;let t=A[k][k];for(let j=k;j<8;j++)A[k][j]/=t;b[k]/=t;for(let i=0;i<8;i++)if(i!==k){let f=A[i][k];for(let j=k;j<8;j++)A[i][j]-=f*A[k][j];b[i]-=f*b[k]}}
 const H=[b[0],b[1],b[2],b[3],b[4],b[5],b[6],b[7],1];
 const f=Math.max(w,h)*.95,cx=w/2,cy=h/2;
 const col=(j)=>new THREE.Vector3((H[j]-cx*H[6+j])/f,(H[3+j]-cy*H[6+j])/f,H[6+j]);
 let r1=col(0),r2=col(1),t=col(2);const scale=2/(r1.length()+r2.length());r1.multiplyScalar(scale);r2.multiplyScalar(scale);t.multiplyScalar(scale);
 if(t.z<0){r1.negate();r2.negate();t.negate()}
 r1.normalize();r2.addScaledVector(r1,-r1.dot(r2)).normalize();let r3=new THREE.Vector3().crossVectors(r1,r2).normalize();
 // Image coordinates y down; Three.js y up, camera faces -Z.
 const m=new THREE.Matrix4().makeBasis(new THREE.Vector3(r1.x,-r1.y,-r1.z),new THREE.Vector3(r2.x,-r2.y,-r2.z),new THREE.Vector3(r3.x,-r3.y,-r3.z));
 const q=new THREE.Quaternion().setFromRotationMatrix(m);
 return {position:new THREE.Vector3(t.x,-t.y,-t.z),quaternion:q};
}
const targetPosition=new THREE.Vector3(),targetQuaternion=new THREE.Quaternion();let hasPose=false;
async function init(){try{start.disabled=true;status.textContent='Requesting camera…';const stream=await navigator.mediaDevices.getUserMedia({audio:false,video:{facingMode:{ideal:'environment'},width:{ideal:1280},height:{ideal:720},frameRate:{ideal:30}}});video.srcObject=stream;await video.play();running=true;start.style.display='none';status.textContent='Searching for QR…';requestAnimationFrame(loop)}catch(e){status.textContent='Camera error: '+e.message;start.disabled=false}}
start.onclick=init;
function loop(now){if(!running)return;requestAnimationFrame(loop);const W=innerWidth,H=innerHeight;renderer.setSize(W,H,false);camera.aspect=W/H;camera.updateProjectionMatrix();
 if(video.readyState>=2 && now-lastScan>65){lastScan=now;const vw=video.videoWidth,vh=video.videoHeight;const sw=640,sh=Math.round(sw*vh/vw);scanCanvas.width=sw;scanCanvas.height=sh;ctx.drawImage(video,0,0,sw,sh);const data=ctx.getImageData(0,0,sw,sh);const result=jsQR(data.data,sw,sh,{inversionAttempts:'dontInvert'});
 if(result){const pose=poseFromCorners(result.location,sw,sh);if(pose){targetPosition.copy(pose.position);targetQuaternion.copy(pose.quaternion);lastSeen=now;detectCount++;if(!hasPose){cube.position.copy(targetPosition);cube.quaternion.copy(targetQuaternion);hasPose=true}status.textContent='QR detected · frames: '+detectCount}}}
 if(now-lastSeen>250){hasPose=false;cube.visible=false;status.textContent='Searching for QR…'}
 }
 if(hasPose){cube.visible=true;cube.position.lerp(targetPosition,.42);cube.quaternion.slerp(targetQuaternion,.42);const outward=new THREE.Vector3(0,0,.016).applyQuaternion(cube.quaternion);cube.position.add(outward)}
 renderer.render(scene,camera);
}
