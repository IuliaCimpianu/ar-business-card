import * as THREE from './vendor/three.module.js';
import {quadValid} from './tracker.js?v=2.2';
export function intrinsics(w,h,factor=.95) { return {f:Math.max(w,h)*factor,cx:w/2,cy:h/2}; }
export function estimatePose(cv,pts,w,h,markerM,factor=.95) {
  if(!quadValid(pts,w,h))return {reason:'Invalid QR outline'};
  const s=markerM/2,{f,cx,cy}=intrinsics(w,h,factor);
  // IPPE_SQUARE requires this exact ordering: TL, TR, BR, BL, y up.
  const object=cv.matFromArray(4,1,cv.CV_64FC3,[-s,s,0,s,s,0,s,-s,0,-s,-s,0]);
  const image=cv.matFromArray(4,1,cv.CV_64FC2,pts.flatMap(p=>[p.x,p.y]));
  const K=cv.matFromArray(3,3,cv.CV_64F,[f,0,cx,0,f,cy,0,0,1]);
  const D=cv.Mat.zeros(4,1,cv.CV_64F),rvec=new cv.Mat(),tvec=new cv.Mat(),Rmat=new cv.Mat(),projected=new cv.Mat();
  try {
    // Iterative planar initialization is stable at exactly frontal poses as well
    // as oblique views; no previous object offset is fed back into estimation.
    if(!cv.solvePnP(object,image,K,D,rvec,tvec,false,cv.SOLVEPNP_ITERATIVE))return {reason:'solvePnP failed'};
    const T=Array.from(tvec.data64F);
    cv.Rodrigues(rvec,Rmat);
    const R=Rmat.data64F;
    if(T[2]<.025||T[2]>6||!T.every(Number.isFinite))return {reason:'Pose distance outside valid range'};
    if(R[8]>0)return {reason:'Marker face points away from camera'};
    cv.projectPoints(object,rvec,tvec,K,D,projected);
    const arr=projected.data64F.length?projected.data64F:projected.data32F;
    const err=Math.sqrt(pts.reduce((v,p,i)=>v+(arr[2*i]-p.x)**2+(arr[2*i+1]-p.y)**2,0)/4);
    if(!Number.isFinite(err)||err>5)return {reason:`Pose reprojection error ${err.toFixed(1)} px`};
    const rotation=new THREE.Matrix4().set(R[0],R[1],R[2],0,-R[3],-R[4],-R[5],0,-R[6],-R[7],-R[8],0,0,0,0,1);
    return {position:new THREE.Vector3(T[0],-T[1],-T[2]),quaternion:new THREE.Quaternion().setFromRotationMatrix(rotation).normalize(),err};
  } catch(e) {return {reason:`Pose error: ${e.message||e}`};}
  finally {for(const m of [object,image,K,D,rvec,tvec,Rmat,projected])m.delete();}
}
