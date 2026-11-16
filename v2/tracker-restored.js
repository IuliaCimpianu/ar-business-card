// Tracking operates on new camera frames only. Each point retains its marker coordinate.
export const BUILD = 'V2.1 RESTORED — FIX-1';
export const EXPECTED_QR = 'https://iuliacimpianu.github.io/ar-business-card/';
const MIN_POINTS = 10;
const xy = (m, i) => ({ x: m.data32F[2*i], y: m.data32F[2*i+1] });
const distance = (a,b) => Math.hypot(a.x-b.x,a.y-b.y);
const dispose = (...items) => items.forEach(m => m?.delete());
export function quadValid(q,w,h) {
  if (!q || q.length !== 4 || q.some(p => !Number.isFinite(p.x+p.y))) return false;
  let sign = 0, area = 0;
  for (let i=0;i<4;i++) {
    const a=q[i], b=q[(i+1)%4], c=q[(i+2)%4];
    const cross=(b.x-a.x)*(c.y-b.y)-(b.y-a.y)*(c.x-b.x);
    if (Math.abs(cross)<1 || (sign && Math.sign(cross)!==sign)) return false;
    sign=Math.sign(cross); area+=a.x*b.y-b.x*a.y;
    if (distance(a,b)<8 || distance(a,b)>Math.hypot(w,h)*1.5) return false;
  }
  // Corners may leave the image: visible inlier features, rather than all four corners,
  // determine whether a partially visible marker is still measurable.
  return Math.abs(area)/2>180 && q.some(p=>p.x>=0&&p.y>=0&&p.x<w&&p.y<h);
}
export function mapPoint(H,p) {
  const a=H.data64F, z=a[6]*p.x+a[7]*p.y+a[8];
  return {x:(a[0]*p.x+a[1]*p.y+a[2])/z,y:(a[3]*p.x+a[4]*p.y+a[5])/z};
}
const unit = [{x:0,y:0},{x:1,y:0},{x:1,y:1},{x:0,y:1}];
export class MarkerTracker {
  constructor(cv,w,h) { this.cv=cv;this.w=w;this.h=h;this.reset(); }
  points(pts) { return this.cv.matFromArray(pts.length,1,this.cv.CV_32FC2,pts.flatMap(p=>[p.x,p.y])); }
  reset(reason='Show the printed QR') {
    dispose(this.prev,this.tracked);this.prev=null;this.tracked=null;
    this.refs=[];this.corners=null;this.reason=reason;this.lastGood=0;
    this.stats={input:0,forward:0,fb:0,inliers:0,rejected:0};
  }
  seed(gray,corners,now) {
    const cv=this.cv;
    const src=this.points(corners),dst=this.points(unit);
    let inverse=null, mask=null,poly=null,list=null,features=null;
    try {
      inverse=cv.getPerspectiveTransform(src,dst);
      mask=cv.Mat.zeros(this.h,this.w,cv.CV_8UC1);
      poly=cv.matFromArray(4,1,cv.CV_32SC2,corners.flatMap(p=>[Math.round(p.x),Math.round(p.y)]));
      list=new cv.MatVector();list.push_back(poly);cv.fillPoly(mask,list,new cv.Scalar(255));
      features=new cv.Mat();
      cv.goodFeaturesToTrack(gray,features,100,.012,5,mask,3,false,.04);
      const pts=[];
      for(let i=0;i<features.rows;i++) {
        const p=xy(features,i);
        if(p.x>=3&&p.y>=3&&p.x<this.w-3&&p.y<this.h-3)pts.push(p);
      }
      if(pts.length<MIN_POINTS) { this.reset('Too few QR features to start optical flow');return false; }
      dispose(this.prev,this.tracked);
      this.prev=gray.clone();this.tracked=this.points(pts);
      this.refs=pts.map(p=>mapPoint(inverse,p));this.corners=corners;
      this.lastGood=now;this.reason='QR verified';
      this.stats={input:pts.length,forward:pts.length,fb:pts.length,inliers:pts.length,rejected:0};
      return true;
    } finally {dispose(src,dst,inverse,mask,poly,list,features);}
  }
  flow(gray,now) {
    const cv=this.cv;
    if(!this.prev||!this.tracked||this.tracked.rows<MIN_POINTS)return null;
    if(now-this.lastGood>220) {this.reset('Camera frame gap exceeded 220 ms');return null;}
    const next=new cv.Mat(),status=new cv.Mat(),err=new cv.Mat();
    const back=new cv.Mat(),backStatus=new cv.Mat(),backErr=new cv.Mat();
    let fwd=null,src=null,dst=null,H=null,mask=null;
    this.stats={input:this.tracked.rows,forward:0,fb:0,inliers:0,rejected:0};
    const fail=reason=>{this.reason=reason;this.stats.rejected=this.stats.input-this.stats.inliers;return null;};
    try {
      const win=new cv.Size(25,25),term=new cv.TermCriteria(cv.TermCriteria_COUNT+cv.TermCriteria_EPS,30,.01);
      cv.calcOpticalFlowPyrLK(this.prev,gray,this.tracked,next,status,err,win,3,term);
      const candidates=[],indices=[];
      for(let i=0;i<this.tracked.rows;i++) {
        const p=xy(next,i);
        if(status.data[i]&&err.data32F[i]<45&&p.x>=2&&p.y>=2&&p.x<this.w-2&&p.y<this.h-2) {
          candidates.push(p);indices.push(i);
        }
      }
      this.stats.forward=candidates.length;
      if(candidates.length<MIN_POINTS)return fail('LK: fewer than 10 visible features');
      // Only valid forward results are passed into the backwards check.
      fwd=this.points(candidates);
      cv.calcOpticalFlowPyrLK(gray,this.prev,fwd,back,backStatus,backErr,win,3,term);
      const refs=[],pixels=[];
      for(let j=0;j<candidates.length;j++) {
        const i=indices[j];
        if(backStatus.data[j]&&distance(xy(back,j),xy(this.tracked,i))<1.8) {
          refs.push(this.refs[i]);pixels.push(candidates[j]);
        }
      }
      this.stats.fb=pixels.length;
      if(pixels.length<MIN_POINTS)return fail('Forward/backward check: fewer than 10 reliable points');
      src=this.points(refs);dst=this.points(pixels);mask=new cv.Mat();
      H=cv.findHomography(src,dst,cv.RANSAC,2.5,mask,500,.995);
      if(H.empty())return fail('RANSAC: no consistent marker plane');
      const goodRefs=[],goodPixels=[];
      for(let i=0;i<pixels.length;i++)if(mask.data[i]) {goodRefs.push(refs[i]);goodPixels.push(pixels[i]);}
      this.stats.inliers=goodPixels.length;this.stats.rejected=this.stats.input-goodPixels.length;
      if(goodPixels.length<MIN_POINTS||goodPixels.length/pixels.length<.6)return fail('RANSAC: insufficient inliers');
      // Reject a homography inferred from a tiny cluster rather than the marker surface.
      const xs=goodRefs.map(p=>p.x),ys=goodRefs.map(p=>p.y);
      if(Math.max(...xs)-Math.min(...xs)<.25||Math.max(...ys)-Math.min(...ys)<.25)
        return fail('Features cover too little of the QR');
      const corners=unit.map(p=>mapPoint(H,p));
      if(!quadValid(corners,this.w,this.h))return fail('Projected QR quadrilateral is invalid');
      dispose(this.prev,this.tracked);
      this.prev=gray.clone();this.tracked=this.points(goodPixels);this.refs=goodRefs;
      this.corners=corners;this.lastGood=now;this.reason='Optical flow verified';
      return corners;
    } catch(e) {return fail(`Optical flow error: ${e.message||e}`);}
    finally {dispose(next,status,err,back,backStatus,backErr,fwd,src,dst,H,mask);}
  }
}

// Full One Euro filter: derivative estimation + speed-dependent cutoff, updated
// once per measured frame. Filtering the four image corners keeps the anchor local.
export class OneEuro {
  constructor(minCutoff=4,beta=.08) {this.minCutoff=minCutoff;this.beta=beta;this.reset();}
  reset(){this.value=null;this.raw=null;this.derivative=0;this.time=null;}
  filter(x,t) {
    if(this.time===null){this.time=t;this.value=this.raw=x;return x;}
    const dt=Math.max(.001,Math.min(.2,(t-this.time)/1000));
    const alpha=cutoff=>1/(1+1/(2*Math.PI*cutoff*dt));
    const d=(x-this.raw)/dt,ad=alpha(1);
    this.derivative+=ad*(d-this.derivative);
    this.value+=alpha(this.minCutoff+this.beta*Math.abs(this.derivative))*(x-this.value);
    this.raw=x;this.time=t;return this.value;
  }
}
export class CornerFilter {
  constructor(){this.filters=Array.from({length:8},()=>new OneEuro());}
  reset(){this.filters.forEach(f=>f.reset());}
  apply(q,t){return q.map((p,i)=>({x:this.filters[2*i].filter(p.x,t),y:this.filters[2*i+1].filter(p.y,t)}));}
}
