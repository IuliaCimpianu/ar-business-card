// Exact module pattern of the existing printed QR (version 4, 33 modules).
// Used to verify the marker's current appearance, including when decoding fails.
export const PATTERN = [
  '111111101110011110101001001111111',
  '100000100010111000110100001000001',
  '101110100100101001101101001011101',
  '101110101110000011000001001011101',
  '101110101111100011010001001011101',
  '100000101100111101001100001000001',
  '111111101010101010101010101111111',
  '000000001100001001011101100000000',
  '100010111101000101000111111111001',
  '101110000010101101100001010001100',
  '111001110001111110100111101001010',
  '010110010110110010011101010000010',
  '000011100000010010000111101011011',
  '111011001101001000111111110100100',
  '001011100001011100101111111001110',
  '111100001111011111100100011111000',
  '101111110100101001011110101010010',
  '000100000100101110100001010001110',
  '011001110110100101001011011101010',
  '111111000010001101100111101100011',
  '010000101111010000111111001111000',
  '101100001000111011010101010101010',
  '001111110110010011010101101010010',
  '000000000110000001111100010010010',
  '111101100111000111001110111110011',
  '000000001110001111000010100010110',
  '111111101001001110101100101011000',
  '100000100101001110000110100010011',
  '101110101111100110001100111111000',
  '101110100100011000111001101110100',
  '101110100100010101101110011111100',
  '100000100001000111011110110000000',
  '111111101100110001001111101001001',
];
export function verifyMarker(gray,corners,w,h,cv) {
  const src=cv.matFromArray(4,1,cv.CV_32FC2,[0,0,1,0,1,1,0,1]);
  const dst=cv.matFromArray(4,1,cv.CV_32FC2,corners.flatMap(p=>[p.x,p.y]));
  let H;
  try {
    H=cv.getPerspectiveTransform(src,dst);const a=H.data64F,n=PATTERN.length;
    const samples=[];let sumBlack=0,sumWhite=0,numBlack=0,numWhite=0;
    for(let y=0;y<n;y++)for(let x=0;x<n;x++) {
      // Exclude the shared finder/timing structures. Identity is checked from data.
      if((x<9&&y<9)||(x>n-10&&y<9)||(x<9&&y>n-10)||x===6||y===6)continue;
      const u=(x+.5)/n,v=(y+.5)/n,z=a[6]*u+a[7]*v+a[8];
      const px=(a[0]*u+a[1]*v+a[2])/z,py=(a[3]*u+a[4]*v+a[5])/z;
      if(px<1||py<1||px>=w-2||py>=h-2)continue;
      const ix=Math.floor(px),iy=Math.floor(py),fx=px-ix,fy=py-iy,k=iy*w+ix,g=gray.data;
      const value=g[k]*(1-fx)*(1-fy)+g[k+1]*fx*(1-fy)+g[k+w]*(1-fx)*fy+g[k+w+1]*fx*fy;
      const black=PATTERN[y][x]==='1';samples.push({value,black});
      if(black){sumBlack+=value;numBlack++;}else{sumWhite+=value;numWhite++;}
    }
    if(numBlack<70||numWhite<70||samples.length<250)return {ok:false,score:0,contrast:0,reason:'Too little visible QR data'};
    const blackMean=sumBlack/numBlack,whiteMean=sumWhite/numWhite,contrast=whiteMean-blackMean;
    const threshold=(blackMean+whiteMean)/2;
    const score=samples.filter(s=>(s.value<threshold)===s.black).length/samples.length;
    const ok=contrast>24&&score>=.83;
    return {ok,score,contrast,count:samples.length,reason:ok?'Printed QR pattern verified':`QR pattern mismatch (${Math.round(score*100)}%, contrast ${Math.round(contrast)})`};
  } finally {src.delete();dst.delete();H?.delete();}
}

// ECC aligns the known printed image with the current frame. This is a new
// measurement of QR pixels, not prediction or a frozen pose during loss.
export class PatternRefiner {
  constructor(cv) {
    this.cv=cv;this.size=132;this.template=new cv.Mat(this.size,this.size,cv.CV_8UC1);
    const n=PATTERN.length;
    for(let y=0;y<this.size;y++)for(let x=0;x<this.size;x++)
      this.template.data[y*this.size+x]=PATTERN[Math.floor(y*n/this.size)][Math.floor(x*n/this.size)]==='1'?0:255;
    this.smooth=new cv.Mat();cv.GaussianBlur(this.template,this.smooth,new cv.Size(3,3),.8,.8,cv.BORDER_REPLICATE);
  }
  refine(gray,corners,w,h) {
    const cv=this.cv,N=this.size;
    const src=cv.matFromArray(4,1,cv.CV_32FC2,[0,0,N,0,N,N,0,N]);
    const dst=cv.matFromArray(4,1,cv.CV_32FC2,corners.flatMap(p=>[p.x,p.y]));
    let H,inv,patch,smoothed,correction,mask;
    try {
      H=cv.getPerspectiveTransform(src,dst);inv=new cv.Mat();cv.invert(H,inv);
      patch=new cv.Mat();smoothed=new cv.Mat();
      cv.warpPerspective(gray,patch,inv,new cv.Size(N,N),cv.INTER_LINEAR,cv.BORDER_CONSTANT,new cv.Scalar(127));
      cv.GaussianBlur(patch,smoothed,new cv.Size(3,3),.8,.8,cv.BORDER_REPLICATE);
      correction=cv.Mat.eye(3,3,cv.CV_32F);mask=new cv.Mat();
      const score=cv.findTransformECC(this.smooth,smoothed,correction,cv.MOTION_HOMOGRAPHY,
        new cv.TermCriteria(cv.TermCriteria_COUNT+cv.TermCriteria_EPS,18,.001),mask,3);
      if(!Number.isFinite(score)||score<.65)return null;
      const a=H.data64F,b=correction.data32F;
      const q=[{x:0,y:0},{x:N,y:0},{x:N,y:N},{x:0,y:N}].map(p=>{
        const z=b[6]*p.x+b[7]*p.y+b[8];
        const u=(b[0]*p.x+b[1]*p.y+b[2])/z,v=(b[3]*p.x+b[4]*p.y+b[5])/z;
        const zz=a[6]*u+a[7]*v+a[8];return {x:(a[0]*u+a[1]*v+a[2])/zz,y:(a[3]*u+a[4]*v+a[5])/zz};
      });
      const maxShift=Math.max(...q.map((p,i)=>Math.hypot(p.x-corners[i].x,p.y-corners[i].y)));
      const verification=verifyMarker(gray,q,w,h,cv);
      if(maxShift>22||!verification.ok)return null;
      return {corners:q,verification,ecc:score};
    } catch {return null;} // A failed registration never supplies a pose.
    finally {for(const m of [src,dst,H,inv,patch,smoothed,correction,mask])m?.delete();}
  }
  dispose(){this.template.delete();this.smooth.delete();}
}
