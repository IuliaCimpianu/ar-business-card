export class QrDecoder {
  constructor() {
    this.worker=null;this.busy=false;this.result=null;this.id=0;this.mode='synchronous fallback';
    if(typeof Worker==='function') {
      try {
        this.worker=new Worker(new URL('./qr-worker.js?v=2.2',import.meta.url));this.mode='background worker';
        this.worker.onmessage=({data})=>{if(data.id===this.id){this.busy=false;this.result=data;}};
        this.worker.onerror=()=>{this.worker?.terminate();this.worker=null;this.busy=false;this.result={id:this.id,q:null,error:'QR worker unavailable; using fallback'};this.mode='synchronous fallback';};
      } catch {this.worker=null;}
    }
  }
  scan(pixels,width,height,time) {
    if(this.busy||this.result)return false;
    const id=++this.id;this.busy=true;
    if(this.worker)this.worker.postMessage({id,width,height,time,buffer:pixels.data.buffer},[pixels.data.buffer]);
    else {
      try {const q=window.jsQR(pixels.data,width,height,{inversionAttempts:'dontInvert'});this.result={id,time,q};}
      catch(e){this.result={id,time,q:null,error:String(e.message||e)};}
      this.busy=false;
    }
    return true;
  }
  take(){const r=this.result;this.result=null;return r;}
  cancel(){this.id++;this.busy=false;this.result=null;}
  dispose(){this.worker?.terminate();this.worker=null;this.cancel();}
}
