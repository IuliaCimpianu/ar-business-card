// Full QR decoding runs off the UI/camera tracking thread.
importScripts('./vendor/jsQR.js');
self.onmessage=({data})=>{
  try {
    const q=self.jsQR(new Uint8ClampedArray(data.buffer),data.width,data.height,{inversionAttempts:'dontInvert'});
    self.postMessage({id:data.id,time:data.time,q:q?{data:q.data,version:q.version,location:q.location}:null});
  } catch(e) {self.postMessage({id:data.id,time:data.time,q:null,error:String(e.message||e)});}
};
