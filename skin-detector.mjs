// MediaPipe's multiclass model distinguishes body/face skin from clothes and background.
// Only model/runtime files are downloaded; image pixels never leave this browser.
const runtime='https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.21';
const model='https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/1/selfie_multiclass_256x256.tflite';
let detectorPromise;
function getDetector() {
  if(!detectorPromise) detectorPromise=(async()=>{
    const {FilesetResolver,ImageSegmenter}=await import(runtime+'/vision_bundle.mjs');
    const files=await FilesetResolver.forVisionTasks(runtime+'/wasm');
    return ImageSegmenter.createFromOptions(files,{
      baseOptions:{modelAssetPath:model,delegate:'CPU'},runningMode:'IMAGE',
      outputCategoryMask:true,outputConfidenceMasks:false
    });
  })().catch(error=>{detectorPromise=null;throw error;});
  return detectorPromise;
}
export async function detectSkinMask(image,isCurrent=()=>true) {
  const detector=await getDetector();
  if(!isCurrent()) throw new Error('Image changed');
  await new Promise(resolve=>requestAnimationFrame(resolve));
  if(!isCurrent()) throw new Error('Image changed');
  let mask;
  detector.segment(image,result=>{
    const category=result.categoryMask;
    if(!category) throw new Error('No category mask');
    const categories=category.getAsUint8Array(),data=new Uint8Array(categories.length);
    for(let i=0;i<data.length;i++) data[i]=categories[i]===2||categories[i]===3?1:0;
    mask={width:category.width,height:category.height,data};
  });
  if(!mask) throw new Error('No skin mask');
  return mask;
}
