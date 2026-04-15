import * as tf from '@tensorflow/tfjs'

// Module-level reusable pre-scale canvas — avoids per-frame DOM allocation
// and shrinks the frame BEFORE TF touches it, so resizeBilinear works on
// a much smaller input (e.g. 960×960 instead of 1920×1080).
let prescaleCanvas = null;
let prescaleCtx = null;

function getPrescaleCanvas(modelW, modelH) {
    if (!prescaleCanvas) {
        prescaleCanvas = document.createElement('canvas');
        prescaleCtx = prescaleCanvas.getContext('2d', { willReadFrequently: true });
    }
    prescaleCanvas.width = modelW;
    prescaleCanvas.height = modelH;
    return { canvas: prescaleCanvas, ctx: prescaleCtx };
}

/**
 * Run TensorFlow.js GraphModel inference on a source element.
 *
 * The output format is identical to detectImage() so the rest of the app
 * doesn't care which backend is running.
 *
 * @param {HTMLImageElement|HTMLVideoElement|HTMLCanvasElement} source
 * @param {HTMLCanvasElement} canvas
 * @param {tf.GraphModel} model
 * @param {number} topk
 * @param {number} iouThreshold
 * @param {number} scoreThreshold
 * @param {number[]} modelInputShape  [1, H, W, 3] (NHWC) or [1, 3, H, W] (NCHW)
 * @returns {Promise<Array<{label: number, probability: number, bounding: number[]}>>}
 */
export async function detectImageTF(source, canvas, model, topk, iouThreshold, scoreThreshold, modelInputShape) {
    const srcW = source.videoWidth || source.naturalWidth || source.width;
    const srcH = source.videoHeight || source.naturalHeight || source.height;
    if (!srcW || !srcH) return [];

    // modelInputShape can be [1, H, W, 3] (NHWC) or [1, 3, H, W] (NCHW)
    // Detect layout by checking which axis holds 3 (the channel dim)
    const isNHWC = modelInputShape[3] === 3;
    const modelH = isNHWC ? modelInputShape[1] : modelInputShape[2];
    const modelW = isNHWC ? modelInputShape[2] : modelInputShape[3];

    // ── 1. Pre-scale on canvas before TF touches it ───────────────────────────
    // Drawing the source down to model resolution via the 2D API is significantly
    // faster than letting tf.image.resizeBilinear handle a full-res camera frame,
    // especially when TF.js falls back to CPU or WASM.
    const { canvas: pc } = getPrescaleCanvas(modelW, modelH);
    prescaleCtx.drawImage(source, 0, 0, modelW, modelH);

    // ── 2. Pre-process ────────────────────────────────────────────────────────
    const inputTensor = tf.tidy(() => {
        // fromPixels on the already-scaled canvas — tiny tensor, fast path
        const img = tf.browser.fromPixels(pc);    // [modelH, modelW, 3] uint8
        const normed = img.toFloat().div(255.0);  // avoid intermediate resizeBilinear

        if (isNHWC) {
            return normed.expandDims(0);           // [1, H, W, 3]
        } else {
            return normed.transpose([2, 0, 1]).expandDims(0); // [1, 3, H, W]
        }
    });

    // ── 3. Inference ─────────────────────────────────────────────────────────
    let rawOutput;
    try {
        rawOutput = model.execute(inputTensor);
    } finally {
        inputTensor.dispose();
    }

    const outputs = Array.isArray(rawOutput) ? rawOutput : [rawOutput];

    // ── 4. Post-process ───────────────────────────────────────────────────────
    const rawShape = outputs[0].shape; // [1, 84, 8400] or [1, 8400, 84]

    // YOLOv8 TF exports are typically [1, 4+classes, num_boxes] (features-first).
    // Detect: if dim[1] < dim[2] → features-first → transpose to boxes-first.
    let predTensor;
    const needsTranspose = rawShape[1] < rawShape[2];
    if (needsTranspose) {
        predTensor = outputs[0].transpose([0, 2, 1]); // → [1, num_boxes, 4+classes]
        outputs[0].dispose();
    } else {
        predTensor = outputs[0];
    }

    // Dispose any extra output tensors without allocating a slice array
    for (let i = 1; i < outputs.length; i++) outputs[i].dispose();

    const predData = await predTensor.data(); // single GPU readback
    const [, numBoxes, stride] = predTensor.shape;
    predTensor.dispose();

    const numClasses = stride - 4;

    // ── 5. Filter + single-pass argmax ────────────────────────────────────────
    // Replaces two separate loops (Math.max + indexOf) with one pass per box.
    const rawBoxes = [];

    for (let i = 0; i < numBoxes; i++) {
        const offset = i * stride;
        const cx = predData[offset];
        const cy = predData[offset + 1];
        const w  = predData[offset + 2];
        const h  = predData[offset + 3];

        let maxScore = -Infinity;
        let classId  = 0;
        for (let c = 0; c < numClasses; c++) {
            const s = predData[offset + 4 + c];
            if (s > maxScore) { maxScore = s; classId = c; }
        }

        if (maxScore < scoreThreshold) continue;

        rawBoxes.push({
            label: classId,
            probability: maxScore,
            bounding: [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]
        });
    }

    // ── 6. NMS ────────────────────────────────────────────────────────────────
    rawBoxes.sort((a, b) => b.probability - a.probability);
    const kept = [];
    const suppressed = new Uint8Array(rawBoxes.length); // typed array, no Set overhead

    for (let i = 0; i < rawBoxes.length; i++) {
        if (suppressed[i]) continue;
        kept.push(rawBoxes[i]);
        if (kept.length >= topk) break;

        for (let j = i + 1; j < rawBoxes.length; j++) {
            if (!suppressed[j] && iou(rawBoxes[i].bounding, rawBoxes[j].bounding) > iouThreshold) {
                suppressed[j] = 1;
            }
        }
    }

    return kept;
}

function iou(a, b) {
    const ix1 = Math.max(a[0], b[0]);
    const iy1 = Math.max(a[1], b[1]);
    const ix2 = Math.min(a[2], b[2]);
    const iy2 = Math.min(a[3], b[3]);
    const interW = ix2 - ix1;
    const interH = iy2 - iy1;
    if (interW <= 0 || interH <= 0) return 0;
    const inter = interW * interH;
    const aArea = (a[2] - a[0]) * (a[3] - a[1]);
    const bArea = (b[2] - b[0]) * (b[3] - b[1]);
    return inter / (aArea + bArea - inter);
}