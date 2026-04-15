import * as tf from '@tensorflow/tfjs'

/**
 * Run TensorFlow.js GraphModel inference on a source element.
 *
 * The output format is identical to detectImage() so the rest of the app
 * doesn't care which backend is running.
 *
 * ⚠️  POST-PROCESSING NOTE:
 * The tensor indices and output names below assume a standard YOLOv8 TF export
 * where the model outputs a single tensor of shape [1, num_boxes, 4 + num_classes].
 * If your model's export is different, adjust the parsing in step 3 to match.
 * Run `console.log(model.outputNames)` after loading to inspect actual output names.
 *
 * @param {HTMLImageElement|HTMLVideoElement} source
 * @param {HTMLCanvasElement} canvas
 * @param {tf.GraphModel} model
 * @param {number} topk
 * @param {number} iouThreshold
 * @param {number} scoreThreshold
 * @param {number[]} modelInputShape  [1, 3, H, W]
 * @returns {Promise<Array<{label: number, probability: number, bounding: number[]}>>}
 */
export async function detectImageTF(source, canvas, model, topk, iouThreshold, scoreThreshold, modelInputShape) {
    const srcW = source.videoWidth || source.naturalWidth || source.width
    const srcH = source.videoHeight || source.naturalHeight || source.height
    if (!srcW || !srcH) return []

    const [, modelH, modelW] = modelInputShape

    // ── 1. Pre-process ────────────────────────────────────────────────────────
    const inputTensor = tf.tidy(() => {
        const img = tf.browser.fromPixels(source)               // [H, W, 3]
        const resized = tf.image.resizeBilinear(img, [modelH, modelW]) // [modelH, modelW, 3]
        const normed = resized.div(255.0)
        // ✅ No transpose — TF SavedModel exports expect NHWC: [1, H, W, 3]
        return normed.expandDims(0)
    })

    // ── 2. Inference ─────────────────────────────────────────────────────────
    let rawOutput
    rawOutput = model.execute(inputTensor)
    inputTensor.dispose()

    const outputs = Array.isArray(rawOutput) ? rawOutput : [rawOutput]

// ── 3. Post-process ───────────────────────────────────────────────────────
    const rawShape = outputs[0].shape  // [1, 84, 8400] from TF export

// ✅ YOLOv8 TF exports are [1, 4+classes, num_boxes] (features-first).
//    ONNX exports are [1, num_boxes, 4+classes] (boxes-first).
//    Detect which layout we have: if dim[1] < dim[2] it's features-first → transpose.
    let predTensor
    const needsTranspose = rawShape[1] < rawShape[2]
    if (needsTranspose) {
        predTensor = outputs[0].transpose([0, 2, 1])  // → [1, num_boxes, 4+classes]
        outputs[0].dispose()
    } else {
        predTensor = outputs[0]
    }
    outputs.slice(1).forEach(t => t.dispose())

    const predData = await predTensor.data()
    const [, numBoxes, stride] = predTensor.shape
    predTensor.dispose()

    const numClasses = stride - 4
    const rawBoxes = []

    for (let i = 0; i < numBoxes; i++) {
        const offset = i * stride
        const cx = predData[offset]
        const cy = predData[offset + 1]
        const w = predData[offset + 2]
        const h = predData[offset + 3]

        let maxScore = -Infinity
        let classId = 0
        for (let c = 0; c < numClasses; c++) {
            const s = predData[offset + 4 + c]
            if (s > maxScore) {
                maxScore = s;
                classId = c
            }
        }

        if (maxScore < scoreThreshold) continue

        rawBoxes.push({
            label: classId,
            probability: maxScore,
            bounding: [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]
        })
    }

    // ── 4. Client-side NMS ───────────────────────────────────────────────────
    rawBoxes.sort((a, b) => b.probability - a.probability)
    const kept = []
    const suppressed = new Set()

    for (let i = 0; i < rawBoxes.length; i++) {
        if (suppressed.has(i)) continue
        kept.push(rawBoxes[i])
        if (kept.length >= topk) break
        for (let j = i + 1; j < rawBoxes.length; j++) {
            if (iou(rawBoxes[i].bounding, rawBoxes[j].bounding) > iouThreshold) {
                suppressed.add(j)
            }
        }
    }

    // ── 5. Draw on canvas ─────────────────────────────────────────────────────
    // canvas.width = srcW
    // canvas.height = srcH
    // const ctx = canvas.getContext('2d')
    // ctx.clearRect(0, 0, srcW, srcH)
    //
    // const scaleX = srcW / modelW
    // const scaleY = srcH / modelH
    //
    // for (const box of kept) {
    //     const [x1, y1, x2, y2] = box.bounding
    //     ctx.strokeStyle = '#ff4d00'   // orange — visually distinct from ONNX yellow
    //     ctx.lineWidth = 2
    //     ctx.strokeRect(x1 * scaleX, y1 * scaleY, (x2 - x1) * scaleX, (y2 - y1) * scaleY)
    //     ctx.fillStyle = 'rgba(255,77,0,0.12)'
    //     ctx.fillRect(x1 * scaleX, y1 * scaleY, (x2 - x1) * scaleX, (y2 - y1) * scaleY)
    // }

    return kept
}

function iou(a, b) {
    const ix1 = Math.max(a[0], b[0])
    const iy1 = Math.max(a[1], b[1])
    const ix2 = Math.min(a[2], b[2])
    const iy2 = Math.min(a[3], b[3])
    const inter = Math.max(0, ix2 - ix1) * Math.max(0, iy2 - iy1)
    if (inter === 0) return 0
    const aArea = (a[2] - a[0]) * (a[3] - a[1])
    const bArea = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (aArea + bArea - inter)
}
