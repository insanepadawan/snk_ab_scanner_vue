import { Tensor } from "onnxruntime-web";

// Module-level reusable canvases — allocated once, never GC'd
let preprocessCanvas = null;
let preprocessCtx = null;
let videoCanvas = null;
let videoCtx = null;

function getPreprocessCanvas(w, h) {
    if (!preprocessCanvas) {
        preprocessCanvas = document.createElement("canvas");
        preprocessCtx = preprocessCanvas.getContext("2d", { willReadFrequently: true });
    }
    preprocessCanvas.width = w;
    preprocessCanvas.height = h;
    return { canvas: preprocessCanvas, ctx: preprocessCtx };
}

function getVideoCanvas(w, h) {
    if (!videoCanvas) {
        videoCanvas = document.createElement("canvas");
        videoCtx = videoCanvas.getContext("2d");
    }
    videoCanvas.width = w;
    videoCanvas.height = h;
    return { canvas: videoCanvas, ctx: videoCtx };
}

/**
 * Detect objects in image or video without OpenCV
 */
export const detectImage = async (
    imageOrVideo,
    canvas,
    session,
    topk,
    iouThreshold,
    scoreThreshold,
    inputShape
) => {
    const [modelWidth, modelHeight] = inputShape.slice(2);

    let source = imageOrVideo;

    // Handle video frames — reuse module-level canvas instead of allocating each call
    if (imageOrVideo.tagName === "VIDEO") {
        const videoWidth = imageOrVideo.videoWidth;
        const videoHeight = imageOrVideo.videoHeight;

        if (videoWidth === 0 || videoHeight === 0) {
            console.warn("Video not ready yet, skipping frame");
            return [];
        }

        const { canvas: vc, ctx: vCtx } = getVideoCanvas(videoWidth, videoHeight);
        vCtx.drawImage(imageOrVideo, 0, 0, videoWidth, videoHeight);
        source = vc;
    }

    // Preprocess image using reused canvas
    const [inputData, scale, dx, dy] = preprocessingNoCV(source, modelWidth, modelHeight);

    const tensor = new Tensor("float32", inputData, inputShape);
    const config = new Tensor(
        "float32",
        new Float32Array([topk, iouThreshold, scoreThreshold])
    );

    let output0 = null;
    let selected = null;

    try {
        const results = await session.net.run({ images: tensor });
        output0 = results.output0;

        const nmsResults = await session.nms.run({ detection: output0, config });
        selected = nmsResults.selected;

        const boxes = [];
        const stride = selected.dims[2];

        for (let idx = 0; idx < selected.dims[1]; idx++) {
            const offset = idx * stride;
            const data = selected.data;

            const box0 = data[offset];
            const box1 = data[offset + 1];
            const box2 = data[offset + 2];
            const box3 = data[offset + 3];

            // Single-pass argmax over scores instead of Math.max(...) + indexOf
            let score = -Infinity;
            let label = 0;
            for (let c = 4; c < stride; c++) {
                const s = data[offset + c];
                if (s > score) { score = s; label = c - 4; }
            }

            if (score < 0.7) continue;

            const x = (box0 - 0.5 * box2 - dx) / scale;
            const y = (box1 - 0.5 * box3 - dy) / scale;
            const w = box2 / scale;
            const h = box3 / scale;

            boxes.push({ label, probability: score, bounding: [x, y, w, h] });
        }

        return boxes;
    } finally {
        tensor.dispose();
        config.dispose();
        if (output0) output0.dispose();
        if (selected) selected.dispose();
        // No canvas cleanup needed — module-level canvases are reused
    }
};

/**
 * Preprocess image without OpenCV.
 * Uses { willReadFrequently: true } context and skips the trailing clearRect.
 */
export const preprocessingNoCV = (image, modelWidth, modelHeight) => {
    const { ctx } = getPreprocessCanvas(modelWidth, modelHeight);

    const scale = Math.min(modelWidth / image.width, modelHeight / image.height);
    const scaledWidth = Math.round(image.width * scale);
    const scaledHeight = Math.round(image.height * scale);
    const dx = Math.floor((modelWidth - scaledWidth) / 2);
    const dy = Math.floor((modelHeight - scaledHeight) / 2);

    // Fill letterbox with black, then draw scaled image
    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, modelWidth, modelHeight);
    ctx.drawImage(image, dx, dy, scaledWidth, scaledHeight);

    const { data, width, height } = ctx.getImageData(0, 0, modelWidth, modelHeight);
    const tensorData = new Float32Array(3 * width * height);
    const pixelCount = width * height;

    // Unrolled channel-planar copy: R plane, G plane, B plane
    for (let i = 0; i < pixelCount; i++) {
        const src = i * 4;
        tensorData[i]                  = data[src]     / 255.0;
        tensorData[i + pixelCount]     = data[src + 1] / 255.0;
        tensorData[i + 2 * pixelCount] = data[src + 2] / 255.0;
    }

    // No clearRect — next call overwrites with fillRect anyway
    return [tensorData, scale, dx, dy];
};

/**
 * Cleanup ONNX sessions
 */
export const cleanupSessions = async (sessions) => {
    if (sessions.net) sessions.net.release();
    if (sessions.nms) sessions.nms.release();
};