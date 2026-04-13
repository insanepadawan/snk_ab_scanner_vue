import { Tensor } from "onnxruntime-web";


let preprocessCanvas = document.createElement("canvas");
let preprocessCtx = preprocessCanvas.getContext("2d");

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
    let offCanvas = null;

    // Handle video frames
    if (imageOrVideo.tagName === "VIDEO") {
        const videoWidth = imageOrVideo.videoWidth;
        const videoHeight = imageOrVideo.videoHeight;

        if (videoWidth === 0 || videoHeight === 0) {
            console.warn("Video not ready yet, skipping frame");
            return [];
        }

        offCanvas = document.createElement("canvas");
        offCanvas.width = videoWidth;
        offCanvas.height = videoHeight;
        const ctx = offCanvas.getContext("2d");
        ctx.drawImage(imageOrVideo, 0, 0, videoWidth, videoHeight);

        source = offCanvas;
    }

    // Preprocess image using canvas
    const [inputData, scale, dx, dy] = preprocessingNoCV(source, modelWidth, modelHeight);

    const tensor = new Tensor("float32", inputData, inputShape);
    const config = new Tensor(
        "float32",
        new Float32Array([topk, iouThreshold, scoreThreshold])
    );

    let output0 = null;
    let selected = null;

    try {
        // Run inference
        const results = await session.net.run({ images: tensor });
        output0 = results.output0;

        const nmsResults = await session.nms.run({ detection: output0, config });
        selected = nmsResults.selected;

        const boxes = [];
        for (let idx = 0; idx < selected.dims[1]; idx++) {
            const data = selected.data.slice(idx * selected.dims[2], (idx + 1) * selected.dims[2]);
            const box = data.slice(0, 4);
            const scores = data.slice(4);
            const score = Math.max(...scores);
            const label = scores.indexOf(score);

            const [x, y, w, h] = [
                (box[0] - 0.5 * box[2] - dx) / scale,
                (box[1] - 0.5 * box[3] - dy) / scale,
                box[2] / scale,
                box[3] / scale,
            ];

            if (score >= 0.7) {
                boxes.push({ label, probability: score, bounding: [x, y, w, h] });
            }
        }

        return boxes;
    } finally {
        tensor.dispose();
        config.dispose();
        if (output0) output0.dispose();
        if (selected) selected.dispose();

        if (offCanvas) {
            offCanvas.width = 0;
            offCanvas.height = 0;
        }
    }
};

/**
 * Preprocess image without OpenCV
 */
export const preprocessingNoCV = (image, modelWidth, modelHeight) => {
    preprocessCanvas.width = modelWidth;
    preprocessCanvas.height = modelHeight;

    const scale = Math.min(modelWidth / image.width, modelHeight / image.height);
    const scaledWidth = Math.round(image.width * scale);
    const scaledHeight = Math.round(image.height * scale);
    const dx = Math.floor((modelWidth - scaledWidth) / 2);
    const dy = Math.floor((modelHeight - scaledHeight) / 2);

    preprocessCtx.fillStyle = "black";
    preprocessCtx.fillRect(0, 0, modelWidth, modelHeight);
    preprocessCtx.drawImage(image, dx, dy, scaledWidth, scaledHeight);

    const { data, width, height } = preprocessCtx.getImageData(0, 0, modelWidth, modelHeight);
    const tensorData = new Float32Array(3 * width * height);

    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const idx = (y * width + x) * 4;
            const base = y * width + x;
            tensorData[base] = data[idx] / 255.0;
            tensorData[base + width * height] = data[idx + 1] / 255.0;
            tensorData[base + 2 * width * height] = data[idx + 2] / 255.0;
        }
    }

    preprocessCtx.clearRect(0, 0, modelWidth, modelHeight);
    return [tensorData, scale, dx, dy];
};

/**
 * Cleanup ONNX sessions
 */
export const cleanupSessions = async (sessions) => {
    if (sessions.net) sessions.net.release();
    if (sessions.nms) sessions.nms.release();
};
