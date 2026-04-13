# YOLO A/B Scanner

Vue 3 + Vite app that runs YOLO object detection with **A/B testing** between two model backends:

| Group | Backend | Model format |
|-------|---------|-------------|
| **A** | onnxruntime-web (WASM) | `last_v6.onnx` + `nms-yolov8.onnx` |
| **B** | TensorFlow.js (GraphModel) | `model.json` + shard `.bin` files |

A/B group is **user-selectable** via the settings panel. Selection persists in `localStorage`. All detection events are logged locally under `ab_log` in `localStorage` — no server reporting.

---

## Project structure

```
yolo-ab-scanner/
├── public/
│   └── model/
│       ├── last_v6.onnx              ← ONNX main model (you provide)
│       ├── nms-yolov8.onnx           ← ONNX NMS model (you provide)
│       ├── ort-wasm-simd-threaded.jsep.wasm  ← ONNX Runtime WASM (you provide)
│       └── tf/
│           ├── model.json            ← TF GraphModel (you provide)
│           └── group1-shard*.bin     ← TF weight shards (you provide)
├── src/
│   ├── components/
│   │   ├── ABSelector.vue   ← A/B group picker UI
│   │   ├── ABLog.vue        ← local event log viewer
│   │   ├── Loader.vue
│   │   ├── Menu.vue
│   │   └── PopUp.vue
│   ├── utils/
│   │   ├── ab.js            ← localStorage A/B helpers
│   │   ├── detect.js        ← ONNX inference + drawing
│   │   ├── detectTF.js      ← TF.js inference + drawing
│   │   ├── download.js      ← fetch with progress
│   │   └── labels.json      ← class index → name map
│   ├── views/
│   │   ├── ScanView.vue     ← main scanner (all logic lives here)
│   │   ├── ModalView.vue
│   │   └── CabinetView.vue
│   ├── App.vue
│   ├── main.js
│   └── style.css
├── index.html
├── vite.config.js
└── package.json
```

---

## Setup

### 1. Install dependencies

```bash
npm install
```

### 2. Add your model files

Copy your model files into `public/model/`:

```
public/model/last_v6.onnx
public/model/nms-yolov8.onnx
public/model/ort-wasm-simd-threaded.jsep.wasm
public/model/tf/model.json
public/model/tf/group1-shard1of1.bin   # (or however many shards you have)
```

> The WASM file can be copied from:
> `node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.jsep.wasm`

### 3. Dev server

```bash
npm run dev
```

The dev server sets the required COOP/COEP headers automatically (needed for WASM multi-threading).

### 4. Production build

```bash
npm run build
npm run preview
```

> ⚠️ Your production server must send these headers for ONNX WASM to work:
> ```
> Cross-Origin-Opener-Policy: same-origin
> Cross-Origin-Embedder-Policy: require-corp
> ```

---

## Adapting detectTF.js

The TF post-processing in `src/utils/detectTF.js` assumes your GraphModel outputs a tensor of shape `[1, num_boxes, 4 + num_classes]` (standard YOLOv8 export).

If your export is different, check what your model actually outputs:

```js
const model = await tf.loadGraphModel('/model/tf/model.json')
console.log('Output names:', model.outputNames)
const dummy = tf.zeros([1, 3, 320, 320])
const out = await model.executeAsync(dummy)
console.log('Output shapes:', Array.isArray(out) ? out.map(t => t.shape) : out.shape)
```

Then adjust step 3 in `detectTF.js` accordingly.

---

## A/B Log

Detection events are stored in `localStorage['ab_log']` as:

```json
[
  { "group": "onnx", "label": "own_goal", "ts": 1712345678901 },
  { "group": "tf",   "label": "offside",   "ts": 1712345679000 }
]
```

View them in the app by clicking the log icon (top right), or read them directly from DevTools → Application → Local Storage.

---

## Environment variables

Create a `.env` file to set your API base URL:

```env
VITE_API_BASE_URL=https://your-api.example.com
```
