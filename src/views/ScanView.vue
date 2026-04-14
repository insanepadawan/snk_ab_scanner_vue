<template>
  <div class="scan-page">

    <!-- ── Top bar ─────────────────────────────────────────────────── -->
    <header class="topbar">
      <span class="topbar__brand">
        <span class="topbar__brand-a">YOLO</span>
        <span class="topbar__brand-b">SCAN</span>
      </span>

      <div class="topbar__center">
        <span class="topbar__group-badge" :class="abGroup || 'none'">
          {{ abGroup ? abGroup.toUpperCase() : 'NO MODEL' }}
        </span>
      </div>

      <div class="topbar__actions">
        <button class="icon-btn" title="A/B Settings" @click="showPanel = !showPanel">
          <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
            <circle cx="8" cy="8" r="2" stroke="currentColor" stroke-width="1.5"/>
            <path d="M8 1v2M8 13v2M1 8h2M13 8h2M3.05 3.05l1.41 1.41M11.54 11.54l1.41 1.41M3.05 12.95l1.41-1.41M11.54 4.46l1.41-1.41" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/>
          </svg>
        </button>
        <button class="icon-btn" title="Log" @click="showLog = !showLog">
          <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
            <rect x="2" y="2" width="12" height="12" rx="1" stroke="currentColor" stroke-width="1.5"/>
            <path d="M5 6h6M5 9h4" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/>
          </svg>
        </button>
      </div>
    </header>

    <!-- ── Side panel: A/B selector + log ────────────────────────── -->
    <transition name="slide">
      <div class="side-panel" v-if="showPanel || showLog">
        <div class="side-panel__inner">
          <ABSelector :current="abGroup" @select="onSelectGroup" />
          <ABLog v-if="showLog" />
        </div>
      </div>
    </transition>

    <!-- ── Main viewport ──────────────────────────────────────────── -->
    <div class="viewport" :class="{ 'panel-open': showPanel || showLog }">

      <!-- Loading -->
      <Loader
        v-if="loading"
        :text="loading.text"
        :progress="loading.progress"
      />

      <!-- Error -->
      <div v-else-if="error" class="error-state">
        <span class="error-state__icon">!</span>
        <p class="error-state__msg">{{ error }}</p>
        <button class="btn-retry" @click="retryLoading">RETRY</button>
      </div>

      <!-- No group chosen yet -->
      <div v-else-if="!abGroup" class="choose-state">
        <p class="choose-state__hint">SELECT A MODEL BACKEND TO BEGIN</p>
        <button class="btn-open-panel" @click="showPanel = true">OPEN SETTINGS →</button>
      </div>

      <!-- Active scanning UI -->
      <template v-else>
        <!-- Video feed -->
        <video
          v-if="useWebcam"
          ref="videoRef"
          class="feed"
          autoplay
          playsinline
          muted
          @play="onVideoPlay"
        />

        <!-- Static image -->
        <img
          v-if="image"
          ref="imageRef"
          class="feed"
          :src="image"
          @load="onImageLoad"
          alt="scan target"
        />

        <!-- Overlay canvas -->
        <canvas ref="canvasRef" class="canvas-overlay" />

        <!-- Detection badge -->
        <transition name="pop">
          <div v-if="detected" class="detection-badge" :class="abGroup">
            <span class="detection-badge__group">{{ abGroup }}</span>
            <span class="detection-badge__label">{{ detected }}</span>
          </div>
        </transition>

        <!-- Corner brackets decoration -->
        <div class="bracket bracket--tl" />
        <div class="bracket bracket--tr" />
        <div class="bracket bracket--bl" />
        <div class="bracket bracket--br" />

        <!-- Bottom controls -->
        <div class="controls">
          <button class="ctrl-btn" @click="openImage" title="Load image">
            <svg width="18" height="18" viewBox="0 0 18 18" fill="none">
              <rect x="1" y="3" width="16" height="12" rx="1" stroke="currentColor" stroke-width="1.5"/>
              <circle cx="6" cy="7.5" r="1.5" stroke="currentColor" stroke-width="1.3"/>
              <path d="M1 12l4-4 3 3 3-3 5 5" stroke="currentColor" stroke-width="1.3" stroke-linejoin="round"/>
            </svg>
          </button>
          <button class="ctrl-btn ctrl-btn--primary" @click="startWebcam" title="Camera">
            <svg width="18" height="18" viewBox="0 0 18 18" fill="none">
              <path d="M1 5h10a1 1 0 011 1v7a1 1 0 01-1 1H1a1 1 0 01-1-1V6a1 1 0 011-1z" stroke="currentColor" stroke-width="1.5"/>
              <path d="M12 8l5-3v8l-5-3V8z" stroke="currentColor" stroke-width="1.5" stroke-linejoin="round"/>
            </svg>
          </button>
          <button class="ctrl-btn" @click="closeMedia" title="Stop">
            <svg width="18" height="18" viewBox="0 0 18 18" fill="none">
              <rect x="4" y="4" width="10" height="10" rx="1" stroke="currentColor" stroke-width="1.5"/>
            </svg>
          </button>
        </div>
      </template>
    </div>

    <!-- hidden file input -->
    <input
      ref="inputImage"
      type="file"
      accept="image/*"
      style="display:none"
      @change="onFileChange"
    />
  </div>
</template>

<script>
import { nextTick, markRaw } from 'vue'
import { Tensor, InferenceSession, env } from 'onnxruntime-web'
import * as tf from '@tensorflow/tfjs'
import { openDB } from 'idb'
import Loader from '../components/Loader.vue'
import ABSelector from '../components/ABSelector.vue'
import ABLog from '../components/ABLog.vue'
import { detectImage, cleanupSessions } from '../utils/detect.js'
import { detectImageTF } from '../utils/detectTF.js'
import { download } from '../utils/download.js'
import { readAbGroup, writeAbGroup, logAbEvent } from '../utils/ab.js'
import labels from '../utils/labels.json'

// ── IndexedDB for WASM binary ─────────────────────────────────────────────────
async function getWasmDb() {
  return await openDB('wasm-cache', 1, {
    upgrade(db) {
      if (!db.objectStoreNames.contains('binaries')) {
        db.createObjectStore('binaries')
      }
    }
  })
}

export default {
  name: 'ScanView',
  components: { Loader, ABSelector, ABLog },

  data() {
    return {
      // ── A/B ──────────────────────────────────────────────────────────────
      abGroup: readAbGroup(),  // 'onnx' | 'tf' | null
      showPanel: false,
      showLog: false,

      // ── ONNX ─────────────────────────────────────────────────────────────
      session: null,

      // ── TF ───────────────────────────────────────────────────────────────
      tfModel: null,

      // ── shared ───────────────────────────────────────────────────────────
      loading: null,
      error: null,
      image: null,
      useWebcam: false,
      videoStream: null,

      // ── model config ──────────────────────────────────────────────────────
      modelName: 'best_v5.onnx',
      tfModelPath: '/model/tf/model.json',
      modelInputShape: [1, 3, 960, 960],
      topk: 100,
      iouThreshold: 0.45,
      scoreThreshold: 0.25,

      // ── detection ─────────────────────────────────────────────────────────
      boxes: [],
      detected: null,
      saving: false,
      rafId: null,
      retryCount: 0,
      maxRetries: 5,
      tensor: null,
      lastSessionId: null,
      sessionRepeatCount: 0,
      maxSessionRepeat: 2,
    }
  },

  computed: {
    modelReady() {
      if (this.abGroup === 'onnx') return !!this.session
      if (this.abGroup === 'tf') return !!this.tfModel
      return false
    }
  },

  async mounted() {
    env.wasm.numThreads = 1
    env.wasm.proxy = false
    console.log('wasmPaths:', env)

    if (this.abGroup) {
      await this.bootForGroup(this.abGroup)
    }
  },

  methods: {
    // ── A/B ──────────────────────────────────────────────────────────────────

    async onSelectGroup(group) {
      if (this.abGroup === group && this.modelReady) {
        this.showPanel = false
        return
      }
      await this.releaseResources()
      this.abGroup = group
      writeAbGroup(group)
      this.showPanel = false
      await this.bootForGroup(group)
    },

    async bootForGroup(group) {
      this.loading = { text: this.$t('Открываем камеру...'), progress: 0 }
      this.error = null
      this.detected = null
      this.boxes = []

      if (group === 'onnx') {
        await this.bootOnnx()
      } else {
        await this.bootTF()
      }
    },

    // ── ONNX boot ─────────────────────────────────────────────────────────────
    async bootOnnx() {
      try {
        const wasmBinary = await this.downloadWasmBinary();
        env.wasm.wasmBinary = wasmBinary;
        await env.wasm.wasmReady;
      } catch (err) {
        console.error('WASM download failed:', err)
        this.error = `Failed to download WASM: ${err.message}`
        this.loading = null
        return
      }

      await nextTick()
      setTimeout(() => this.initializeOnnxApp(), 500)
    },

    async initializeOnnxApp() {
      try {
        this.loading = { text: this.$t('Открываем камеру...'), progress: 0 }
        await this.loadOnnxModels()

        if (this.session) {
          const currentId = this.session.net?.handler?.sessionId
          if (currentId) {
            if (this.lastSessionId === currentId) {
              this.sessionRepeatCount++
              if (this.sessionRepeatCount >= this.maxSessionRepeat) {
                console.warn('Session repeated, forcing hard reload')
                window.location.reload(true)
                return
              }
            } else {
              this.lastSessionId = currentId
              this.sessionRepeatCount = 0
            }
          }
          await this.startWebcam()
        }

        this.loading = null
        this.error = null
      } catch (err) {
        console.error('ONNX init error:', err)
        this.error = `Initialization failed: ${err.message}`
        this.loading = null
      }
    },

    async loadOnnxModels() {
      if (this.session) return

      try {
        const baseModelURL = '/model'
        const ts = Date.now()
        this.loading = { text: this.$t('Открываем камеру...'), progress: 30 }

        let arrBufNet = await this.loadModelFromIndexedDB(this.modelName)
        this.loading.progress = 35

        if (!arrBufNet) {
          this.loading.progress = 40
          arrBufNet = await download(
            `${baseModelURL}/${this.modelName}?ts=${ts}`,
            f => { this.loading.progress = 40 + Math.round(f * 10) }
          )
          this.loading.progress = 50
          await this.saveModelToIndexedDB(this.modelName, arrBufNet)
          console.log('here')
        }

        this.loading.progress = 50
        const yolov26 = await InferenceSession.create(arrBufNet, {
          executionProviders: ['wasm'],
          graphOptimizationLevel: 'basic'
        })
        this.loading.progress = 60

        let arrBufNMS = await this.loadModelFromIndexedDB('nms-yolov8.onnx')
        if (!arrBufNMS) {
          arrBufNMS = await download(`${baseModelURL}/nms-yolov8.onnx?ts=${ts}`)
          await this.saveModelToIndexedDB('nms-yolov8.onnx', arrBufNMS)
        }

        const nms = await InferenceSession.create(arrBufNMS, {
          executionProviders: ['wasm'],
          graphOptimizationLevel: 'basic'
        })

        this.loading.progress = 65

        this.tensor = new Tensor(
          'float32',
          new Float32Array(this.modelInputShape.reduce((a, b) => a * b)),
          this.modelInputShape
        )
        await yolov26.run({ images: this.tensor })
        this.loading.progress = 70

        this.session = { net: yolov26, nms: nms }
        this.loading.progress = 99

        this.loading = null
        this.error = null
        this.retryCount = 0
        arrBufNet = null
        arrBufNMS = null
      } catch (err) {
        console.error('Error loading ONNX models:', err)
        this.error = `Failed to load models: ${err.message}`
        this.loading = null
        await this.retryLoading()
      }
    },

    // ── TF boot ───────────────────────────────────────────────────────────────

    async bootTF() {
      try {
        await nextTick()
        // ✅ Match ONNX's 500ms deferral — lets Vue's scheduler and the router
        //    fully settle before TF.js touches the engine
        await new Promise(resolve => setTimeout(resolve, 500))
        await this.loadTFModel()
        if (this.tfModel) await this.startWebcam()
        this.loading = null
        this.error = null
      } catch (err) {
        console.error('TF boot error:', err)
        this.error = `TF init failed: ${err.message}`
        this.loading = null
      }
    },

    async loadTFModel() {
      if (this.tfModel) return

      try {
        this.loading = { text: this.$t('Открываем камеру...'), progress: 10 }

        const backendsToTry = ['webgl', 'wasm', 'cpu']
        let backendReady = false
        for (const backend of backendsToTry) {
          try {
            await tf.setBackend(backend)
            await tf.ready()
            const probe = tf.scalar(1)
            probe.dataSync()
            probe.dispose()
            console.log(`[TF] Using backend: ${tf.getBackend()}`)
            backendReady = true
            break
          } catch (e) {
            console.warn(`[TF] Backend '${backend}' failed:`, e.message)
          }
        }
        if (!backendReady) throw new Error('No TF.js backend could be initialized')

        this.loading.progress = 20

        const base = this.tfModelPath.replace(/[^/]+$/, '')
        const modelJson = await fetch(this.tfModelPath).then(r => {
          if (!r.ok) throw new Error(`model.json fetch failed: ${r.status}`)
          return r.json()
        })
        this.loading.progress = 30

        const manifest = modelJson.weightsManifest ?? []
        const weightSpecs = manifest.flatMap(g => g.weights)
        const allPaths = manifest.flatMap(g => g.paths)

        const shardBuffers = []
        for (let i = 0; i < allPaths.length; i++) {
          const r = await fetch(base + allPaths[i])
          if (!r.ok) throw new Error(`Weight shard fetch failed: ${allPaths[i]} (${r.status})`)
          shardBuffers.push(await r.arrayBuffer())
          this.loading.progress = 30 + Math.round(((i + 1) / allPaths.length) * 50)
        }

        const totalBytes = shardBuffers.reduce((s, b) => s + b.byteLength, 0)
        const merged = new Uint8Array(totalBytes)
        let offset = 0
        for (const buf of shardBuffers) {
          merged.set(new Uint8Array(buf), offset)
          offset += buf.byteLength
        }

        this.loading.progress = 85

        // ✅ Single ModelArtifacts object — new API
        const model = await tf.loadGraphModel(
            tf.io.fromMemory({
              modelTopology: modelJson.modelTopology,
              weightSpecs,
              weightData:   merged.buffer,
              format:       modelJson.format,
              generatedBy:  modelJson.generatedBy,
              convertedBy:  modelJson.convertedBy,
            })
        )

        this.tfModel = markRaw(model)
        this.loading.progress = 95
        this.tfInputShape = this.tfModel.inputs[0].shape

        // ✅ execute() — no control flow in this graph
        const dummy = tf.zeros(this.tfInputShape)
        const result = this.tfModel.execute(dummy)
        dummy.dispose()
        if (Array.isArray(result)) result.forEach(t => t.dispose())
        else result.dispose()

        this.loading = null
        this.error = null
        this.retryCount = 0
      } catch (err) {
        console.error('Error loading TF model:', err)
        this.error = `Failed to load TF model: ${err.message}`
        this.loading = null
        await this.retryLoading()
      }
    },

    // ── WASM / IndexedDB helpers ──────────────────────────────────────────────

    async downloadWasmBinary() {
      const key = 'ort-wasm-simd-threaded.jsep.wasm'
      const wasmUrl = '/model/ort-wasm-simd-threaded.jsep.wasm'
      const db = await getWasmDb()

      try {
        const cached = await db.get('binaries', key)
        if (cached) {
          console.log('✅ Loaded WASM from cache');
          this.loading.progress = 25
          return cached
        }
      } catch (err) {
        console.warn('IndexedDB read failed, clearing...', err)
        await db.close()
        await indexedDB.deleteDatabase('wasm-cache')
        return await this.downloadWasmBinary()
      }

      const arrayBuffer = await download(
        wasmUrl,
        f => { this.loading.progress = Math.min(25, Math.round(f * 25)) }
      )
      this.loading.progress = 25

      const db2 = await getWasmDb()
      await db2.put('binaries', arrayBuffer, key)
      return arrayBuffer
    },

    openDB() {
      return new Promise((resolve, reject) => {
        const request = indexedDB.open('onnx-models', 1)
        request.onupgradeneeded = event => {
          const db = event.target.result
          if (!db.objectStoreNames.contains('models')) {
            db.createObjectStore('models')
          }
        }
        request.onsuccess = event => {
          const db = event.target.result
          if (!db.objectStoreNames.contains('models')) {
            db.close()
            indexedDB.deleteDatabase('onnx-models')
            return this.openDB().then(resolve).catch(reject)
          }
          resolve(db)
        }
        request.onerror = event => reject(event.target.error)
      })
    },

    async saveModelToIndexedDB(key, buffer) {
      const db = await this.openDB()
      return new Promise((resolve, reject) => {
        const tx = db.transaction('models', 'readwrite')
        tx.objectStore('models').put(buffer, key)
        tx.oncomplete = () => resolve(true)
        tx.onerror = e => reject(e)
      })
    },

    async loadModelFromIndexedDB(key) {
      const db = await this.openDB()
      return new Promise((resolve, reject) => {
        const tx = db.transaction('models', 'readonly')
        const req = tx.objectStore('models').get(key)
        req.onsuccess = () => resolve(req.result || null)
        req.onerror = e => reject(e)
      })
    },

    // ── Detection: image ──────────────────────────────────────────────────────

    async onImageLoad() {
      if (this.abGroup === 'onnx' && this.session) {
        this.boxes = await detectImage(
            this.$refs.imageRef,
            this.$refs.canvasRef,
            this.session,
            this.topk,
            this.iouThreshold,
            this.scoreThreshold,
            this.modelInputShape
        );
      } else if (this.abGroup === 'tf' && this.tfModel) {
        this.boxes = await detectImageTF(
            this.$refs.imageRef, this.$refs.canvasRef,
            this.tfModel, this.topk, this.iouThreshold, this.scoreThreshold,
            this.tfInputShape   // ✅ was: this.modelInputShape
        )
      }

      if (this.boxes.length > 0) {
        this.detected = labels[this.boxes[0].label]
        if (this.detected !== 'old') await this.saveResults(this.detected)
      }
    },

    // ── Detection: webcam loop ────────────────────────────────────────────────

    async onVideoPlay() {
      const isOnnx = this.abGroup === 'onnx'
      if (isOnnx && !this.session) return
      if (!isOnnx && !this.tfModel) return

      const video = this.$refs.videoRef
      const canvas = this.$refs.canvasRef
      const frameInterval = 1000 / 2
      let lastFrameTime = 0
      let predictedCount = {}
      let lockTrigger = false
      const stableThreshold = 6
      const minConfidence = 0.82

      const loop = async timestamp => {
        if (video.paused || video.ended) return

        if (timestamp - lastFrameTime >= frameInterval) {
          lastFrameTime = timestamp

          try {
            if (isOnnx) {
              this.boxes = await detectImage(
                  video,
                  canvas,
                  this.session,
                  this.topk,
                  this.iouThreshold,
                  this.scoreThreshold,
                  this.modelInputShape
              );
              console.log(this.boxes)
            } else {
              this.boxes = await detectImageTF(
                  video, canvas, this.tfModel,
                  this.topk, this.iouThreshold, this.scoreThreshold,
                  this.tfInputShape   // ✅ was: this.modelInputShape
              )
            }
          } catch (e) {
            console.warn('Detection frame error:', e)
          }

          if (this.boxes.length > 0) {
            const top = this.boxes[0]
            const label = labels[top.label]
            const score = top.probability || 0

            if (score >= minConfidence) {
              predictedCount[label] = (predictedCount[label] || 0) + 1
              for (const l in predictedCount) if (l !== label) predictedCount[l] = 0

              if (!lockTrigger && predictedCount[label] >= stableThreshold) {
                this.detected = label
                if (this.detected !== 'old') {
                  if (isOnnx) await cleanupSessions(this.session)
                  await this.saveResults(this.detected)
                  lockTrigger = true
                  return
                }
              }
            } else {
              predictedCount[label] = 0
            }
          } else {
            predictedCount = {}
          }
        }

        this.rafId = requestAnimationFrame(loop)
      }

      this.rafId = requestAnimationFrame(loop)
    },

    // ── Media helpers ─────────────────────────────────────────────────────────

    onFileChange(e) {
      if (!e.target.files[0]) return
      if (this.image) URL.revokeObjectURL(this.image)
      this.image = URL.createObjectURL(e.target.files[0])
      this.useWebcam = false
    },

    openImage() {
      this.$refs.inputImage.click()
    },

    async startWebcam() {
      try {
        this.closeMedia()
        await nextTick()

        const constraints = { video: { facingMode: { ideal: 'environment' } } }
        await new Promise(r => setTimeout(r, 300))
        this.videoStream = await navigator.mediaDevices.getUserMedia(constraints)
        this.useWebcam = true

        await nextTick()
        if (this.$refs.videoRef) {
          this.$refs.videoRef.srcObject = this.videoStream
          await this.$refs.videoRef.play()
        }
      } catch (err) {
        console.error('Webcam error:', err)
        this.error = 'Failed to access camera: ' + err.message
      }
    },

    closeMedia() {
      if (this.rafId) { cancelAnimationFrame(this.rafId); this.rafId = null }
      if (this.image) { URL.revokeObjectURL(this.image); this.image = null }
      if (this.videoStream) {
        this.videoStream.getTracks().forEach(t => t.stop())
        this.videoStream = null
      }
      if (this.$refs.videoRef) this.$refs.videoRef.srcObject = null
      this.useWebcam = false
    },

    async releaseResources() {
      try {
        if (this.session) {
          if (this.session.net) await this.session.net.release()
          if (this.session.nms) await this.session.nms.release()
          this.session = null
        }
        if (this.tfModel) {
          this.tfModel.dispose()
          this.tfModel = null
        }
        this.boxes = []
        this.detected = null
        this.tensor = null
        this.closeMedia()
      } catch (err) {
        console.warn('releaseResources failed:', err)
      }
    },

    async release() {
      if (this.session) {
        try {
          if (this.session.net) await this.session.net.release()
          if (this.session.nms) await this.session.nms.release()
        } catch (e) { console.warn('release failed', e) }
        this.session = null
      }
      if (this.tfModel) {
        try { this.tfModel.dispose() } catch (e) { console.warn('tf dispose failed', e) }
        this.tfModel = null
      }
    },

    async retryLoading() {
      this.closeMedia()
      await this.release()
      if (this.tensor) { this.tensor.data = null; this.tensor = null }
      this.boxes = []
      this.detected = null
      this.retryCount++

      if (this.retryCount < this.maxRetries) {
        setTimeout(() => this.bootForGroup(this.abGroup), 1000)
      } else {
        this.error = this.$t('Не удалось загрузить модель после нескольких попыток.')
      }
    },

    // ── Save / post-scan ──────────────────────────────────────────────────────

    async saveResults(payload) {
      if (this.saving) return
      this.saving = true

      // Local A/B tracking
      logAbEvent(this.abGroup, payload)

      const layer = {
        corner: 'corner',
        foooul: 'foooul',
        goaaaal: 'goaaaal',
        handball: 'handball',
        offside: 'offside',
        own_goal: 'own_goal',
        penalty: 'penalty',
        red_card: 'red_card',
        vaaaar: 'vaaaar'
      }

      if (payload !== 'old') {
        try {
          const data = await axios.post('/scan/store', { bar: layer[payload] })
          const modal = data.data.modal
          const type = data.data.type || 'null'

          await Promise.race([
            this.closeMedia(),
            new Promise(r => setTimeout(r, 2000))
          ])

          await this.deleteFromCaches()
          await this.releaseResources()

          setTimeout(() => {
            window.location.replace(`/cabinet?type=${type}&t=${Date.now()}#/modal/${modal}`)
          }, 400)
        } catch (e) {
          console.error('Scan store failed', e?.response?.data || e)
        } finally {
          this.saving = false
        }
      } else {
        this.saving = false
      }
    },

    async deleteFromCaches() {
      if (!('caches' in window)) return
      const cacheNames = await caches.keys()
      for (const name of cacheNames) {
        const cache = await caches.open(name)
        for (const req of await cache.keys()) {
          if (req.url.endsWith('.wasm')) await cache.delete(req)
        }
      }
    }
  },

  beforeUnmount() {
    this.closeMedia()
  }
}
</script>

<style scoped>
.scan-page {
  display: flex;
  flex-direction: column;
  height: 100vh;
  width: 100vw;
  overflow: hidden;
  background: var(--bg);
}

/* ── Top bar ─────────────────────────────────────────────────────────────── */
.topbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  height: 44px;
  padding: 0 16px;
  border-bottom: 1px solid var(--border);
  flex-shrink: 0;
  z-index: 10;
  background: var(--bg);
}

.topbar__brand {
  font-family: var(--font-display);
  font-size: 22px;
  letter-spacing: 0.06em;
  line-height: 1;
}
.topbar__brand-a { color: var(--accent); }
.topbar__brand-b { color: var(--text); margin-left: 4px; }

.topbar__center {
  position: absolute;
  left: 50%;
  transform: translateX(-50%);
}

.topbar__group-badge {
  font-size: 9px;
  letter-spacing: 0.2em;
  padding: 3px 10px;
  border: 1px solid var(--border);
  color: var(--text-dim);
  text-transform: uppercase;
  transition: all var(--transition);
}
.topbar__group-badge.onnx { color: var(--accent); border-color: var(--accent); }
.topbar__group-badge.tf   { color: var(--accent2); border-color: var(--accent2); }

.topbar__actions {
  display: flex;
  gap: 4px;
}

.icon-btn {
  width: 32px;
  height: 32px;
  background: none;
  border: 1px solid transparent;
  color: var(--text-dim);
  display: flex;
  align-items: center;
  justify-content: center;
  transition: color var(--transition), border-color var(--transition);
}
.icon-btn:hover { color: var(--text); border-color: var(--border); }

/* ── Side panel ─────────────────────────────────────────────────────────── */
.side-panel {
  position: fixed;
  top: 44px;
  right: 0;
  bottom: 0;
  width: 280px;
  background: var(--surface);
  border-left: 1px solid var(--border);
  z-index: 50;
  overflow-y: auto;
}

.slide-enter-active,
.slide-leave-active { transition: transform 220ms cubic-bezier(0.4,0,0.2,1); }
.slide-enter-from,
.slide-leave-to { transform: translateX(100%); }

/* ── Viewport ────────────────────────────────────────────────────────────── */
.viewport {
  flex: 1;
  position: relative;
  overflow: hidden;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: margin-right 220ms cubic-bezier(0.4,0,0.2,1);
}
.viewport.panel-open { margin-right: 280px; }

/* ── Feed ──────────────────────────────────────────────────────────────── */
.feed {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.canvas-overlay {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  pointer-events: none;
}

/* ── Corner brackets ─────────────────────────────────────────────────────── */
.bracket {
  position: absolute;
  width: 28px;
  height: 28px;
  pointer-events: none;
  z-index: 2;
}
.bracket--tl { top: 16px;    left: 16px;   border-top: 2px solid var(--accent); border-left: 2px solid var(--accent); }
.bracket--tr { top: 16px;    right: 16px;  border-top: 2px solid var(--accent); border-right: 2px solid var(--accent); }
.bracket--bl { bottom: 80px; left: 16px;   border-bottom: 2px solid var(--accent); border-left: 2px solid var(--accent); }
.bracket--br { bottom: 80px; right: 16px;  border-bottom: 2px solid var(--accent); border-right: 2px solid var(--accent); }

/* ── Detection badge ──────────────────────────────────────────────────── */
.detection-badge {
  position: absolute;
  top: 16px;
  left: 50%;
  transform: translateX(-50%);
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 6px 16px;
  background: var(--bg);
  border: 1px solid var(--accent);
  z-index: 10;
}
.detection-badge.tf { border-color: var(--accent2); }

.detection-badge__group {
  font-size: 9px;
  letter-spacing: 0.15em;
  color: var(--accent);
  text-transform: uppercase;
}
.detection-badge.tf .detection-badge__group { color: var(--accent2); }

.detection-badge__label {
  font-family: var(--font-display);
  font-size: 20px;
  line-height: 1;
  color: var(--text);
  letter-spacing: 0.08em;
}

.pop-enter-active { transition: all 200ms cubic-bezier(0.34,1.56,0.64,1); }
.pop-leave-active { transition: all 150ms ease; }
.pop-enter-from  { opacity: 0; transform: translateX(-50%) scale(0.8); }
.pop-leave-to    { opacity: 0; transform: translateX(-50%) scale(0.9); }

/* ── Controls ─────────────────────────────────────────────────────────── */
.controls {
  position: absolute;
  bottom: 20px;
  left: 50%;
  transform: translateX(-50%);
  display: flex;
  gap: 10px;
  z-index: 10;
}

.ctrl-btn {
  width: 48px;
  height: 48px;
  background: rgba(10,10,10,0.85);
  border: 1px solid var(--border);
  color: var(--text-dim);
  display: flex;
  align-items: center;
  justify-content: center;
  transition: all var(--transition);
  backdrop-filter: blur(4px);
}
.ctrl-btn:hover { border-color: var(--text-dim); color: var(--text); }
.ctrl-btn--primary { border-color: var(--accent); color: var(--accent); }
.ctrl-btn--primary:hover { background: rgba(232,255,0,0.08); }

/* ── States ────────────────────────────────────────────────────────────── */
.error-state,
.choose-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 16px;
  text-align: center;
  padding: 40px;
}

.error-state__icon {
  font-family: var(--font-display);
  font-size: 64px;
  color: var(--accent2);
  line-height: 1;
}

.error-state__msg {
  font-size: 12px;
  color: var(--text-dim);
  max-width: 260px;
  line-height: 1.6;
}

.choose-state__hint {
  font-size: 11px;
  letter-spacing: 0.15em;
  color: var(--text-dim);
}

.btn-retry,
.btn-open-panel {
  font-family: var(--font-mono);
  font-size: 11px;
  letter-spacing: 0.15em;
  padding: 8px 20px;
  background: none;
  border: 1px solid var(--accent);
  color: var(--accent);
  cursor: pointer;
  transition: background var(--transition);
}
.btn-retry:hover,
.btn-open-panel:hover { background: rgba(232,255,0,0.08); }
</style>
