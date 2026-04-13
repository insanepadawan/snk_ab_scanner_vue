<template>
  <div class="ab-selector">
    <div class="ab-selector__header">
      <span class="ab-selector__label">MODEL BACKEND</span>
      <span class="ab-selector__badge" :class="current">{{ current || '—' }}</span>
    </div>
    <div class="ab-selector__buttons">
      <button
        class="ab-btn"
        :class="{ active: current === 'onnx' }"
        @click="$emit('select', 'onnx')"
      >
        <span class="ab-btn__tag">A</span>
        <span class="ab-btn__name">ONNX Runtime</span>
        <span class="ab-btn__desc">onnxruntime-web / WASM</span>
      </button>
      <button
        class="ab-btn"
        :class="{ active: current === 'tf' }"
        @click="$emit('select', 'tf')"
      >
        <span class="ab-btn__tag">B</span>
        <span class="ab-btn__name">TensorFlow.js</span>
        <span class="ab-btn__desc">GraphModel / WebGL</span>
      </button>
    </div>
  </div>
</template>

<script setup>
defineProps({ current: String })
defineEmits(['select'])
</script>

<style scoped>
.ab-selector {
  padding: 16px;
  border-bottom: 1px solid var(--border);
}

.ab-selector__header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 10px;
}

.ab-selector__label {
  font-size: 10px;
  letter-spacing: 0.15em;
  color: var(--text-dim);
}

.ab-selector__badge {
  font-size: 10px;
  letter-spacing: 0.1em;
  padding: 2px 8px;
  border: 1px solid var(--border);
  text-transform: uppercase;
}
.ab-selector__badge.onnx { color: var(--accent); border-color: var(--accent); }
.ab-selector__badge.tf   { color: var(--accent2); border-color: var(--accent2); }

.ab-selector__buttons {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 8px;
}

.ab-btn {
  background: var(--surface);
  border: 1px solid var(--border);
  padding: 10px 12px;
  text-align: left;
  display: flex;
  flex-direction: column;
  gap: 2px;
  transition: border-color var(--transition), background var(--transition);
  cursor: pointer;
}

.ab-btn:hover {
  border-color: var(--text-dim);
  background: var(--surface2);
}

.ab-btn.active {
  border-color: var(--accent);
  background: rgba(232, 255, 0, 0.04);
}
.ab-btn.active:last-child {
  border-color: var(--accent2);
  background: rgba(255, 77, 0, 0.04);
}

.ab-btn__tag {
  font-family: var(--font-display);
  font-size: 20px;
  line-height: 1;
  color: var(--text-dim);
  transition: color var(--transition);
}
.ab-btn.active .ab-btn__tag { color: var(--accent); }
.ab-btn.active:last-child .ab-btn__tag { color: var(--accent2); }

.ab-btn__name {
  font-size: 11px;
  font-weight: 700;
  color: var(--text);
}

.ab-btn__desc {
  font-size: 9px;
  color: var(--text-dim);
  letter-spacing: 0.05em;
}
</style>
