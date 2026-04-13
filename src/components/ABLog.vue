<template>
  <div class="ab-log">
    <div class="ab-log__header">
      <span class="ab-log__title">A/B LOG</span>
      <button class="ab-log__clear" @click="clear">CLEAR</button>
    </div>

    <div v-if="!entries.length" class="ab-log__empty">No events yet</div>

    <div v-else class="ab-log__stats">
      <div class="stat" v-for="(s, group) in stats" :key="group" :class="group">
        <span class="stat__group">{{ group.toUpperCase() }}</span>
        <span class="stat__count">{{ s.total }} detections</span>
        <div class="stat__labels">
          <span v-for="(count, label) in s.labels" :key="label" class="stat__pill">
            {{ label }}&nbsp;<strong>{{ count }}</strong>
          </span>
        </div>
      </div>
    </div>

    <div class="ab-log__rows">
      <div class="ab-log__row" v-for="(e, i) in entriesDesc" :key="i" :class="e.group">
        <span class="ab-log__row-group">{{ e.group }}</span>
        <span class="ab-log__row-label">{{ e.label }}</span>
        <span class="ab-log__row-ts">{{ fmt(e.ts) }}</span>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import { readAbLog, clearAbLog } from '../utils/ab.js'

const entries = ref([])

onMounted(() => { entries.value = readAbLog() })

function clear() {
  clearAbLog()
  entries.value = []
}

const entriesDesc = computed(() => [...entries.value].reverse().slice(0, 50))

const stats = computed(() => {
  const s = {}
  for (const e of entries.value) {
    if (!s[e.group]) s[e.group] = { total: 0, labels: {} }
    s[e.group].total++
    s[e.group].labels[e.label] = (s[e.group].labels[e.label] || 0) + 1
  }
  return s
})

function fmt(ts) {
  return new Date(ts).toLocaleTimeString('en', { hour12: false })
}
</script>

<style scoped>
.ab-log { padding: 16px; font-size: 11px; }

.ab-log__header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 12px;
}

.ab-log__title {
  font-size: 10px;
  letter-spacing: 0.15em;
  color: var(--text-dim);
}

.ab-log__clear {
  font-size: 9px;
  font-family: var(--font-mono);
  letter-spacing: 0.1em;
  color: var(--text-dim);
  background: none;
  border: 1px solid var(--border);
  padding: 2px 6px;
  cursor: pointer;
  transition: color var(--transition), border-color var(--transition);
}
.ab-log__clear:hover { color: var(--accent2); border-color: var(--accent2); }

.ab-log__empty { color: var(--text-muted); text-align: center; padding: 20px 0; }

.ab-log__stats { display: flex; flex-direction: column; gap: 8px; margin-bottom: 16px; }

.stat {
  border: 1px solid var(--border);
  padding: 8px 10px;
}
.stat.onnx { border-left: 3px solid var(--accent); }
.stat.tf   { border-left: 3px solid var(--accent2); }

.stat__group { display: block; font-size: 9px; letter-spacing: 0.12em; color: var(--text-dim); margin-bottom: 2px; }
.stat__count { display: block; font-weight: 700; margin-bottom: 6px; }
.stat__labels { display: flex; flex-wrap: wrap; gap: 4px; }

.stat__pill {
  background: var(--surface2);
  border: 1px solid var(--border);
  padding: 1px 6px;
  font-size: 9px;
  letter-spacing: 0.04em;
}

.ab-log__rows { max-height: 200px; overflow-y: auto; display: flex; flex-direction: column; gap: 1px; }

.ab-log__row {
  display: grid;
  grid-template-columns: 36px 1fr auto;
  gap: 8px;
  padding: 4px 6px;
  background: var(--surface);
  font-size: 10px;
  align-items: center;
}

.ab-log__row-group {
  font-size: 9px;
  letter-spacing: 0.08em;
  font-weight: 700;
}
.ab-log__row.onnx .ab-log__row-group { color: var(--accent); }
.ab-log__row.tf   .ab-log__row-group { color: var(--accent2); }

.ab-log__row-ts { color: var(--text-muted); font-size: 9px; }
</style>
