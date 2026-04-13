<template>
  <div class="loader">
    <div class="loader__inner">
      <div class="loader__bar-track">
        <div class="loader__bar" :style="{ width: progress != null ? progress + '%' : '0%' }" />
      </div>
      <div class="loader__meta">
        <span class="loader__text">{{ text || 'LOADING' }}</span>
        <span class="loader__pct" v-if="progress != null">{{ Math.round(progress) }}%</span>
        <span class="loader__spinner" v-else>
          <span v-for="n in 3" :key="n" class="dot" :style="{ animationDelay: n * 200 + 'ms' }">.</span>
        </span>
      </div>
    </div>
  </div>
</template>

<script setup>
defineProps({
  text: String,
  progress: Number
})
</script>

<style scoped>
.loader {
  position: absolute;
  inset: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  background: var(--bg);
  z-index: 100;
}

.loader__inner {
  width: 260px;
}

.loader__bar-track {
  height: 3px;
  background: var(--border);
  margin-bottom: 12px;
}

.loader__bar {
  height: 100%;
  background: var(--accent);
  transition: width 300ms ease;
}

.loader__meta {
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-family: var(--font-mono);
  font-size: 11px;
  color: var(--text-dim);
  letter-spacing: 0.1em;
  text-transform: uppercase;
}

.loader__pct {
  color: var(--accent);
}

.loader__spinner {
  color: var(--accent);
  letter-spacing: 2px;
}

.dot {
  display: inline-block;
  animation: blink 1s infinite;
  opacity: 0;
}

@keyframes blink {
  0%, 100% { opacity: 0; }
  50% { opacity: 1; }
}
</style>
