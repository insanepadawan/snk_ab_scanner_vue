<template>
  <div class="cabinet-page">
    <header class="cabinet-topbar">
      <span class="cabinet-topbar__title">CABINET</span>
      <button class="cabinet-topbar__back" @click="$router.push('/scan')">← SCAN</button>
    </header>

    <div class="cabinet-body">
      <div class="cabinet-card">
        <div class="cabinet-card__label">TYPE</div>
        <div class="cabinet-card__value">{{ type || '—' }}</div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { useRoute } from 'vue-router'
const route = useRoute()
const type = route.query.type
</script>

<style scoped>
.cabinet-page {
  height: 100vh;
  display: flex;
  flex-direction: column;
  background: var(--bg);
}

.cabinet-topbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  height: 44px;
  padding: 0 16px;
  border-bottom: 1px solid var(--border);
}

.cabinet-topbar__title {
  font-family: var(--font-display);
  font-size: 22px;
  letter-spacing: 0.06em;
  color: var(--text);
}

.cabinet-topbar__back {
  font-family: var(--font-mono);
  font-size: 10px;
  letter-spacing: 0.12em;
  background: none;
  border: 1px solid var(--border);
  color: var(--text-dim);
  padding: 4px 10px;
  cursor: pointer;
  transition: all 120ms;
}
.cabinet-topbar__back:hover { color: var(--text); border-color: var(--text-dim); }

.cabinet-body {
  flex: 1;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 40px;
}

.cabinet-card {
  border: 1px solid var(--border);
  padding: 32px 48px;
  text-align: center;
}

.cabinet-card__label {
  font-size: 10px;
  letter-spacing: 0.2em;
  color: var(--text-dim);
  margin-bottom: 8px;
}

.cabinet-card__value {
  font-family: var(--font-display);
  font-size: 48px;
  color: var(--accent);
  letter-spacing: 0.05em;
  text-transform: uppercase;
}
</style>
