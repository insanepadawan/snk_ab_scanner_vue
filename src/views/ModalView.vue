<template>
  <div class="modal-page">
    <div class="modal-box">
      <div class="modal-box__tag">RESULT</div>
      <h1 class="modal-box__name">{{ name }}</h1>
      <p class="modal-box__sub">Detection confirmed</p>
      <button class="modal-box__btn" @click="$router.push('/scan')">↩ SCAN AGAIN</button>
    </div>
  </div>
</template>

<script setup>
import { useRoute } from 'vue-router'
const route = useRoute()
const name = route.params.name
</script>

<style scoped>
.modal-page {
  height: 100vh;
  display: flex;
  align-items: center;
  justify-content: center;
  background: var(--bg);
}

.modal-box {
  border: 1px solid var(--border);
  padding: 40px 48px;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 12px;
  text-align: center;
}

.modal-box__tag {
  font-size: 10px;
  letter-spacing: 0.2em;
  color: var(--text-dim);
}

.modal-box__name {
  font-family: var(--font-display);
  font-size: 52px;
  color: var(--accent);
  line-height: 1;
  letter-spacing: 0.05em;
  text-transform: uppercase;
}

.modal-box__sub {
  font-size: 11px;
  color: var(--text-dim);
  letter-spacing: 0.1em;
}

.modal-box__btn {
  margin-top: 16px;
  font-family: var(--font-mono);
  font-size: 11px;
  letter-spacing: 0.15em;
  padding: 8px 20px;
  background: none;
  border: 1px solid var(--border);
  color: var(--text-dim);
  cursor: pointer;
  transition: all 120ms;
}
.modal-box__btn:hover {
  border-color: var(--text);
  color: var(--text);
}
</style>
