<template>
  <div class="popup-overlay" @click.self="$emit('close')">
    <div class="popup">
      <button class="popup__close" @click="$emit('close')">✕</button>
      <slot />
    </div>
  </div>
</template>

<script setup>
defineEmits(['close'])
</script>

<style scoped>
.popup-overlay {
  position: fixed;
  inset: 0;
  background: rgba(0,0,0,0.8);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 200;
}

.popup {
  background: var(--surface);
  border: 1px solid var(--border);
  padding: 24px;
  min-width: 280px;
  max-width: 90vw;
  position: relative;
}

.popup__close {
  position: absolute;
  top: 12px;
  right: 12px;
  background: none;
  color: var(--text-dim);
  font-size: 14px;
  cursor: pointer;
  transition: color var(--transition);
}
.popup__close:hover { color: var(--text); }
</style>
