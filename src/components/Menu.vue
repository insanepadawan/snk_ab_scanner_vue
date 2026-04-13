<template>
  <nav class="menu">
    <slot />
  </nav>
</template>

<style scoped>
.menu {
  display: flex;
  gap: 8px;
  padding: 0 16px;
  border-bottom: 1px solid var(--border);
  height: 40px;
  align-items: center;
}
</style>
