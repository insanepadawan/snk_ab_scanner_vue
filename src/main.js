import { createApp } from 'vue'
import { createRouter, createWebHashHistory } from 'vue-router'
import { createI18n } from 'vue-i18n'
import axios from 'axios'
import App from './App.vue'
import ScanView from './views/ScanView.vue'
import ModalView from './views/ModalView.vue'
import CabinetView from './views/CabinetView.vue'
import './style.css'

const i18n = createI18n({
  locale: 'ru',
  legacy: false,
  messages: {
    ru: {
      'Открываем камеру...': 'Открываем камеру...',
      'Не удалось загрузить модель после нескольких попыток.': 'Не удалось загрузить модель после нескольких попыток.'
    },
    en: {
      'Открываем камеру...': 'Opening camera...',
      'Не удалось загрузить модель после нескольких attempts.': 'Failed to load model after several attempts.'
    }
  }
})

const router = createRouter({
  history: createWebHashHistory(),
  routes: [
    { path: '/', redirect: '/scan' },
    { path: '/scan', component: ScanView },
    { path: '/modal/:name', component: ModalView },
    { path: '/cabinet', component: CabinetView }
  ]
})

window.axios = axios
axios.defaults.baseURL = import.meta.env.VITE_API_BASE_URL || ''

const app = createApp(App)
app.use(router)
app.use(i18n)
app.mount('#app')
