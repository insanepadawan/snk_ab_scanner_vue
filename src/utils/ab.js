const AB_GROUP_KEY = 'ab_model_group'
const AB_LOG_KEY = 'ab_log'

/**
 * Read the stored A/B group.
 * @returns {'onnx'|'tf'|null}
 */
export function readAbGroup() {
  return localStorage.getItem(AB_GROUP_KEY) || null
}

/**
 * Persist the chosen A/B group.
 * @param {'onnx'|'tf'} group
 */
export function writeAbGroup(group) {
  localStorage.setItem(AB_GROUP_KEY, group)
}

/**
 * Log a detection event locally for A/B analysis.
 * @param {'onnx'|'tf'} group
 * @param {string} label
 */
export function logAbEvent(group, label) {
  const log = JSON.parse(localStorage.getItem(AB_LOG_KEY) || '[]')
  log.push({ group, label, ts: Date.now() })
  localStorage.setItem(AB_LOG_KEY, JSON.stringify(log))
}

/**
 * Return all locally stored A/B events.
 * @returns {Array<{group: string, label: string, ts: number}>}
 */
export function readAbLog() {
  return JSON.parse(localStorage.getItem(AB_LOG_KEY) || '[]')
}

/**
 * Clear the A/B log.
 */
export function clearAbLog() {
  localStorage.removeItem(AB_LOG_KEY)
}
