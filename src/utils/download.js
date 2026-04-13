/**
 * Download a file from a URL and return its ArrayBuffer.
 * @param {string} url
 * @param {function} [onProgress] - called with fraction 0..1
 * @returns {Promise<ArrayBuffer>}
 */
export async function download(url, onProgress) {
  const response = await fetch(url)
  if (!response.ok) throw new Error(`Download failed: ${response.statusText} (${url})`)

  const contentLength = response.headers.get('content-length')
  const total = contentLength ? parseInt(contentLength, 10) : 0

  const reader = response.body.getReader()
  const chunks = []
  let loaded = 0
  let lastUpdate = 0

  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    chunks.push(value)
    loaded += value.length
    if (total && onProgress && loaded - lastUpdate > total * 0.005) {
      onProgress(loaded / total)
      lastUpdate = loaded
    }
  }

  const totalLength = chunks.reduce((acc, c) => acc + c.length, 0)
  const combined = new Uint8Array(totalLength)
  let offset = 0
  for (const chunk of chunks) {
    combined.set(chunk, offset)
    offset += chunk.length
  }
  return combined.buffer
}
