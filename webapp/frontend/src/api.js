const j = (r) => r.json()

const okOrThrow = async (r) => {
  const data = await r.json()
  if (!r.ok) throw new Error(data.error || 'Request failed')
  return data
}

export const api = {
  process: (body) =>
    fetch('/api/process', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    }).then(okOrThrow),
  record: (formData) =>
    fetch('/api/record', { method: 'POST', body: formData }).then(okOrThrow),
  status: () => fetch('/api/status').then(j),
  recordings: () => fetch('/api/recordings').then(j),
  recording: (name) => fetch(`/api/recording/${encodeURIComponent(name)}`).then(j),
  events: () => new EventSource('/api/events'),
}

export const media = (name, rel) =>
  `/media/${encodeURIComponent(name)}/${rel}`

export const LANG = { fa: 'Farsi', he: 'Hebrew', iw: 'Hebrew', ar: 'Arabic', en: 'English' }
export const langName = (c) => LANG[c] || c || '?'

export const fmtTs = (s) => {
  s = Number(s || 0)
  const m = Math.floor(s / 60)
  return `${m}:${(s - 60 * m).toFixed(1).padStart(4, '0')}`
}
