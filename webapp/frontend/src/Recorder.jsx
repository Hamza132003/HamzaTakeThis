import React, { useEffect, useRef, useState } from 'react'
import { api } from './api.js'

// Mic recorder: captures RAW audio (browser DSP disabled — the pipeline does
// the enhancement itself), shows a live level meter, and on Stop uploads the
// blob to /api/record where the full-quality pipeline runs.
export default function Recorder({ options, onStarted, disabled }) {
  const [state, setState] = useState('idle') // idle | recording | uploading
  const [elapsed, setElapsed] = useState(0)
  const [level, setLevel] = useState(0)
  const [error, setError] = useState('')

  const mediaRef = useRef(null)      // { stream, recorder, chunks, ctx }
  const rafRef = useRef(0)
  const timerRef = useRef(0)

  const cleanup = () => {
    cancelAnimationFrame(rafRef.current)
    clearInterval(timerRef.current)
    const m = mediaRef.current
    if (m) {
      try { m.stream.getTracks().forEach((t) => t.stop()) } catch {}
      try { m.ctx && m.ctx.close() } catch {}
    }
    mediaRef.current = null
    setLevel(0)
  }

  useEffect(() => cleanup, [])

  const start = async () => {
    setError('')
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          noiseSuppression: false,
          echoCancellation: false,
          autoGainControl: false,
        },
      })
      const mime = ['audio/webm;codecs=opus', 'audio/webm', 'audio/mp4', '']
        .find((m) => !m || MediaRecorder.isTypeSupported(m)) || ''
      const recorder = new MediaRecorder(stream, mime ? { mimeType: mime } : {})
      const chunks = []
      recorder.ondataavailable = (e) => { if (e.data.size) chunks.push(e.data) }
      recorder.start(1000)

      // Level meter
      const ctx = new (window.AudioContext || window.webkitAudioContext)()
      const analyser = ctx.createAnalyser()
      analyser.fftSize = 512
      ctx.createMediaStreamSource(stream).connect(analyser)
      const buf = new Uint8Array(analyser.frequencyBinCount)
      const pump = () => {
        analyser.getByteTimeDomainData(buf)
        let peak = 0
        for (let i = 0; i < buf.length; i++) peak = Math.max(peak, Math.abs(buf[i] - 128))
        setLevel(Math.min(1, peak / 100))
        rafRef.current = requestAnimationFrame(pump)
      }
      pump()

      mediaRef.current = { stream, recorder, chunks, ctx }
      const t0 = Date.now()
      timerRef.current = setInterval(() => setElapsed((Date.now() - t0) / 1000), 200)
      setElapsed(0)
      setState('recording')
    } catch (e) {
      setError(e.name === 'NotAllowedError'
        ? 'Microphone access was denied — allow it in the browser and retry.'
        : `Could not start recording: ${e.message}`)
    }
  }

  const stop = async () => {
    const m = mediaRef.current
    if (!m) return
    setState('uploading')
    const { recorder, chunks } = m
    const done = new Promise((res) => { recorder.onstop = res })
    try { recorder.stop() } catch {}
    await done
    cleanup()

    try {
      const blob = new Blob(chunks, { type: recorder.mimeType || 'audio/webm' })
      if (blob.size < 2000) throw new Error('Recording was empty.')
      const fd = new FormData()
      fd.append('file', blob, 'recording.webm')
      Object.entries(options || {}).forEach(([k, v]) => fd.append(k, v))
      await api.record(fd)
      setState('idle')
      onStarted && onStarted()
    } catch (e) {
      setError(e.message)
      setState('idle')
    }
  }

  const mins = Math.floor(elapsed / 60)
  const secs = (elapsed % 60).toFixed(0).padStart(2, '0')

  return (
    <div>
      {error && <div className="error">{error}</div>}
      <div className="rec-row">
        {state !== 'recording' ? (
          <button className="btn" onClick={start}
            disabled={disabled || state === 'uploading'}>
            {state === 'uploading' ? 'Uploading…' : '● Start recording'}
          </button>
        ) : (
          <button className="btn rec-stop" onClick={stop}>■ Stop &amp; process</button>
        )}
        <div className="rec-meta">
          <span className={`rec-dot ${state === 'recording' ? 'live' : ''}`} />
          <span className="mono rec-time">{mins}:{secs}</span>
        </div>
      </div>
      <div className="meter"><i style={{ width: `${Math.round(level * 100)}%` }} /></div>
      <p className="hint" style={{ marginTop: 10 }}>
        Recording is kept raw (no browser noise suppression) — the AI pipeline
        does the enhancement. When you press Stop, the full-quality pipeline
        runs automatically.
      </p>
    </div>
  )
}
