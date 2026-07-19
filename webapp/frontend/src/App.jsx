import React, { useEffect, useState, useCallback, useRef } from 'react'
import Detail from './Detail.jsx'
import Recorder from './Recorder.jsx'
import { api, media, langName } from './api.js'

function Segmented({ value, onChange, options }) {
  return (
    <div className="seg">
      {options.map(([v, label]) => (
        <button key={v} className={v === value ? 'on' : ''} onClick={() => onChange(v)} type="button">
          <span>{label}</span>
        </button>
      ))}
    </div>
  )
}

function Switch({ on, onChange, label }) {
  return (
    <div className={`switch ${on ? 'on' : ''}`} role="switch" aria-checked={on}
      onClick={() => onChange(!on)}>
      <div className="track"><div className="knob" /></div>
      <span>{label}</span>
    </div>
  )
}

const LANG_OPTIONS = [['auto', 'Auto'], ['fa', 'Farsi'], ['he', 'Hebrew'], ['ar', 'Arabic'], ['en', 'English']]

function InputPanel({ onStarted, disabled }) {
  const [tab, setTab] = useState('folder')
  const [folder, setFolder] = useState('')
  const [language, setLanguage] = useState('auto')
  const [device, setDevice] = useState('auto')
  const [token, setToken] = useState('')
  const [overlap, setOverlap] = useState(true)
  const [error, setError] = useState('')

  const shared = { language, device, hf_token: token, separate_overlap: overlap }

  const submitFolder = async () => {
    setError('')
    try {
      await api.process({ folder, ...shared })
      onStarted()
    } catch (e) {
      setError(e.message)
    }
  }

  return (
    <div className="glass fade-in">
      <div className="tabs">
        <button className={tab === 'folder' ? 'on' : ''} onClick={() => setTab('folder')}>📁 Folder</button>
        <button className={tab === 'mic' ? 'on' : ''} onClick={() => setTab('mic')}>🎙️ Microphone</button>
      </div>

      {tab === 'folder' && (
        <>
          {error && <div className="error">{error}</div>}
          <div className="field">
            <label>Folder containing your recordings</label>
            <input className="txt mono" value={folder} onChange={(e) => setFolder(e.target.value)}
              placeholder="C:\Users\Waleed.Alawneh\Recordings" />
            <span className="hint">Every audio/video file in this folder is processed in one batch.</span>
          </div>
        </>
      )}

      <div className="grid2">
        <div className="field">
          <label>Language (one per recording)</label>
          <Segmented value={language} onChange={setLanguage} options={LANG_OPTIONS} />
          <span className="hint">Auto detects the language; pick one to force it. Arabic = transcribe only.</span>
        </div>
        <div className="field">
          <label>Device</label>
          <Segmented value={device} onChange={setDevice}
            options={[['auto', 'Auto'], ['cuda', 'GPU'], ['cpu', 'CPU']]} />
        </div>
      </div>
      <div className="field">
        <label>HuggingFace token <span className="hint">— enables speaker counting &amp; overlap un-mixing</span></label>
        <input className="txt mono" value={token} onChange={(e) => setToken(e.target.value)}
          placeholder="hf_… (leave blank for single-speaker)" />
      </div>

      {tab === 'folder' ? (
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 16, flexWrap: 'wrap', marginTop: 6 }}>
          <Switch on={overlap} onChange={setOverlap} label="Un-mix overlapping speech" />
          <button className="btn" onClick={submitFolder} disabled={disabled}>
            {disabled ? 'Working…' : 'Process folder'}
          </button>
        </div>
      ) : (
        <>
          <div style={{ marginBottom: 14 }}>
            <Switch on={overlap} onChange={setOverlap} label="Un-mix overlapping speech" />
          </div>
          <Recorder options={shared} onStarted={onStarted} disabled={disabled} />
        </>
      )}
    </div>
  )
}

function Progress({ job }) {
  const stageFrac = job.steps ? (job.step - 1 + (job.substep || 0)) / job.steps : 0
  const frac = job.total ? (job.done + stageFrac) / job.total : 0
  const pct = Math.max(0, Math.min(100, Math.round(frac * 100)))
  return (
    <div className="glass fade-in">
      <div className="prog-head">
        <div>
          <p className="eyebrow" style={{ marginBottom: 6 }}>Working…</p>
          <div className="mono" style={{ fontSize: 14 }}>{job.current || '—'}</div>
          <div style={{ color: 'var(--muted)', fontSize: 13, marginTop: 4 }}>
            {job.stage} · {job.done} / {job.total} files
          </div>
        </div>
        <div className="ring" style={{ '--p': pct }}><i>{pct}%</i></div>
      </div>
      <div className="bar"><i style={{ width: pct + '%' }} /></div>
      {job.log?.length > 0 && <div className="joblog mono">{job.log.join('\n')}</div>}
    </div>
  )
}

function Card({ rec, onOpen }) {
  return (
    <div className="card fade-in" onClick={() => onOpen(rec.name)}>
      <div className="thumb" style={{
        backgroundImage: rec.thumb ? `url(${media(rec.name, rec.thumb)})` : 'none',
      }} />
      <div className="body">
        <h3>{rec.filename}</h3>
        <div className="chips">
          <span className="chip teal">{rec.num_speakers} speaker{rec.num_speakers === 1 ? '' : 's'}</span>
          <span className="chip">{Math.round(rec.duration_sec)}s</span>
          {(rec.languages || []).map((l) => <span key={l} className="chip amber">{langName(l)}</span>)}
          {rec.warnings > 0 && <span className="chip warn">⚠ {rec.warnings}</span>}
        </div>
        {rec.top_tags?.length > 0 && (
          <div className="chips" style={{ marginTop: 8 }}>
            {rec.top_tags.map((t) => <span key={t} className="chip">{t}</span>)}
          </div>
        )}
      </div>
    </div>
  )
}

export default function App() {
  const [view, setView] = useState({ type: 'home' })
  const [recordings, setRecordings] = useState([])
  const [job, setJob] = useState({ running: false })
  const jobRef = useRef(job)
  jobRef.current = job

  const loadRecordings = useCallback(() => {
    api.recordings().then(setRecordings).catch(() => {})
  }, [])

  useEffect(() => { loadRecordings() }, [loadRecordings])

  const applyJob = useCallback((s) => {
    if (jobRef.current.running && !s.running) loadRecordings()
    setJob(s)
  }, [loadRecordings])

  // Progress via SSE; falls back to polling only while a job is running.
  useEffect(() => {
    let es, timer, alive = true
    const startSSE = () => {
      try {
        es = api.events()
        es.onmessage = (e) => { if (alive) applyJob(JSON.parse(e.data)) }
        es.onerror = () => {
          es.close()
          es = null
          // fall back to polling while running, retry SSE occasionally
          const poll = async () => {
            if (!alive) return
            try { applyJob(await api.status()) } catch {}
            if (jobRef.current.running) timer = setTimeout(poll, 2000)
            else timer = setTimeout(startSSE, 5000)
          }
          poll()
        }
      } catch {
        /* EventSource unsupported: one status fetch */
        api.status().then(applyJob).catch(() => {})
      }
    }
    api.status().then(applyJob).catch(() => {})
    startSSE()
    return () => { alive = false; es && es.close(); clearTimeout(timer) }
  }, [applyJob])

  const onStarted = useCallback(() => { api.status().then(applyJob).catch(() => {}) }, [applyJob])

  return (
    <div className="shell">
      <div className="topbar">
        <div className="logo"><span>🎙️</span></div>
        <div className="brand">
          <h1>Voice Isolator</h1>
          <p className="mono">local · offline · no cloud APIs</p>
        </div>
      </div>

      {view.type === 'detail' ? (
        <div className="fade-in" key={'d-' + view.name}>
          <Detail name={view.name} onBack={() => setView({ type: 'home' })} />
        </div>
      ) : (
        <div className="fade-in" key="home">
          <div className="hero" style={{ marginBottom: 24 }}>
            <h2>Isolate the voice.<br /><span className="grad">Understand the noise.</span></h2>
            <p>Separate speech from noise at studio quality, identify what the noise
              is, count the speakers, un-mix overlapping talk, and transcribe —
              word by word — with English and Arabic translations. Entirely on
              your machine.</p>
          </div>

          <InputPanel onStarted={onStarted} disabled={job.running} />

          {job.running && <Progress job={job} />}

          <div className="glass">
            <p className="eyebrow">Results {recordings.length > 0 && `· ${recordings.length}`}</p>
            {recordings.length === 0 ? (
              <p className="hint">No recordings processed yet. Point at a folder or record with the microphone above.</p>
            ) : (
              <div className="gallery">
                {recordings.map((r) => <Card key={r.name} rec={r} onOpen={(name) => setView({ type: 'detail', name })} />)}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
