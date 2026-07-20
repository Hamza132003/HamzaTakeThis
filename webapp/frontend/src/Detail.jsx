import React, { useEffect, useMemo, useRef, useState } from 'react'
import { api, media, langName, fmtTs } from './api.js'

function SpecBlock({ name, rel, waveRel, audioRel, label, audioRef }) {
  if (!rel && !waveRel) return null
  return (
    <div className="spec fade-in">
      <div className="cap"><b>{label}</b></div>
      {waveRel && (
        <img src={media(name, waveRel)} alt={`${label} waveform`} loading="lazy" />
      )}
      {rel && (
        <img src={media(name, rel)} alt={`${label} spectrogram`} loading="lazy"
          style={waveRel ? { marginTop: 8 } : undefined} />
      )}
      {audioRel && (
        <audio controls preload="none" src={media(name, audioRel)} ref={audioRef || undefined} />
      )}
    </div>
  )
}

const sentClass = (s) =>
  s === 'positive' ? 'good' : s === 'negative' ? 'warn' : ''

const RTL = ['fa', 'he', 'ar', 'iw']
const SPK_COLORS = ['#35d2cb', '#f2b45a', '#6ed69a', '#8fa8ff', '#e78a7b', '#d98fff']

// Minimal markdown: **bold**, bullet lines, blank-line paragraphs.
function Mini({ text, rtl }) {
  const blocks = String(text || '').split(/\n/)
  const render = (line, i) => {
    const parts = line.split(/\*\*(.+?)\*\*/g)
    const inner = parts.map((p, j) => (j % 2 ? <b key={j}>{p}</b> : p))
    const t = line.trim()
    if (!t) return null
    if (/^[-•*]\s+/.test(t)) return <li key={i}>{inner}</li>
    return <p key={i}>{inner}</p>
  }
  return <div className={`mini ${rtl ? 'rtl ar-text' : ''}`}>{blocks.map(render)}</div>
}

function SummaryCard({ summary }) {
  const [lang, setLang] = useState('en')
  if (!summary) return null
  const body = lang === 'ar' ? summary.arabic : summary.english
  return (
    <div className="glass summary fade-in">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', flexWrap: 'wrap', gap: 10 }}>
        <p className="eyebrow" style={{ marginBottom: 10 }}>🧠 AI analysis — setting · topic · speakers · activity</p>
        {summary.arabic && (
          <div className="seg mini">
            {[['en', 'English'], ['ar', 'عربي']].map(([v, label]) => (
              <button key={v} className={lang === v ? 'on' : ''}
                onClick={() => setLang(v)} type="button"><span>{label}</span></button>
            ))}
          </div>
        )}
      </div>
      <Mini text={body} rtl={lang === 'ar'} />
      <p className="hint" style={{ marginTop: 10 }}>
        Generated locally by {summary.model}. Inferences, not facts — verify against the audio.
      </p>
    </div>
  )
}

// Merge consecutive same-speaker segments (< 2 s apart) into readable turns.
function mergeTurns(speech) {
  const sorted = [...(speech || [])].sort((a, b) => a.start - b.start)
  const out = []
  for (const s of sorted) {
    const last = out[out.length - 1]
    if (last && last.speaker === s.speaker && last.language === s.language &&
        s.start - last.end < 2.0 && !s.quality?.flagged && !last.flagged) {
      last.end = s.end
      last.text += ' ' + (s.text || '')
      last.english = [last.english, s.english].filter(Boolean).join(' ')
      last.arabic = [last.arabic, s.arabic].filter(Boolean).join(' ')
      last.words = [...(last.words || []), ...(s.words || [])]
      last.parts += 1
    } else {
      out.push({
        speaker: s.speaker, language: s.language, start: s.start, end: s.end,
        text: s.text || '', english: s.english || '', arabic: s.arabic || '',
        words: s.words || [], emotion: s.emotion, sentiment: s.sentiment,
        flagged: !!s.quality?.flagged, reason: s.quality?.reason, parts: 1,
      })
    }
  }
  return out
}

function Words({ words, onSeek, rtl }) {
  if (!words || !words.length) {
    return <p className="hint">No word-level data for this turn.</p>
  }
  return (
    <div className={`words ${rtl ? 'rtl ar-text' : ''}`}>
      {words.map((w, i) => (
        <span key={i}
          className={`word ${w.probability < 0.5 ? 'low' : ''}`}
          style={{ opacity: 0.45 + 0.55 * Math.min(1, w.probability ?? 1) }}
          title={`${fmtTs(w.start)} · confidence ${(w.probability ?? 0).toFixed(2)}`}
          onClick={() => onSeek && onSeek(w.start)}>
          {w.word}
        </span>
      ))}
    </div>
  )
}

function Turn({ t, color, viewMode, onSeek }) {
  const rtl = RTL.includes(t.language)
  return (
    <div className={`turn2 ${t.flagged ? 'flagged' : ''}`} style={{ '--spk': color }}>
      <div className="turn2-head">
        <span className="spk-badge" style={{ background: color }} />
        <b style={{ color }}>{t.speaker}</b>
        <span className="seek" onClick={() => onSeek && onSeek(t.start)}>
          {fmtTs(t.start)}–{fmtTs(t.end)}
        </span>
        <span className="hint">{langName(t.language)}</span>
        <span className={`chip ${sentClass(t.sentiment)}`}>{t.emotion} / {t.sentiment}</span>
        {t.flagged && <span className="chip warn" title={t.reason || ''}>⚠ low confidence</span>}
      </div>
      {(viewMode === 'all' || viewMode === 'original') &&
        <div className={`orig ${rtl ? 'rtl ar-text' : ''}`}>{t.text}</div>}
      {(viewMode === 'all' || viewMode === 'english') && t.english &&
        <div className="en">{t.english}</div>}
      {(viewMode === 'all' || viewMode === 'arabic') && t.language !== 'ar' && t.arabic &&
        <div className="ar">{t.arabic}</div>}
      {viewMode === 'english' && !t.english && <p className="hint">No English for this turn.</p>}
      {viewMode === 'arabic' && t.language === 'ar' &&
        <div className="ar">{t.text}</div>}
      {viewMode === 'words' && <Words words={t.words} onSeek={onSeek} rtl={rtl} />}
    </div>
  )
}

export default function Detail({ name, onBack }) {
  const [r, setR] = useState(null)
  const [err, setErr] = useState('')
  const [viewMode, setViewMode] = useState('all')
  const voiceAudioRef = useRef(null)

  useEffect(() => {
    api.recording(name).then(setR).catch(() => setErr('Could not load report.'))
  }, [name])

  const turns = useMemo(() => mergeTurns(r?.speech), [r])

  if (err) return <div className="glass"><button className="back" onClick={onBack}>← all recordings</button><div className="error">{err}</div></div>
  if (!r) return <div className="glass">Loading…</div>

  const specs = r.spectrograms || {}
  const waves = r.waveforms || {}
  const tracks = r.speaker_tracks || {}
  const hasWords = (r.speech || []).some((s) => (s.words || []).length)
  const speakers = [...new Set(turns.map((t) => t.speaker))]
  const colorOf = (spk) => SPK_COLORS[speakers.indexOf(spk) % SPK_COLORS.length]

  const seek = (t) => {
    const a = voiceAudioRef.current
    if (!a) return
    a.currentTime = Math.max(0, t)
    a.play().catch(() => {})
  }

  return (
    <div className="fade-in">
      <button className="back" onClick={onBack}>← all recordings</button>

      {(r.warnings || []).map((w, i) => (
        <div className="banner" key={i}>⚠ {w}</div>
      ))}

      <div className="glass">
        <h2 style={{ margin: '0 0 14px', fontSize: 22 }}>{r.filename}</h2>
        <div className="meta" style={{ marginBottom: 8 }}>
          <span><b>{Number(r.duration_sec).toFixed(1)}s</b> duration</span>
          <span><b>{r.num_speakers}</b> speaker(s)</span>
          <span>languages: <b>{(r.languages_detected || []).map(langName).join(', ') || '—'}</b></span>
          <span>setting: <b>{r.language_setting}</b></span>
        </div>
        <div className="meta">
          <span>separation: <b>{r.separation_method}</b></span>
          <span>diarization: <b>{r.diarization_method}</b></span>
          {r.diarization_status?.input_branch && (
            <span>diarization input: <b>{r.diarization_status.input_branch}</b>
              {r.diarization_status.input_file
                ? ` (${r.diarization_status.input_file})` : ''}</span>
          )}
          {r.diarization_status && (
            <span>real diarization:{' '}
              <b>{r.diarization_status.genuine_pyannote ? 'yes' : 'NO (fallback)'}</b></span>
          )}
          <span>overlap: <b>{r.overlap_method}</b></span>
          <span>device: <b>{r.device}</b></span>
          {r.sample_rate && <span>rate: <b>{(r.sample_rate / 1000).toFixed(0)} kHz</b></span>}
        </div>
      </div>

      <SummaryCard summary={r.summary} />

      <div className="glass">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', flexWrap: 'wrap', gap: 10 }}>
          <p className="eyebrow" style={{ marginBottom: 10 }}>Transcript</p>
          <div className="seg mini">
            {[['all', 'All'], ['original', 'Original'], ['english', 'English'], ['arabic', 'Arabic'],
              ...(hasWords ? [['words', 'Words']] : [])].map(([v, label]) => (
              <button key={v} className={viewMode === v ? 'on' : ''}
                onClick={() => setViewMode(v)} type="button"><span>{label}</span></button>
            ))}
          </div>
        </div>
        {viewMode === 'words' && (
          <p className="hint" style={{ margin: '4px 0 10px' }}>
            Click a word to play the isolated voice from that moment. Faded /
            dashed words have low confidence.
          </p>
        )}
        {!turns.length && <p className="hint">No intelligible speech transcribed.</p>}
        <div className="feed">
          {turns.map((t, i) => (
            <Turn key={i} t={t} color={colorOf(t.speaker)} viewMode={viewMode} onSeek={seek} />
          ))}
        </div>
      </div>

      <div className="glass">
        <p className="eyebrow">Waveforms, spectrograms &amp; isolated audio</p>
        <SpecBlock name={name} rel={specs.original} waveRel={waves.original}
          audioRel={r.audio?.original || 'audio_48k_mono.wav'}
          label="Original — voice + noise mixed" />
        <SpecBlock name={name} rel={specs.voice} waveRel={waves.voice}
          audioRel={r.audio?.voice} label="Isolated voice" audioRef={voiceAudioRef} />
        <SpecBlock name={name} rel={specs.noise} waveRel={waves.noise}
          audioRel={r.audio?.noise} label="Isolated noise" />
        {(r.speakers || []).map((spk) => (
          <SpecBlock key={spk} name={name} rel={specs[`speaker_${spk}`]}
            waveRel={waves[`speaker_${spk}`]}
            audioRel={tracks[spk]} label={`${spk} — cleaned track`} />
        ))}
        {r.timeline && (
          <div className="spec fade-in">
            <div className="cap"><b>Speaker activity — who spoke when</b></div>
            <img src={media(name, r.timeline)} alt="speaker activity timeline" loading="lazy" />
          </div>
        )}
      </div>

      <div className="glass">
        <p className="eyebrow">Noise — what &amp; when</p>
        {(r.noise?.top_tags || []).length ? (
          <>
            <div className="chips" style={{ marginBottom: 14 }}>
              {r.noise.top_tags.map((t, i) => (
                <span key={i} className="chip amber">{t.label} · {t.confidence.toFixed(2)}</span>
              ))}
            </div>
            {(r.noise.events || []).length > 0 && (
              <div style={{ overflowX: 'auto' }}>
                <table>
                  <thead><tr><th>Time</th><th>Sound</th><th>Confidence</th></tr></thead>
                  <tbody>
                    {r.noise.events.map((e, i) => (
                      <tr key={i}>
                        <td className="num">{fmtTs(e.start)}–{fmtTs(e.end)}</td>
                        <td>{e.label}</td>
                        <td className="num">{e.confidence.toFixed(2)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </>
        ) : <p className="hint">No distinct non-speech sounds detected.</p>}
      </div>

      {(r.overlaps || []).length > 0 && (
        <div className="glass">
          <p className="eyebrow">Overlapping speech (un-mixed)</p>
          <div className="chips">
            {r.overlaps.map((o, i) => (
              <span key={i} className="chip">{fmtTs(o.start)}–{fmtTs(o.end)} · {o.speakers.join(' + ')}</span>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
