import { useState } from 'react'
import type { ChangeEvent, FormEvent } from 'react'
import { createRoot } from 'react-dom/client'
import './style.css'

type Settings = { top_k: number; target_words: number; reading_level: string }
type Result = {
  observation: any
  retrieval: any
  trace: any
  story: any
  consistency: any
  baselines: Record<string, any>
}

const API_URL = import.meta.env.VITE_API_URL ?? 'http://localhost:8000'

function App() {
  const [file, setFile] = useState<File | null>(null)
  const [preview, setPreview] = useState('')
  const [settings, setSettings] = useState<Settings>({ top_k: 5, target_words: 400, reading_level: 'general' })
  const [result, setResult] = useState<Result | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [showTrace, setShowTrace] = useState(false)
  const [showBaselines, setShowBaselines] = useState(false)

  const chooseImage = (event: ChangeEvent<HTMLInputElement>) => {
    const selected = event.target.files?.[0]
    if (!selected) return
    setFile(selected)
    setPreview(URL.createObjectURL(selected))
    setResult(null)
    setError('')
  }

  const generate = async (event: FormEvent) => {
    event.preventDefault()
    if (!file) return setError('Choose an image before generating a story.')
    setLoading(true)
    setError('')
    const form = new FormData()
    form.append('image', file)
    form.append('top_k', String(settings.top_k))
    form.append('target_words', String(settings.target_words))
    form.append('reading_level', settings.reading_level)
    try {
      const response = await fetch(`${API_URL}/api/generate`, { method: 'POST', body: form })
      const payload = await response.json()
      if (!response.ok) throw new Error(payload.detail ?? 'The pipeline could not complete.')
      setResult(payload)
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : 'Unable to reach the Python API.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <main className="shell">
      <header className="topbar">
        <div className="brand-mark" aria-label="Story Studio logo"><svg viewBox="0 0 32 32" aria-hidden="true"><path d="M6 7.5A3.5 3.5 0 0 1 9.5 4H26v20H9.5A3.5 3.5 0 0 0 6 27.5v-20Z" /><path d="M6 7.5A3.5 3.5 0 0 0 9.5 11H26M11 15h10M11 19h7" /></svg></div>
        <div><p className="eyebrow">Evidence-grounded cultural reasoning</p><h1>Story Studio</h1></div>
        <span className="mode-pill">● Mock-ready pipeline</span>
      </header>

      <section className="hero">
        <div><p className="eyebrow warm">FROM IMAGE TO NARRATIVE</p><h2>Stories with context,<br /><em>not assumptions.</em></h2>
          <p className="hero-copy">Upload a scene and explore a respectful Tamil cultural story grounded in retrieved evidence and transparent reasoning.</p>
        </div>
        <div className="hero-stat"><strong>7</strong><span>reasoning stages</span></div>
        <div className="hero-stat"><strong>100%</strong><span>inspectable output</span></div>
      </section>

      <form className="workspace" onSubmit={generate}>
        <aside className="control-panel">
          <div className="section-label"><span>01</span> INPUT IMAGE</div>
          <label className={`dropzone ${preview ? 'has-image' : ''}`}>
            {preview ? <img src={preview} alt="Selected scene" /> : <><span className="upload-icon">↑</span><strong>Drop a scene here</strong><span>or click to browse · JPG, PNG</span></>}
            <input type="file" accept="image/*" onChange={chooseImage} />
          </label>
          {file && <p className="file-name">✓ {file.name}</p>}

          <div className="section-label settings-label"><span>02</span> PIPELINE SETTINGS</div>
          <label className="field-label">Evidence records <output>{settings.top_k}</output><input type="range" min="1" max="10" value={settings.top_k} onChange={e => setSettings({ ...settings, top_k: Number(e.target.value) })} /></label>
          <label className="field-label">Story length <output>{settings.target_words} words</output><input type="range" min="100" max="800" step="50" value={settings.target_words} onChange={e => setSettings({ ...settings, target_words: Number(e.target.value) })} /></label>
          <label className="field-label">Reading level<select value={settings.reading_level} onChange={e => setSettings({ ...settings, reading_level: e.target.value })}><option value="general">General audience</option><option value="simple">Simple</option><option value="academic">Academic</option></select></label>
          <button className="generate-button" disabled={loading || !file}>{loading ? 'Analysing scene…' : 'Generate story  →'}</button>
          {error && <p className="error">{error}</p>}
          <p className="privacy">Your image is processed by the local Python pipeline and is not stored by this frontend.</p>
        </aside>

        <section className="results">
          {!result && !loading && <div className="empty-state"><div className="empty-orbit"><svg viewBox="0 0 32 32" aria-hidden="true"><path d="M6 7.5A3.5 3.5 0 0 1 9.5 4H26v20H9.5A3.5 3.5 0 0 0 6 27.5v-20Z" /><path d="M6 7.5A3.5 3.5 0 0 0 9.5 11H26M11 15h10M11 19h7" /></svg></div><h3>Your story will appear here</h3><p>Begin with a visual clue. The pipeline will separate what is observed from what needs cultural evidence.</p></div>}
          {loading && <div className="empty-state"><div className="loader" /><h3>Reading the scene…</h3><p>Visual understanding, retrieval, reasoning, and verification are running.</p></div>}
          {result && <ResultView result={result} showTrace={showTrace} setShowTrace={setShowTrace} showBaselines={showBaselines} setShowBaselines={setShowBaselines} />}
        </section>
      </form>
    </main>
  )
}

function ResultView({ result, showTrace, setShowTrace, showBaselines, setShowBaselines }: { result: Result; showTrace: boolean; setShowTrace: (v: boolean) => void; showBaselines: boolean; setShowBaselines: (v: boolean) => void }) {
  const { observation: obs, retrieval, story, consistency, trace } = result
  return <div className="result-stack">
    <div className="result-heading"><div><p className="eyebrow warm">ANALYSIS COMPLETE</p><h2>A story rooted in what we can see.</h2></div><span className="verified">✓ Evidence checked</span></div>
    <section className="card observation-card"><div className="card-title">Visual observations <span>01 / 07</span></div><div className="observation-grid"><div><small>SETTING</small><p>{obs.setting || 'Not specified'}</p></div><div><small>ACTIVITIES</small><p>{obs.activities?.join(' · ') || 'No activity confidently identified'}</p></div></div>{obs.cultural_clues?.length > 0 && <div><small>CULTURAL CLUES — UNCONFIRMED</small><div className="chips">{obs.cultural_clues.map((x: string) => <span key={x}>{x}</span>)}</div></div>}</section>
    <section className="card"><div className="card-title">Retrieved cultural evidence <span>{retrieval.retrieved_records?.length ?? 0} records</span></div><p className="query">Query: {retrieval.query_used}</p><div className="evidence-list">{retrieval.retrieved_records?.map((e: any) => <article className="evidence" key={e.record_id}><div><strong>{e.topic}</strong><span className="score">{Math.round(e.relevance_score * 100)}% match</span></div><p>{e.evidence_text}</p><small>{e.source_title} · {e.cultural_region}</small></article>)}</div></section>
    <section className="card story-card"><div className="card-title">Generated Tamil cultural story <span>{story.word_count} words · {story.model_used || 'mock model'}</span></div><p className="story-text">{story.story_text}</p>{story.cultural_elements_included?.length > 0 && <div className="chips">{story.cultural_elements_included.map((x: string) => <span className="gold-chip" key={x}>{x}</span>)}</div>}</section>
    <section className="card warning-card"><div className="card-title">Consistency check <span>{consistency.passed ? 'PASSED' : 'REVIEW NEEDED'}</span></div>{consistency.warnings?.length ? consistency.warnings.map((w: any, index: number) => <p className="warning" key={`${w.message ?? w}-${index}`}>⚠ {typeof w === 'string' ? w : w.message}</p>) : <p className="success">✓ No cultural consistency warnings detected.</p>}</section>
    <div className="toggle-row"><button type="button" onClick={() => setShowTrace(!showTrace)}>{showTrace ? 'Hide' : 'Show'} reasoning trace</button><button type="button" onClick={() => setShowBaselines(!showBaselines)}>{showBaselines ? 'Hide' : 'Compare'} baselines</button></div>
    {showTrace && <section className="card trace-card"><div className="card-title">Research inspector</div><p>{trace.observation_summary}</p><p><strong>Evidence coverage:</strong> {trace.evidence_coverage} · {trace.total_evidence_used}/{trace.total_evidence_retrieved} used</p>{trace.pre_generation_warnings?.map((x: string) => <p className="warning" key={x}>⚠ {x}</p>)}</section>}
    {showBaselines && <section className="card baseline-card"><div className="card-title">Baseline comparison</div>{Object.entries(result.baselines).map(([key, value]) => <div className="baseline" key={key}><small>{key.replace('_', ' ').toUpperCase()}</small><p>{value.story_text}</p></div>)}</section>}
  </div>
}

createRoot(document.getElementById('app')!).render(<App />)
