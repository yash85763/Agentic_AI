import React, { useEffect, useState } from 'react'
import axios from 'axios'
import clsx from 'clsx'
import MonacoEditor from '../components/MonacoEditor'
import FileExplorer from '../components/FileExplorer'

interface CognitiveTree {
  [key: string]: CognitiveTree | string
}

function languageOf(path: string): string {
  if (path.endsWith('.md')) return 'markdown'
  if (path.endsWith('.json')) return 'json'
  if (path.endsWith('.yaml') || path.endsWith('.yml')) return 'yaml'
  if (path.endsWith('.py')) return 'python'
  return 'plaintext'
}

export default function Studio() {
  const [tree, setTree] = useState<CognitiveTree | null>(null)
  const [selectedPath, setSelectedPath] = useState<string | null>(null)
  const [content, setContent] = useState('')
  const [dirty, setDirty] = useState(false)
  const [saving, setSaving] = useState(false)
  const [previewTask, setPreviewTask] = useState('expense consolidation')
  const [preview, setPreview] = useState<{ system_prompt: string; token_estimate: number } | null>(null)
  const [previewOpen, setPreviewOpen] = useState(false)

  useEffect(() => {
    loadTree()
  }, [])

  const loadTree = async () => {
    try {
      const res = await axios.get('/api/cognitive')
      setTree(res.data)
    } catch (e) {
      console.error(e)
    }
  }

  const loadFile = async (path: string) => {
    if (dirty && !confirm('Discard unsaved changes?')) return
    try {
      const res = await axios.get(`/api/cognitive/${path}`)
      setSelectedPath(path)
      setContent(res.data.content)
      setDirty(false)
    } catch (e) {
      console.error(e)
    }
  }

  const save = async () => {
    if (!selectedPath) return
    setSaving(true)
    try {
      await axios.put(`/api/cognitive/${selectedPath}`, { content })
      setDirty(false)
    } catch (e) {
      console.error(e)
      alert('Save failed')
    } finally {
      setSaving(false)
    }
  }

  const createFile = async () => {
    const path = prompt('Path (e.g. skills/my-new-skill.md):')
    if (!path) return
    try {
      await axios.post(`/api/cognitive/${path}`, { content: '# New file\n' })
      await loadTree()
      await loadFile(path)
    } catch (e: any) {
      alert(e.response?.data?.detail ?? 'Create failed')
    }
  }

  const deleteFile = async () => {
    if (!selectedPath) return
    if (!confirm(`Delete ${selectedPath}?`)) return
    try {
      await axios.delete(`/api/cognitive/${selectedPath}`)
      setSelectedPath(null)
      setContent('')
      setDirty(false)
      await loadTree()
    } catch (e) {
      console.error(e)
    }
  }

  const showPreview = async () => {
    try {
      const res = await axios.get('/api/cognitive-preview', {
        params: { task: previewTask },
      })
      setPreview(res.data)
      setPreviewOpen(true)
    } catch (e) {
      console.error(e)
    }
  }

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key === 's') {
        e.preventDefault()
        if (selectedPath && dirty) save()
      }
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [selectedPath, dirty, content])

  return (
    <div className="flex h-screen">
      {/* File explorer */}
      <aside className="w-72 bg-slate-900 border-r border-slate-700/60 overflow-y-auto shrink-0">
        <div className="p-4 border-b border-slate-700/60 flex items-center justify-between">
          <h2 className="text-sm font-semibold text-slate-200">Cognitive Filesystem</h2>
          <button
            onClick={createFile}
            className="text-slate-400 hover:text-slate-100 p-1"
            title="New file"
          >
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                d="M12 9v3m0 0v3m0-3h3m-3 0H9m12 0a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          </button>
        </div>
        {tree ? (
          <FileExplorer
            tree={tree}
            selected={selectedPath}
            onSelect={loadFile}
          />
        ) : (
          <p className="p-4 text-xs text-slate-500">Loading...</p>
        )}
      </aside>

      {/* Editor */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Toolbar */}
        <div className="px-4 py-2 border-b border-slate-700/60 bg-slate-900 flex items-center justify-between">
          <div className="flex items-center gap-2 text-sm min-w-0">
            {selectedPath ? (
              <>
                <svg className="w-4 h-4 text-blue-400 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
                    d="M9 13h6m-3-3v6m5 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                </svg>
                <span className="text-slate-200 truncate">{selectedPath}</span>
                {dirty && <span className="text-amber-400 text-xs">● unsaved</span>}
              </>
            ) : (
              <span className="text-slate-500">Select a file to edit</span>
            )}
          </div>
          <div className="flex items-center gap-2">
            <button onClick={showPreview} className="btn-ghost text-xs">
              Preview Context
            </button>
            {selectedPath && (
              <>
                <button
                  onClick={deleteFile}
                  className="btn-ghost text-xs text-red-400 hover:text-red-300"
                >
                  Delete
                </button>
                <button
                  onClick={save}
                  disabled={!dirty || saving}
                  className="btn-primary text-xs"
                >
                  {saving ? 'Saving...' : 'Save (⌘S)'}
                </button>
              </>
            )}
          </div>
        </div>

        {/* Editor area */}
        <div className="flex-1 min-h-0">
          {selectedPath ? (
            <MonacoEditor
              value={content}
              language={languageOf(selectedPath)}
              onChange={v => {
                setContent(v ?? '')
                setDirty(true)
              }}
            />
          ) : (
            <div className="h-full flex items-center justify-center text-slate-500 text-sm">
              <div className="text-center">
                <svg className="w-12 h-12 mx-auto mb-3 text-slate-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1}
                    d="M9 13h6m-3-3v6m5 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                </svg>
                <p>Select a file from the explorer</p>
                <p className="text-xs mt-1 text-slate-600">
                  Edit soul, skills, knowledge, and memory files — changes load at the next agent run
                </p>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Preview Drawer */}
      {previewOpen && preview && (
        <div className="fixed inset-0 bg-black/60 z-50 flex items-center justify-center p-8" onClick={() => setPreviewOpen(false)}>
          <div
            className="bg-slate-900 border border-slate-700 rounded-xl shadow-2xl w-full max-w-4xl max-h-[80vh] flex flex-col"
            onClick={e => e.stopPropagation()}
          >
            <div className="px-6 py-4 border-b border-slate-700 flex items-center justify-between">
              <div>
                <h3 className="text-base font-semibold text-slate-100">Assembled System Prompt</h3>
                <p className="text-xs text-slate-500 mt-1">
                  Task: "{previewTask}" · ~{Math.round(preview.token_estimate)} tokens
                </p>
              </div>
              <div className="flex items-center gap-2">
                <input
                  className="input text-xs w-64"
                  value={previewTask}
                  onChange={e => setPreviewTask(e.target.value)}
                  placeholder="Test task description..."
                />
                <button onClick={showPreview} className="btn-secondary text-xs">
                  Refresh
                </button>
                <button onClick={() => setPreviewOpen(false)} className="btn-ghost text-xs">
                  ✕
                </button>
              </div>
            </div>
            <div className="flex-1 overflow-auto p-6">
              <pre className="text-xs text-slate-300 whitespace-pre-wrap font-mono">
                {preview.system_prompt}
              </pre>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
