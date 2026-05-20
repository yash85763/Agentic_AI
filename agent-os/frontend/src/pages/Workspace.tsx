import React, { useState, useCallback, useEffect } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import { useDropzone } from 'react-dropzone'
import axios from 'axios'
import clsx from 'clsx'
import AgentFeed from '../components/AgentFeed'

interface UploadedFile {
  file_id: string
  filename: string
  size: number
}

interface Job {
  id: string
  status: string
  task_description: string
  file_ids: string[]
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`
}

export default function Workspace() {
  const navigate = useNavigate()
  const { jobId } = useParams<{ jobId?: string }>()

  const [files, setFiles] = useState<UploadedFile[]>([])
  const [task, setTask] = useState('')
  const [uploading, setUploading] = useState(false)
  const [submitting, setSubmitting] = useState(false)
  const [recentJobs, setRecentJobs] = useState<Job[]>([])
  const [activeJobId, setActiveJobId] = useState<string | null>(jobId ?? null)

  useEffect(() => {
    setActiveJobId(jobId ?? null)
  }, [jobId])

  useEffect(() => {
    axios
      .get('/api/jobs')
      .then(res => setRecentJobs(res.data.slice(0, 10)))
      .catch(console.error)
  }, [activeJobId])

  const onDrop = useCallback(async (accepted: File[]) => {
    if (accepted.length === 0) return
    setUploading(true)
    try {
      const formData = new FormData()
      accepted.forEach(f => formData.append('files', f))
      const res = await axios.post<UploadedFile[]>('/api/files/upload', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      })
      setFiles(prev => [...prev, ...res.data])
    } catch (e) {
      console.error('Upload failed', e)
      alert('Upload failed — check console for details')
    } finally {
      setUploading(false)
    }
  }, [])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': ['.xlsx'],
      'application/vnd.ms-excel': ['.xls'],
      'text/csv': ['.csv'],
      'application/json': ['.json'],
    },
    disabled: uploading,
  })

  const removeFile = (id: string) => setFiles(prev => prev.filter(f => f.file_id !== id))

  const submitJob = async () => {
    if (!task.trim() || files.length === 0) {
      alert('Please enter a task and upload at least one file')
      return
    }
    setSubmitting(true)
    try {
      const res = await axios.post<Job>('/api/jobs', {
        task_description: task,
        file_ids: files.map(f => f.file_id),
      })
      setActiveJobId(res.data.id)
      navigate(`/workspace/${res.data.id}`)
    } catch (e) {
      console.error('Submit failed', e)
      alert('Submit failed — check console')
    } finally {
      setSubmitting(false)
    }
  }

  const startNew = () => {
    setActiveJobId(null)
    setFiles([])
    setTask('')
    navigate('/workspace')
  }

  return (
    <div className="flex h-screen">
      {/* Sidebar: recent jobs */}
      <aside className="w-64 bg-slate-900 border-r border-slate-700/60 overflow-y-auto shrink-0">
        <div className="p-4 border-b border-slate-700/60">
          <button onClick={startNew} className="btn-primary w-full text-sm">
            + New Analysis
          </button>
        </div>
        <div className="p-2">
          <p className="text-xs font-semibold text-slate-500 uppercase tracking-wider px-2 py-2">
            Recent Jobs
          </p>
          <div className="space-y-1">
            {recentJobs.length === 0 ? (
              <p className="text-xs text-slate-500 px-2 py-4">No jobs yet</p>
            ) : (
              recentJobs.map(job => (
                <button
                  key={job.id}
                  onClick={() => {
                    setActiveJobId(job.id)
                    navigate(`/workspace/${job.id}`)
                  }}
                  className={clsx(
                    'w-full text-left px-3 py-2 rounded-lg text-xs transition-colors',
                    activeJobId === job.id
                      ? 'bg-blue-500/20 text-blue-300 border border-blue-500/30'
                      : 'text-slate-300 hover:bg-slate-800'
                  )}
                >
                  <p className="truncate font-medium">{job.task_description}</p>
                  <p className="text-slate-500 text-[10px] mt-0.5 capitalize">{job.status}</p>
                </button>
              ))
            )}
          </div>
        </div>
      </aside>

      {/* Main area */}
      <div className="flex-1 overflow-y-auto">
        {!activeJobId ? (
          <div className="max-w-3xl mx-auto p-8">
            <h1 className="text-2xl font-bold text-slate-100 mb-2">New Analysis</h1>
            <p className="text-sm text-slate-400 mb-8">
              Upload data files and describe what you want the agents to do.
            </p>

            {/* File upload area */}
            <div
              {...getRootProps()}
              className={clsx(
                'border-2 border-dashed rounded-xl p-10 text-center cursor-pointer transition-colors',
                isDragActive
                  ? 'border-blue-500 bg-blue-500/10'
                  : 'border-slate-600 hover:border-slate-500 hover:bg-slate-800/50',
                uploading && 'opacity-50 pointer-events-none'
              )}
            >
              <input {...getInputProps()} />
              <svg
                className="w-10 h-10 mx-auto text-slate-500 mb-3"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={1.5}
                  d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
                />
              </svg>
              {uploading ? (
                <p className="text-slate-400 text-sm">Uploading...</p>
              ) : isDragActive ? (
                <p className="text-blue-300 text-sm">Drop files here</p>
              ) : (
                <>
                  <p className="text-slate-300 text-sm font-medium">
                    Drop Excel/CSV files here, or click to browse
                  </p>
                  <p className="text-slate-500 text-xs mt-1">.xlsx, .xls, .csv, .json</p>
                </>
              )}
            </div>

            {/* Uploaded files list */}
            {files.length > 0 && (
              <div className="mt-4 space-y-2">
                {files.map(f => (
                  <div
                    key={f.file_id}
                    className="flex items-center justify-between bg-slate-800 border border-slate-700 rounded-lg px-3 py-2"
                  >
                    <div className="flex items-center gap-3 min-w-0">
                      <svg className="w-4 h-4 text-blue-400 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
                          d="M9 13h6m-3-3v6m5 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                      </svg>
                      <div className="min-w-0">
                        <p className="text-sm text-slate-200 truncate">{f.filename}</p>
                        <p className="text-xs text-slate-500">{formatBytes(f.size)}</p>
                      </div>
                    </div>
                    <button
                      onClick={() => removeFile(f.file_id)}
                      className="text-slate-500 hover:text-red-400 transition-colors p-1"
                    >
                      <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                          d="M6 18L18 6M6 6l12 12" />
                      </svg>
                    </button>
                  </div>
                ))}
              </div>
            )}

            {/* Task input */}
            <div className="mt-6">
              <label className="block text-sm font-medium text-slate-300 mb-2">
                What do you want the agents to do?
              </label>
              <textarea
                value={task}
                onChange={e => setTask(e.target.value)}
                placeholder="e.g. Consolidate these team expense reports for Q1, validate totals against budgets, and generate an executive summary with charts."
                rows={4}
                className="input resize-none"
              />
            </div>

            <div className="mt-6 flex justify-end">
              <button
                onClick={submitJob}
                disabled={submitting || files.length === 0 || !task.trim()}
                className="btn-primary inline-flex items-center gap-2"
              >
                {submitting ? (
                  <>
                    <span className="spinner"></span> Launching...
                  </>
                ) : (
                  <>
                    Run Analysis
                    <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                        d="M13 9l3 3m0 0l-3 3m3-3H8m13 0a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                  </>
                )}
              </button>
            </div>
          </div>
        ) : (
          <div className="h-full">
            <AgentFeed jobId={activeJobId} />
          </div>
        )}
      </div>
    </div>
  )
}
