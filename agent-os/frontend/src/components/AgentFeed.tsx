import React, { useState, useRef, useEffect } from 'react'
import ReactMarkdown from 'react-markdown'
import clsx from 'clsx'
import EChartRenderer from './EChartRenderer'
import type { AgentEvent } from '../hooks/useJobStream'

interface AgentFeedProps {
  events: AgentEvent[]
  isConnected: boolean
  error: string | null
  jobId: string | null
  onRetry?: () => void
}

// --- Individual Event Cards ---

function ThinkingCard({ data }: { data: { text?: string; streaming?: boolean } }) {
  const [collapsed, setCollapsed] = useState(false)
  return (
    <div className="bg-slate-800/60 border border-slate-700/60 rounded-xl overflow-hidden">
      <button
        onClick={() => setCollapsed(v => !v)}
        className="w-full flex items-center gap-2.5 px-4 py-3 text-left hover:bg-slate-700/30 transition-colors"
      >
        <div className="w-2 h-2 rounded-full bg-purple-400 shrink-0 animate-pulse" />
        <span className="text-sm font-medium text-purple-300">Thinking</span>
        {data.streaming && (
          <span className="ml-auto text-xs text-purple-400/60 italic">streaming...</span>
        )}
        <svg
          className={clsx('w-4 h-4 text-slate-500 ml-auto transition-transform shrink-0', collapsed && 'rotate-180')}
          fill="none" viewBox="0 0 24 24" stroke="currentColor"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
        </svg>
      </button>
      {!collapsed && (
        <div className="px-4 pb-4 text-sm text-slate-400 leading-relaxed whitespace-pre-wrap border-t border-slate-700/40 pt-3 font-mono text-xs">
          {data.text ?? ''}
        </div>
      )}
    </div>
  )
}

function CodeGeneratedCard({ data }: { data: { language?: string; code?: string; filename?: string } }) {
  const [copied, setCopied] = useState(false)
  const handleCopy = () => {
    navigator.clipboard.writeText(data.code ?? '')
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  return (
    <div className="bg-slate-900 border border-slate-700/60 rounded-xl overflow-hidden">
      <div className="flex items-center justify-between px-4 py-2.5 bg-slate-800/60 border-b border-slate-700/40">
        <div className="flex items-center gap-2">
          <svg className="w-4 h-4 text-green-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4" />
          </svg>
          <span className="text-sm font-medium text-green-300">Code Generated</span>
          {data.filename && (
            <span className="text-xs text-slate-400 font-mono bg-slate-700 px-1.5 py-0.5 rounded">{data.filename}</span>
          )}
        </div>
        <div className="flex items-center gap-2">
          {data.language && (
            <span className="text-xs text-slate-500 uppercase">{data.language}</span>
          )}
          <button
            onClick={handleCopy}
            className="text-xs text-slate-400 hover:text-slate-200 px-2 py-1 rounded bg-slate-700 hover:bg-slate-600 transition-colors"
          >
            {copied ? 'Copied!' : 'Copy'}
          </button>
        </div>
      </div>
      <pre className="overflow-x-auto p-4 text-xs leading-relaxed text-slate-300 font-mono">
        <code>{data.code ?? ''}</code>
      </pre>
    </div>
  )
}

function CodeExecutingCard({ data }: { data: { message?: string; sandboxId?: string } }) {
  return (
    <div className="bg-slate-800/40 border border-yellow-700/30 rounded-xl px-4 py-3 flex items-center gap-3">
      <div className="shrink-0">
        <svg className="w-5 h-5 text-yellow-400 animate-spin" fill="none" viewBox="0 0 24 24">
          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
        </svg>
      </div>
      <div>
        <p className="text-sm font-medium text-yellow-300">
          {data.message ?? 'Running in Docker sandbox...'}
        </p>
        {data.sandboxId && (
          <p className="text-xs text-slate-500 font-mono mt-0.5">sandbox: {data.sandboxId}</p>
        )}
      </div>
    </div>
  )
}

function CodeResultCard({ data }: { data: { stdout?: string; stderr?: string; returnValue?: unknown; executionTime?: number; success?: boolean } }) {
  const [collapsed, setCollapsed] = useState(true)
  const success = data.success !== false

  return (
    <div className={clsx(
      'border rounded-xl overflow-hidden',
      success ? 'bg-slate-800/40 border-green-700/30' : 'bg-slate-800/40 border-red-700/30'
    )}>
      <button
        onClick={() => setCollapsed(v => !v)}
        className="w-full flex items-center gap-2.5 px-4 py-3 text-left hover:bg-slate-700/20 transition-colors"
      >
        <svg
          className={clsx('w-4 h-4 shrink-0', success ? 'text-green-400' : 'text-red-400')}
          fill="none" viewBox="0 0 24 24" stroke="currentColor"
        >
          {success ? (
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
          ) : (
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 14l2-2m0 0l2-2m-2 2l-2-2m2 2l2 2m7-2a9 9 0 11-18 0 9 9 0 0118 0z" />
          )}
        </svg>
        <span className={clsx('text-sm font-medium', success ? 'text-green-300' : 'text-red-300')}>
          {success ? 'Execution Succeeded' : 'Execution Failed'}
        </span>
        {data.executionTime !== undefined && (
          <span className="text-xs text-slate-500 ml-auto">{data.executionTime.toFixed(2)}s</span>
        )}
        <svg
          className={clsx('w-4 h-4 text-slate-500 transition-transform shrink-0', !collapsed && 'rotate-180')}
          fill="none" viewBox="0 0 24 24" stroke="currentColor"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
        </svg>
      </button>
      {!collapsed && (
        <div className="border-t border-slate-700/40 p-4 space-y-3">
          {data.stdout && (
            <div>
              <p className="text-xs font-semibold text-slate-500 mb-1 uppercase tracking-wider">stdout</p>
              <pre className="text-xs text-slate-300 font-mono whitespace-pre-wrap overflow-x-auto bg-slate-900 rounded p-2">{data.stdout}</pre>
            </div>
          )}
          {data.stderr && (
            <div>
              <p className="text-xs font-semibold text-red-500 mb-1 uppercase tracking-wider">stderr</p>
              <pre className="text-xs text-red-300 font-mono whitespace-pre-wrap overflow-x-auto bg-slate-900 rounded p-2">{data.stderr}</pre>
            </div>
          )}
          {data.returnValue !== undefined && (
            <div>
              <p className="text-xs font-semibold text-slate-500 mb-1 uppercase tracking-wider">return value</p>
              <pre className="text-xs text-slate-300 font-mono overflow-x-auto bg-slate-900 rounded p-2">
                {JSON.stringify(data.returnValue, null, 2)}
              </pre>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

function ChartReadyCard({ data }: { data: { option?: Record<string, unknown>; title?: string; chartId?: string } }) {
  if (!data.option) return null
  return (
    <div className="bg-slate-800/40 border border-blue-700/30 rounded-xl overflow-hidden">
      <div className="flex items-center gap-2 px-4 py-2.5 border-b border-slate-700/40">
        <svg className="w-4 h-4 text-blue-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
        </svg>
        <span className="text-sm font-medium text-blue-300">Chart Ready</span>
        {data.chartId && <span className="text-xs text-slate-500 font-mono ml-auto">{data.chartId}</span>}
      </div>
      <div className="p-4">
        <EChartRenderer
          option={data.option}
          height={280}
          title={data.title}
        />
      </div>
    </div>
  )
}

function ValidationCard({ data }: { data: { checks?: Array<{ label: string; passed: boolean; message?: string }> } }) {
  const checks = data.checks ?? []
  const passCount = checks.filter(c => c.passed).length

  return (
    <div className="bg-slate-800/40 border border-slate-700/60 rounded-xl overflow-hidden">
      <div className="flex items-center justify-between px-4 py-2.5 border-b border-slate-700/40">
        <div className="flex items-center gap-2">
          <svg className="w-4 h-4 text-cyan-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-6 9l2 2 4-4" />
          </svg>
          <span className="text-sm font-medium text-cyan-300">Validation</span>
        </div>
        <span className="text-xs text-slate-400">{passCount}/{checks.length} passed</span>
      </div>
      <div className="p-4 space-y-2">
        {checks.map((check, i) => (
          <div key={i} className="flex items-start gap-2.5">
            <div className={clsx('mt-0.5 shrink-0', check.passed ? 'text-green-400' : 'text-red-400')}>
              {check.passed ? (
                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                </svg>
              ) : (
                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              )}
            </div>
            <div className="flex-1 min-w-0">
              <p className="text-sm text-slate-300">{check.label}</p>
              {check.message && (
                <p className="text-xs text-slate-500 mt-0.5">{check.message}</p>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

function ReportSectionCard({ data }: { data: { title?: string; content?: string; sectionId?: string } }) {
  const [collapsed, setCollapsed] = useState(false)

  return (
    <div className="bg-slate-800/40 border border-slate-600/40 rounded-xl overflow-hidden">
      <button
        onClick={() => setCollapsed(v => !v)}
        className="w-full flex items-center gap-2.5 px-4 py-3 text-left hover:bg-slate-700/20 transition-colors"
      >
        <svg className="w-4 h-4 text-slate-400 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
        </svg>
        <span className="text-sm font-medium text-slate-200 flex-1">{data.title ?? 'Report Section'}</span>
        <svg
          className={clsx('w-4 h-4 text-slate-500 transition-transform shrink-0', collapsed && 'rotate-180')}
          fill="none" viewBox="0 0 24 24" stroke="currentColor"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
        </svg>
      </button>
      {!collapsed && data.content && (
        <div className="border-t border-slate-700/40 px-4 py-4 prose prose-sm prose-invert max-w-none">
          <ReactMarkdown>{data.content}</ReactMarkdown>
        </div>
      )}
    </div>
  )
}

function CompleteCard({ data, jobId }: { data: { reportUrl?: string; downloadUrls?: Record<string, string>; summary?: string; jobId?: string }; jobId: string | null }) {
  return (
    <div className="bg-green-900/20 border border-green-500/40 rounded-xl px-5 py-4">
      <div className="flex items-center gap-3 mb-3">
        <svg className="w-6 h-6 text-green-400 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
        </svg>
        <h3 className="text-base font-semibold text-green-300">Analysis Complete</h3>
      </div>
      {data.summary && (
        <p className="text-sm text-slate-300 mb-4 leading-relaxed">{data.summary}</p>
      )}
      <div className="flex flex-wrap gap-2">
        {jobId && (
          <a
            href={`/report/${jobId}`}
            className="btn-primary text-sm flex items-center gap-1.5"
          >
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
            </svg>
            View Report
          </a>
        )}
        {data.downloadUrls && Object.entries(data.downloadUrls).map(([fmt, url]) => (
          <a
            key={fmt}
            href={url}
            download
            className="btn-secondary text-sm flex items-center gap-1.5"
          >
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
            </svg>
            Download {fmt.toUpperCase()}
          </a>
        ))}
      </div>
    </div>
  )
}

function ErrorCard({ data, onRetry }: { data: { message?: string; code?: string; retryable?: boolean }; onRetry?: () => void }) {
  return (
    <div className="bg-red-900/20 border border-red-500/40 rounded-xl px-5 py-4">
      <div className="flex items-center gap-3 mb-2">
        <svg className="w-5 h-5 text-red-400 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
        </svg>
        <h3 className="text-sm font-semibold text-red-300">Agent Error</h3>
        {data.code && <span className="text-xs text-red-500 font-mono ml-auto">{data.code}</span>}
      </div>
      <p className="text-sm text-slate-300 mb-3">{data.message ?? 'An unexpected error occurred.'}</p>
      {data.retryable && onRetry && (
        <button onClick={onRetry} className="btn-danger text-sm flex items-center gap-1.5">
          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
          </svg>
          Retry
        </button>
      )}
    </div>
  )
}

function StatusCard({ data }: { data: { message?: string } }) {
  return (
    <div className="flex items-center gap-2 text-xs text-slate-500 px-2">
      <div className="w-1.5 h-1.5 rounded-full bg-slate-600" />
      {data.message ?? 'Status update'}
    </div>
  )
}

function EventCard({ event, jobId, onRetry }: { event: AgentEvent; jobId: string | null; onRetry?: () => void }) {
  const data = event.data as Record<string, unknown>

  switch (event.type) {
    case 'thinking':
      return <ThinkingCard data={data as Parameters<typeof ThinkingCard>[0]['data']} />
    case 'code_generated':
      return <CodeGeneratedCard data={data as Parameters<typeof CodeGeneratedCard>[0]['data']} />
    case 'code_executing':
      return <CodeExecutingCard data={data as Parameters<typeof CodeExecutingCard>[0]['data']} />
    case 'code_result':
      return <CodeResultCard data={data as Parameters<typeof CodeResultCard>[0]['data']} />
    case 'chart_ready':
      return <ChartReadyCard data={data as Parameters<typeof ChartReadyCard>[0]['data']} />
    case 'validation':
      return <ValidationCard data={data as Parameters<typeof ValidationCard>[0]['data']} />
    case 'report_section':
      return <ReportSectionCard data={data as Parameters<typeof ReportSectionCard>[0]['data']} />
    case 'complete':
      return <CompleteCard data={data as Parameters<typeof CompleteCard>[0]['data']} jobId={jobId} />
    case 'error':
      return <ErrorCard data={data as Parameters<typeof ErrorCard>[0]['data']} onRetry={onRetry} />
    case 'status':
    case 'file_uploaded':
      return <StatusCard data={data as Parameters<typeof StatusCard>[0]['data']} />
    default:
      return (
        <div className="text-xs text-slate-600 font-mono px-2">
          [{event.type}] {JSON.stringify(data).slice(0, 80)}
        </div>
      )
  }
}

export default function AgentFeed({ events, isConnected, error, jobId, onRetry }: AgentFeedProps) {
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [events.length])

  if (!jobId) {
    return (
      <div className="flex-1 flex items-center justify-center text-slate-600">
        <div className="text-center">
          <svg className="w-12 h-12 mx-auto mb-3 text-slate-700" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
          </svg>
          <p className="text-sm">Submit a task to start the agent</p>
        </div>
      </div>
    )
  }

  return (
    <div className="flex flex-col h-full">
      {/* Connection status bar */}
      <div className="flex items-center gap-2 px-4 py-2 border-b border-slate-700/60 bg-slate-800/40">
        <div className={clsx('w-2 h-2 rounded-full shrink-0', isConnected ? 'bg-green-400 animate-pulse' : 'bg-slate-600')} />
        <span className="text-xs text-slate-400">
          {isConnected ? 'Connected — streaming events' : 'Disconnected'}
        </span>
        {error && (
          <span className="text-xs text-yellow-400 ml-2">{error}</span>
        )}
        <span className="ml-auto text-xs text-slate-600">{events.length} events</span>
      </div>

      {/* Events list */}
      <div className="flex-1 overflow-y-auto p-4 space-y-3">
        {events.length === 0 && isConnected && (
          <div className="flex items-center gap-2 text-slate-500 text-sm">
            <svg className="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
            </svg>
            Waiting for agent to start...
          </div>
        )}

        {events.map((event) => (
          <div key={event.id} className="flex gap-3">
            {/* Timeline dot */}
            <div className="flex flex-col items-center">
              <div className="w-2 h-2 rounded-full bg-slate-600 mt-2 shrink-0" />
              <div className="w-px flex-1 bg-slate-700/40 mt-1" />
            </div>
            {/* Card */}
            <div className="flex-1 pb-1">
              <p className="text-xs text-slate-600 mb-1.5 font-mono">
                {new Date(event.timestamp).toLocaleTimeString()}
              </p>
              <EventCard event={event} jobId={jobId} onRetry={onRetry} />
            </div>
          </div>
        ))}
        <div ref={bottomRef} />
      </div>
    </div>
  )
}
