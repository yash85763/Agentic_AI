import { useState, useEffect, useRef, useCallback } from 'react'

export type AgentEventType =
  | 'thinking'
  | 'code_generated'
  | 'code_executing'
  | 'code_result'
  | 'chart_ready'
  | 'validation'
  | 'report_section'
  | 'complete'
  | 'error'
  | 'status'
  | 'file_uploaded'

export interface AgentEvent {
  id: string
  type: AgentEventType
  timestamp: string
  jobId: string
  data: Record<string, unknown>
}

export interface ThinkingData {
  text: string
  streaming?: boolean
}

export interface CodeGeneratedData {
  language: string
  code: string
  filename?: string
}

export interface CodeExecutingData {
  message: string
  sandboxId?: string
}

export interface CodeResultData {
  stdout?: string
  stderr?: string
  returnValue?: unknown
  executionTime?: number
  success: boolean
}

export interface ChartReadyData {
  option: Record<string, unknown>
  title?: string
  chartId?: string
}

export interface ValidationData {
  checks: Array<{
    label: string
    passed: boolean
    message?: string
  }>
}

export interface ReportSectionData {
  title: string
  content: string
  sectionId?: string
}

export interface CompleteData {
  reportUrl?: string
  downloadUrls?: Record<string, string>
  summary?: string
  jobId: string
}

export interface ErrorData {
  message: string
  code?: string
  retryable?: boolean
}

export interface UseJobStreamResult {
  events: AgentEvent[]
  isConnected: boolean
  error: string | null
  latestEvent: AgentEvent | null
  clearEvents: () => void
}

const MAX_RECONNECT_ATTEMPTS = 5
const RECONNECT_BASE_DELAY = 1000

export function useJobStream(jobId: string | null): UseJobStreamResult {
  const [events, setEvents] = useState<AgentEvent[]>([])
  const [isConnected, setIsConnected] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [latestEvent, setLatestEvent] = useState<AgentEvent | null>(null)

  const esRef = useRef<EventSource | null>(null)
  const reconnectAttemptsRef = useRef(0)
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const isTerminalRef = useRef(false)

  const clearEvents = useCallback(() => {
    setEvents([])
    setLatestEvent(null)
    setError(null)
  }, [])

  const disconnect = useCallback(() => {
    if (reconnectTimerRef.current) {
      clearTimeout(reconnectTimerRef.current)
      reconnectTimerRef.current = null
    }
    if (esRef.current) {
      esRef.current.close()
      esRef.current = null
    }
    setIsConnected(false)
  }, [])

  const connect = useCallback((id: string) => {
    if (isTerminalRef.current) return

    disconnect()

    const url = `/api/jobs/${id}/stream`
    const es = new EventSource(url)
    esRef.current = es

    es.onopen = () => {
      setIsConnected(true)
      setError(null)
      reconnectAttemptsRef.current = 0
    }

    es.onmessage = (ev) => {
      try {
        const parsed: AgentEvent = JSON.parse(ev.data)
        const event: AgentEvent = {
          ...parsed,
          id: parsed.id ?? `${Date.now()}-${Math.random().toString(36).slice(2)}`,
          timestamp: parsed.timestamp ?? new Date().toISOString(),
        }

        setEvents(prev => [...prev, event])
        setLatestEvent(event)

        // Stop reconnecting if we hit a terminal event
        if (event.type === 'complete' || event.type === 'error') {
          isTerminalRef.current = true
          es.close()
          esRef.current = null
          setIsConnected(false)
        }
      } catch {
        console.warn('[useJobStream] Failed to parse SSE message:', ev.data)
      }
    }

    es.onerror = () => {
      setIsConnected(false)
      es.close()
      esRef.current = null

      if (isTerminalRef.current) return

      const attempts = reconnectAttemptsRef.current
      if (attempts < MAX_RECONNECT_ATTEMPTS) {
        const delay = RECONNECT_BASE_DELAY * Math.pow(2, attempts)
        reconnectAttemptsRef.current += 1
        setError(`Connection lost. Reconnecting in ${Math.round(delay / 1000)}s... (attempt ${attempts + 1}/${MAX_RECONNECT_ATTEMPTS})`)
        reconnectTimerRef.current = setTimeout(() => {
          connect(id)
        }, delay)
      } else {
        setError('Failed to connect to agent stream after multiple attempts.')
      }
    }
  }, [disconnect])

  useEffect(() => {
    if (!jobId) {
      disconnect()
      return
    }

    isTerminalRef.current = false
    reconnectAttemptsRef.current = 0
    setEvents([])
    setLatestEvent(null)
    setError(null)

    connect(jobId)

    return () => {
      disconnect()
    }
  }, [jobId, connect, disconnect])

  return { events, isConnected, error, latestEvent, clearEvents }
}
