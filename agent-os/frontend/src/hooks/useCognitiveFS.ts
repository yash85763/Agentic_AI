import { useState, useCallback, useEffect } from 'react'
import axios from 'axios'

export interface FSNode {
  name: string
  path: string
  type: 'file' | 'directory'
  children?: FSNode[]
  size?: number
  modified?: string
  mimeType?: string
}

export interface UseCognitiveFSResult {
  tree: FSNode[]
  selectedPath: string | null
  content: string
  loading: boolean
  saving: boolean
  error: string | null
  selectFile: (path: string) => Promise<void>
  save: (path: string, content: string) => Promise<void>
  create: (path: string, initialContent?: string) => Promise<void>
  remove: (path: string) => Promise<void>
  refresh: () => Promise<void>
}

async function fetchTree(): Promise<FSNode[]> {
  const res = await axios.get<{ tree: FSNode[] } | FSNode[]>('/api/cognitive')
  const data = res.data
  if (Array.isArray(data)) return data
  return (data as { tree: FSNode[] }).tree ?? []
}

async function fetchContent(path: string): Promise<string> {
  const encoded = encodeURIComponent(path)
  const res = await axios.get<{ content: string } | string>(`/api/cognitive/${encoded}`)
  const data = res.data
  if (typeof data === 'string') return data
  return (data as { content: string }).content ?? ''
}

export function useCognitiveFS(): UseCognitiveFSResult {
  const [tree, setTree] = useState<FSNode[]>([])
  const [selectedPath, setSelectedPath] = useState<string | null>(null)
  const [content, setContent] = useState<string>('')
  const [loading, setLoading] = useState(false)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const refresh = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const data = await fetchTree()
      setTree(data)
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Failed to load file tree'
      setError(msg)
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    refresh()
  }, [refresh])

  const selectFile = useCallback(async (path: string) => {
    setSelectedPath(path)
    setLoading(true)
    setError(null)
    try {
      const text = await fetchContent(path)
      setContent(text)
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Failed to load file content'
      setError(msg)
      setContent('')
    } finally {
      setLoading(false)
    }
  }, [])

  const save = useCallback(async (path: string, newContent: string) => {
    setSaving(true)
    setError(null)
    try {
      const encoded = encodeURIComponent(path)
      await axios.put(`/api/cognitive/${encoded}`, { content: newContent })
      setContent(newContent)
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Failed to save file'
      setError(msg)
      throw err
    } finally {
      setSaving(false)
    }
  }, [])

  const create = useCallback(async (path: string, initialContent = '') => {
    setLoading(true)
    setError(null)
    try {
      const encoded = encodeURIComponent(path)
      await axios.post(`/api/cognitive/${encoded}`, { content: initialContent })
      await refresh()
      await selectFile(path)
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Failed to create file'
      setError(msg)
      throw err
    } finally {
      setLoading(false)
    }
  }, [refresh, selectFile])

  const remove = useCallback(async (path: string) => {
    setLoading(true)
    setError(null)
    try {
      const encoded = encodeURIComponent(path)
      await axios.delete(`/api/cognitive/${encoded}`)
      if (selectedPath === path) {
        setSelectedPath(null)
        setContent('')
      }
      await refresh()
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Failed to delete file'
      setError(msg)
      throw err
    } finally {
      setLoading(false)
    }
  }, [selectedPath, refresh])

  return {
    tree,
    selectedPath,
    content,
    loading,
    saving,
    error,
    selectFile,
    save,
    create,
    remove,
    refresh,
  }
}
