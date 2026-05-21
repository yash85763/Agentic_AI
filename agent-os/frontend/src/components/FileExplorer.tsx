import React, { useState, useCallback } from 'react'
import clsx from 'clsx'
import type { FSNode } from '../hooks/useCognitiveFS'

interface FileExplorerProps {
  tree: FSNode[]
  selectedPath: string | null
  loading: boolean
  onSelect: (path: string) => void
  onCreate: (path: string, content?: string) => Promise<void>
  onDelete: (path: string) => Promise<void>
  onRefresh: () => Promise<void>
}

function getFileIcon(name: string): string {
  const ext = name.split('.').pop()?.toLowerCase()
  switch (ext) {
    case 'md': return '📄'
    case 'json': return '🔵'
    case 'yaml':
    case 'yml': return '🟡'
    case 'ts':
    case 'tsx': return '🟦'
    case 'py': return '🐍'
    case 'sh': return '⚙️'
    default: return '📃'
  }
}

interface TreeNodeProps {
  node: FSNode
  selectedPath: string | null
  depth: number
  onSelect: (path: string) => void
  onDelete: (path: string) => Promise<void>
  onCreateChild: (parentPath: string) => void
}

function TreeNode({ node, selectedPath, depth, onSelect, onDelete, onCreateChild }: TreeNodeProps) {
  const [expanded, setExpanded] = useState(depth < 2)
  const [hovering, setHovering] = useState(false)
  const [deleting, setDeleting] = useState(false)

  const isSelected = selectedPath === node.path
  const isDir = node.type === 'directory'

  const handleDelete = useCallback(async (e: React.MouseEvent) => {
    e.stopPropagation()
    if (!confirm(`Delete "${node.name}"?`)) return
    setDeleting(true)
    try {
      await onDelete(node.path)
    } finally {
      setDeleting(false)
    }
  }, [node, onDelete])

  const handleCreateChild = useCallback((e: React.MouseEvent) => {
    e.stopPropagation()
    onCreateChild(node.path)
  }, [node.path, onCreateChild])

  return (
    <div>
      <div
        className={clsx(
          'group flex items-center gap-1.5 px-2 py-1.5 rounded-md cursor-pointer transition-colors text-sm select-none',
          isSelected && !isDir ? 'bg-blue-600/20 text-blue-300' : 'text-slate-300 hover:bg-slate-700/50'
        )}
        style={{ paddingLeft: `${(depth * 12) + 8}px` }}
        onMouseEnter={() => setHovering(true)}
        onMouseLeave={() => setHovering(false)}
        onClick={() => {
          if (isDir) setExpanded(v => !v)
          else onSelect(node.path)
        }}
      >
        {isDir ? (
          <svg
            className={clsx('w-3.5 h-3.5 text-slate-500 transition-transform shrink-0', expanded && 'rotate-90')}
            fill="none" viewBox="0 0 24 24" stroke="currentColor"
          >
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
          </svg>
        ) : (
          <span className="text-xs shrink-0">{getFileIcon(node.name)}</span>
        )}

        {isDir && (
          <svg className="w-4 h-4 text-yellow-400/70 shrink-0" fill="currentColor" viewBox="0 0 20 20">
            <path d="M2 6a2 2 0 012-2h5l2 2h5a2 2 0 012 2v6a2 2 0 01-2 2H4a2 2 0 01-2-2V6z" />
          </svg>
        )}

        <span className="flex-1 truncate text-xs font-medium">{node.name}</span>

        {/* Actions */}
        {(hovering || isSelected) && (
          <div className="flex items-center gap-0.5 shrink-0" onClick={e => e.stopPropagation()}>
            {isDir && (
              <button
                onClick={handleCreateChild}
                title="New file in folder"
                className="p-0.5 rounded text-slate-500 hover:text-slate-200 hover:bg-slate-600"
              >
                <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
                </svg>
              </button>
            )}
            <button
              onClick={handleDelete}
              disabled={deleting}
              title="Delete"
              className="p-0.5 rounded text-slate-500 hover:text-red-400 hover:bg-slate-600"
            >
              {deleting ? (
                <svg className="w-3.5 h-3.5 animate-spin" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                </svg>
              ) : (
                <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                </svg>
              )}
            </button>
          </div>
        )}
      </div>

      {isDir && expanded && node.children && (
        <div>
          {node.children.map(child => (
            <TreeNode
              key={child.path}
              node={child}
              selectedPath={selectedPath}
              depth={depth + 1}
              onSelect={onSelect}
              onDelete={onDelete}
              onCreateChild={onCreateChild}
            />
          ))}
          {node.children.length === 0 && (
            <div
              className="text-xs text-slate-600 italic"
              style={{ paddingLeft: `${((depth + 1) * 12) + 8}px`, paddingTop: '4px', paddingBottom: '4px' }}
            >
              Empty folder
            </div>
          )}
        </div>
      )}
    </div>
  )
}

interface NewFileDialogProps {
  parentPath: string
  onConfirm: (path: string) => void
  onCancel: () => void
}

function NewFileDialog({ parentPath, onConfirm, onCancel }: NewFileDialogProps) {
  const [name, setName] = useState('')

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (!name.trim()) return
    const fullPath = parentPath ? `${parentPath}/${name.trim()}` : name.trim()
    onConfirm(fullPath)
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm">
      <form
        onSubmit={handleSubmit}
        className="bg-slate-800 border border-slate-600 rounded-xl p-5 w-80 shadow-2xl"
      >
        <h3 className="text-sm font-semibold text-slate-200 mb-3">New File</h3>
        {parentPath && (
          <p className="text-xs text-slate-400 mb-2 font-mono">{parentPath}/</p>
        )}
        <input
          autoFocus
          type="text"
          placeholder="filename.md"
          value={name}
          onChange={e => setName(e.target.value)}
          className="input w-full text-sm mb-4"
        />
        <div className="flex gap-2 justify-end">
          <button type="button" onClick={onCancel} className="btn-secondary text-sm px-3 py-1.5">
            Cancel
          </button>
          <button type="submit" disabled={!name.trim()} className="btn-primary text-sm px-3 py-1.5">
            Create
          </button>
        </div>
      </form>
    </div>
  )
}

export default function FileExplorer({
  tree,
  selectedPath,
  loading,
  onSelect,
  onCreate,
  onDelete,
  onRefresh,
}: FileExplorerProps) {
  const [newFileParent, setNewFileParent] = useState<string | null>(null)

  const handleCreateChild = useCallback((parentPath: string) => {
    setNewFileParent(parentPath)
  }, [])

  const handleNewFileConfirm = useCallback(async (path: string) => {
    setNewFileParent(null)
    await onCreate(path, '')
  }, [onCreate])

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="flex items-center justify-between px-3 py-2 border-b border-slate-700/60">
        <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Files</span>
        <div className="flex items-center gap-1">
          <button
            onClick={() => setNewFileParent('')}
            title="New file"
            className="p-1 rounded text-slate-400 hover:text-slate-200 hover:bg-slate-700"
          >
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
            </svg>
          </button>
          <button
            onClick={onRefresh}
            title="Refresh"
            className="p-1 rounded text-slate-400 hover:text-slate-200 hover:bg-slate-700"
          >
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
            </svg>
          </button>
        </div>
      </div>

      {/* Tree */}
      <div className="flex-1 overflow-y-auto py-2 px-1">
        {loading ? (
          <div className="space-y-1 px-2 py-1">
            {Array.from({ length: 6 }).map((_, i) => (
              <div key={i} className="h-6 bg-slate-700/50 rounded animate-pulse" style={{ width: `${50 + i * 8}%` }} />
            ))}
          </div>
        ) : tree.length === 0 ? (
          <div className="text-center py-8 px-3">
            <svg className="w-8 h-8 text-slate-600 mx-auto mb-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-6l-2-2H5a2 2 0 00-2 2z" />
            </svg>
            <p className="text-xs text-slate-500">No files yet</p>
            <button
              onClick={() => setNewFileParent('')}
              className="mt-2 text-xs text-blue-400 hover:text-blue-300"
            >
              Create first file
            </button>
          </div>
        ) : (
          tree.map(node => (
            <TreeNode
              key={node.path}
              node={node}
              selectedPath={selectedPath}
              depth={0}
              onSelect={onSelect}
              onDelete={onDelete}
              onCreateChild={handleCreateChild}
            />
          ))
        )}
      </div>

      {/* New File Dialog */}
      {newFileParent !== null && (
        <NewFileDialog
          parentPath={newFileParent}
          onConfirm={handleNewFileConfirm}
          onCancel={() => setNewFileParent(null)}
        />
      )}
    </div>
  )
}
