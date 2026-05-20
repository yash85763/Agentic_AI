import React from 'react'
import Editor, { OnChange } from '@monaco-editor/react'

interface Props {
  value: string
  language: string
  onChange?: (value: string | undefined) => void
  readOnly?: boolean
}

export default function MonacoEditor({ value, language, onChange, readOnly = false }: Props) {
  return (
    <Editor
      height="100%"
      defaultLanguage={language}
      language={language}
      value={value}
      onChange={onChange as OnChange}
      theme="vs-dark"
      options={{
        readOnly,
        minimap: { enabled: false },
        fontSize: 13,
        fontFamily: 'ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace',
        wordWrap: 'on',
        lineNumbers: 'on',
        scrollBeyondLastLine: false,
        automaticLayout: true,
        renderLineHighlight: 'gutter',
        tabSize: 2,
        formatOnPaste: true,
        formatOnType: true,
        padding: { top: 12 },
      }}
      loading={
        <div className="h-full flex items-center justify-center text-slate-500 text-sm">
          Loading editor...
        </div>
      }
    />
  )
}
