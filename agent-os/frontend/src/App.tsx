import React, { useState } from 'react'
import { Routes, Route, NavLink, useNavigate } from 'react-router-dom'
import clsx from 'clsx'
import Dashboard from './pages/Dashboard'
import Workspace from './pages/Workspace'
import ReportViewer from './pages/ReportViewer'
import Studio from './pages/Studio'

const NAV_ITEMS = [
  {
    to: '/',
    label: 'Dashboard',
    icon: (
      <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
          d="M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 001 1m-6 0h6" />
      </svg>
    ),
    end: true,
  },
  {
    to: '/workspace',
    label: 'Workspace',
    icon: (
      <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
          d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
      </svg>
    ),
  },
  {
    to: '/studio',
    label: 'Studio',
    icon: (
      <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
          d="M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4" />
      </svg>
    ),
  },
]

function Sidebar({ collapsed, onToggle }: { collapsed: boolean; onToggle: () => void }) {
  return (
    <aside
      className={clsx(
        'flex flex-col bg-slate-900 border-r border-slate-700/60 transition-all duration-300 h-screen sticky top-0 shrink-0',
        collapsed ? 'w-16' : 'w-56'
      )}
    >
      {/* Logo */}
      <div className="flex items-center gap-3 px-4 py-5 border-b border-slate-700/60">
        <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-blue-500 to-cyan-500 flex items-center justify-center shrink-0">
          <svg className="w-5 h-5 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
              d="M13 10V3L4 14h7v7l9-11h-7z" />
          </svg>
        </div>
        {!collapsed && (
          <span className="font-bold text-lg text-gradient tracking-tight">AgentOS</span>
        )}
      </div>

      {/* Nav */}
      <nav className="flex-1 px-2 py-4 space-y-1">
        {NAV_ITEMS.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            end={item.end}
            className={({ isActive }) =>
              clsx(
                'flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-colors duration-150',
                isActive
                  ? 'bg-blue-600/20 text-blue-400 border border-blue-500/30'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800'
              )
            }
          >
            {item.icon}
            {!collapsed && <span>{item.label}</span>}
          </NavLink>
        ))}
      </nav>

      {/* Footer toggle */}
      <div className="p-2 border-t border-slate-700/60">
        <button
          onClick={onToggle}
          className="w-full flex items-center justify-center px-3 py-2 rounded-lg text-slate-500 hover:text-slate-300 hover:bg-slate-800 transition-colors duration-150"
          title={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
        >
          <svg
            className={clsx('w-4 h-4 transition-transform duration-300', collapsed && 'rotate-180')}
            fill="none" viewBox="0 0 24 24" stroke="currentColor"
          >
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 19l-7-7 7-7m8 14l-7-7 7-7" />
          </svg>
          {!collapsed && <span className="ml-2 text-xs">Collapse</span>}
        </button>
      </div>
    </aside>
  )
}

export default function App() {
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false)

  return (
    <div className="flex min-h-screen bg-slate-900">
      <Sidebar collapsed={sidebarCollapsed} onToggle={() => setSidebarCollapsed(v => !v)} />

      <main className="flex-1 overflow-auto">
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/workspace" element={<Workspace />} />
          <Route path="/workspace/:jobId" element={<Workspace />} />
          <Route path="/report/:jobId" element={<ReportViewer />} />
          <Route path="/studio" element={<Studio />} />
        </Routes>
      </main>
    </div>
  )
}
