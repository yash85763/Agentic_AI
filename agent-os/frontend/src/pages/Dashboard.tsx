import React, { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import axios from 'axios'
import { format, formatDistanceToNow } from 'date-fns'
import clsx from 'clsx'

type JobStatus = 'pending' | 'running' | 'complete' | 'failed'

interface Job {
  id: string
  status: JobStatus
  task_description: string
  file_ids: string[]
  created_at: string
  updated_at?: string
}

interface HealthCheck {
  status: string
  checks: Record<string, string>
  version: string
}

function StatusBadge({ status }: { status: JobStatus }) {
  const styles: Record<JobStatus, string> = {
    pending: 'badge-slate',
    running: 'badge-blue animate-pulse',
    complete: 'badge-green',
    failed: 'badge-red',
  }
  return <span className={styles[status]}>{status}</span>
}

function StatCard({
  label,
  value,
  accent,
  icon,
}: {
  label: string
  value: number | string
  accent: 'blue' | 'green' | 'amber' | 'red'
  icon: React.ReactNode
}) {
  const accents = {
    blue: 'text-blue-400 bg-blue-500/10 border-blue-500/20',
    green: 'text-green-400 bg-green-500/10 border-green-500/20',
    amber: 'text-amber-400 bg-amber-500/10 border-amber-500/20',
    red: 'text-red-400 bg-red-500/10 border-red-500/20',
  }
  return (
    <div className="card">
      <div className="flex items-start justify-between">
        <div>
          <p className="text-sm text-slate-400 mb-1">{label}</p>
          <p className="text-3xl font-bold text-slate-100">{value}</p>
        </div>
        <div className={clsx('p-2 rounded-lg border', accents[accent])}>{icon}</div>
      </div>
    </div>
  )
}

function HealthIndicator({ checks }: { checks: Record<string, string> }) {
  return (
    <div className="card">
      <h3 className="text-sm font-semibold text-slate-300 mb-3">System Health</h3>
      <div className="space-y-2">
        {Object.entries(checks).map(([k, v]) => (
          <div key={k} className="flex items-center justify-between text-sm">
            <span className="text-slate-400 capitalize">{k.replace('_', ' ')}</span>
            <div className="flex items-center gap-2">
              <span
                className={clsx(
                  'status-dot',
                  v === 'ok' ? 'bg-green-500' : v === 'missing' ? 'bg-amber-500' : 'bg-red-500'
                )}
              />
              <span className={v === 'ok' ? 'text-green-400' : 'text-red-400'}>{v}</span>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

export default function Dashboard() {
  const [jobs, setJobs] = useState<Job[]>([])
  const [health, setHealth] = useState<HealthCheck | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    Promise.all([axios.get('/api/jobs'), axios.get('/api/health')])
      .then(([jobsRes, healthRes]) => {
        setJobs(jobsRes.data)
        setHealth(healthRes.data)
      })
      .catch(console.error)
      .finally(() => setLoading(false))
  }, [])

  const stats = {
    total: jobs.length,
    running: jobs.filter(j => j.status === 'running').length,
    completed: jobs.filter(j => j.status === 'complete').length,
    failed: jobs.filter(j => j.status === 'failed').length,
  }

  return (
    <div className="p-8 max-w-7xl mx-auto">
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-2xl font-bold text-slate-100">Dashboard</h1>
          <p className="text-sm text-slate-400 mt-1">
            Overview of agent activity and system status
          </p>
        </div>
        <Link to="/workspace" className="btn-primary inline-flex items-center gap-2">
          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
          </svg>
          New Analysis
        </Link>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
        <StatCard
          label="Total Jobs"
          value={stats.total}
          accent="blue"
          icon={
            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                d="M9 17V7m0 10a2 2 0 01-2 2H5a2 2 0 01-2-2V7a2 2 0 012-2h2a2 2 0 012 2m0 10a2 2 0 002 2h2a2 2 0 002-2M9 7a2 2 0 012-2h2a2 2 0 012 2m0 10V7m0 10a2 2 0 002 2h2a2 2 0 002-2V7a2 2 0 00-2-2h-2a2 2 0 00-2 2" />
            </svg>
          }
        />
        <StatCard
          label="Running"
          value={stats.running}
          accent="amber"
          icon={
            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          }
        />
        <StatCard
          label="Completed"
          value={stats.completed}
          accent="green"
          icon={
            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                d="M5 13l4 4L19 7" />
            </svg>
          }
        />
        <StatCard
          label="Failed"
          value={stats.failed}
          accent="red"
          icon={
            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          }
        />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 card">
          <h3 className="text-sm font-semibold text-slate-300 mb-4">Recent Jobs</h3>
          {loading ? (
            <p className="text-slate-500 text-sm">Loading...</p>
          ) : jobs.length === 0 ? (
            <div className="text-center py-8">
              <p className="text-slate-500 text-sm mb-4">No jobs yet</p>
              <Link to="/workspace" className="btn-primary inline-block">
                Start your first analysis
              </Link>
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="text-left text-slate-400 border-b border-slate-700">
                    <th className="py-2 px-2 font-medium">Task</th>
                    <th className="py-2 px-2 font-medium">Status</th>
                    <th className="py-2 px-2 font-medium">Created</th>
                    <th className="py-2 px-2 font-medium"></th>
                  </tr>
                </thead>
                <tbody>
                  {jobs.slice(0, 10).map(job => (
                    <tr
                      key={job.id}
                      className="border-b border-slate-700/40 hover:bg-slate-700/20"
                    >
                      <td className="py-2 px-2 max-w-md truncate text-slate-200">
                        {job.task_description}
                      </td>
                      <td className="py-2 px-2">
                        <StatusBadge status={job.status} />
                      </td>
                      <td className="py-2 px-2 text-slate-400">
                        {formatDistanceToNow(new Date(job.created_at), { addSuffix: true })}
                      </td>
                      <td className="py-2 px-2">
                        <Link
                          to={job.status === 'complete' ? `/report/${job.id}` : `/workspace/${job.id}`}
                          className="text-blue-400 hover:text-blue-300 text-xs"
                        >
                          {job.status === 'complete' ? 'View report →' : 'Open →'}
                        </Link>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>

        <div className="space-y-4">
          {health && <HealthIndicator checks={health.checks} />}
          <div className="card">
            <h3 className="text-sm font-semibold text-slate-300 mb-3">Quick Actions</h3>
            <div className="space-y-2">
              <Link to="/workspace" className="btn-secondary w-full text-sm text-center block">
                New Analysis
              </Link>
              <Link to="/studio" className="btn-ghost w-full text-sm text-center block">
                Edit Agent Config
              </Link>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
