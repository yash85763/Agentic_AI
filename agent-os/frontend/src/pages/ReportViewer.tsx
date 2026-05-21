import React, { useEffect, useState } from 'react'
import { useParams, Link } from 'react-router-dom'
import axios from 'axios'
import ReactMarkdown from 'react-markdown'
import EChartRenderer from '../components/EChartRenderer'

interface ReportSection {
  id: string
  title: string
  content: string
  chart?: any
}

interface Report {
  job_id: string
  title: string
  executive_summary: string
  sections: ReportSection[]
  charts: any[]
  validation_summary?: any
  generated_at: string
}

interface Job {
  id: string
  status: string
  task_description: string
  result?: {
    report?: Report
    charts?: any[]
    merged_path?: string
    validation?: any
  }
}

export default function ReportViewer() {
  const { jobId } = useParams<{ jobId: string }>()
  const [job, setJob] = useState<Job | null>(null)
  const [loading, setLoading] = useState(true)
  const [regenerating, setRegenerating] = useState<string | null>(null)

  useEffect(() => {
    if (!jobId) return
    axios
      .get<Job>(`/api/jobs/${jobId}`)
      .then(res => setJob(res.data))
      .catch(console.error)
      .finally(() => setLoading(false))
  }, [jobId])

  const downloadExcel = async () => {
    if (!jobId) return
    try {
      const res = await axios.get(`/api/jobs/${jobId}/export/excel`, {
        responseType: 'blob',
      })
      const url = URL.createObjectURL(res.data)
      const a = document.createElement('a')
      a.href = url
      a.download = `report-${jobId}.xlsx`
      a.click()
      URL.revokeObjectURL(url)
    } catch (e) {
      console.error(e)
      alert('Excel export failed')
    }
  }

  const downloadPDF = async () => {
    window.print()
  }

  const regenerateSection = async (sectionId: string) => {
    setRegenerating(sectionId)
    try {
      await axios.post(`/api/jobs/${jobId}/sections/${sectionId}/regenerate`)
      const res = await axios.get<Job>(`/api/jobs/${jobId}`)
      setJob(res.data)
    } catch (e) {
      console.error(e)
    } finally {
      setRegenerating(null)
    }
  }

  if (loading) {
    return (
      <div className="p-8">
        <div className="animate-pulse space-y-4">
          <div className="h-8 bg-slate-800 rounded w-1/3"></div>
          <div className="h-4 bg-slate-800 rounded w-2/3"></div>
          <div className="h-64 bg-slate-800 rounded"></div>
        </div>
      </div>
    )
  }

  if (!job) {
    return (
      <div className="p-8 text-center">
        <p className="text-slate-400">Report not found.</p>
        <Link to="/" className="text-blue-400 hover:text-blue-300 text-sm mt-2 inline-block">
          ← Back to Dashboard
        </Link>
      </div>
    )
  }

  const report = job.result?.report
  const charts = job.result?.charts ?? report?.charts ?? []

  return (
    <div className="max-w-5xl mx-auto p-8 print:p-0">
      {/* Header */}
      <div className="flex items-start justify-between mb-8 print:hidden">
        <div>
          <Link to="/" className="text-slate-500 hover:text-slate-300 text-sm">
            ← Back to Dashboard
          </Link>
          <h1 className="text-2xl font-bold text-slate-100 mt-2">
            {report?.title ?? job.task_description}
          </h1>
          {report?.generated_at && (
            <p className="text-sm text-slate-400 mt-1">
              Generated {new Date(report.generated_at).toLocaleString()}
            </p>
          )}
        </div>
        <div className="flex gap-2">
          <button onClick={downloadPDF} className="btn-secondary text-sm inline-flex items-center gap-2">
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
            </svg>
            PDF
          </button>
          <button onClick={downloadExcel} className="btn-primary text-sm inline-flex items-center gap-2">
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
            </svg>
            Excel
          </button>
        </div>
      </div>

      {!report ? (
        <div className="card text-center py-12">
          <p className="text-slate-400">
            {job.status === 'complete'
              ? 'Report data not available for this job.'
              : `Job is ${job.status}. Report will appear here when complete.`}
          </p>
          <Link to={`/workspace/${jobId}`} className="text-blue-400 hover:text-blue-300 text-sm mt-3 inline-block">
            View live agent activity →
          </Link>
        </div>
      ) : (
        <>
          {/* Executive Summary */}
          {report.executive_summary && (
            <section className="card mb-6">
              <h2 className="text-lg font-semibold text-slate-100 mb-3">Executive Summary</h2>
              <div className="prose prose-invert prose-sm max-w-none">
                <ReactMarkdown>{report.executive_summary}</ReactMarkdown>
              </div>
            </section>
          )}

          {/* Validation summary */}
          {job.result?.validation && (
            <section className="card mb-6">
              <h2 className="text-lg font-semibold text-slate-100 mb-3 flex items-center gap-2">
                <svg className="w-5 h-5 text-green-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                    d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                Data Validation
              </h2>
              <pre className="text-xs text-slate-300 bg-slate-900 rounded p-3 overflow-auto">
                {JSON.stringify(job.result.validation, null, 2)}
              </pre>
            </section>
          )}

          {/* Charts */}
          {charts.length > 0 && (
            <section className="mb-6">
              <h2 className="text-lg font-semibold text-slate-100 mb-3">Visualizations</h2>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {charts.map((chart, i) => (
                  <div key={i} className="card">
                    <EChartRenderer option={chart.config ?? chart} height={320} />
                  </div>
                ))}
              </div>
            </section>
          )}

          {/* Sections */}
          {report.sections?.map(section => (
            <section key={section.id} className="card mb-6">
              <div className="flex items-start justify-between mb-3 print:hidden">
                <h2 className="text-lg font-semibold text-slate-100">{section.title}</h2>
                <button
                  onClick={() => regenerateSection(section.id)}
                  disabled={regenerating === section.id}
                  className="btn-ghost text-xs"
                >
                  {regenerating === section.id ? (
                    <span className="spinner"></span>
                  ) : (
                    <>↻ Regenerate</>
                  )}
                </button>
              </div>
              <h2 className="text-lg font-semibold text-slate-100 mb-3 hidden print:block">
                {section.title}
              </h2>
              <div className="prose prose-invert prose-sm max-w-none">
                <ReactMarkdown>{section.content}</ReactMarkdown>
              </div>
              {section.chart && (
                <div className="mt-4">
                  <EChartRenderer option={section.chart} height={320} />
                </div>
              )}
            </section>
          ))}
        </>
      )}
    </div>
  )
}
