import React, { useRef, useCallback } from 'react'
import ReactECharts from 'echarts-for-react'
import type { EChartsOption } from 'echarts'

interface EChartRendererProps {
  option: Record<string, unknown>
  height?: number | string
  title?: string
  className?: string
}

function SkeletonChart({ height }: { height: number | string }) {
  return (
    <div
      className="bg-slate-700/40 rounded-lg animate-pulse flex items-center justify-center"
      style={{ height }}
    >
      <svg className="w-12 h-12 text-slate-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1}
          d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
      </svg>
    </div>
  )
}

class ChartErrorBoundary extends React.Component<
  { children: React.ReactNode; fallback?: React.ReactNode },
  { hasError: boolean; errorMessage: string }
> {
  constructor(props: { children: React.ReactNode; fallback?: React.ReactNode }) {
    super(props)
    this.state = { hasError: false, errorMessage: '' }
  }

  static getDerivedStateFromError(error: Error) {
    return { hasError: true, errorMessage: error.message }
  }

  render() {
    if (this.state.hasError) {
      return this.props.fallback ?? (
        <div className="bg-red-900/20 border border-red-700/40 rounded-lg p-4 text-red-400 text-sm">
          <p className="font-medium">Chart rendering failed</p>
          <p className="text-xs mt-1 text-red-500">{this.state.errorMessage}</p>
        </div>
      )
    }
    return this.props.children
  }
}

const DARK_THEME_DEFAULTS: Partial<EChartsOption> = {
  backgroundColor: 'transparent',
  textStyle: { color: '#94a3b8' },
  title: { textStyle: { color: '#e2e8f0' } },
  legend: { textStyle: { color: '#94a3b8' } },
  xAxis: {
    axisLine: { lineStyle: { color: '#334155' } },
    axisTick: { lineStyle: { color: '#334155' } },
    axisLabel: { color: '#94a3b8' },
    splitLine: { lineStyle: { color: '#1e293b' } },
  },
  yAxis: {
    axisLine: { lineStyle: { color: '#334155' } },
    axisTick: { lineStyle: { color: '#334155' } },
    axisLabel: { color: '#94a3b8' },
    splitLine: { lineStyle: { color: '#1e293b' } },
  },
  tooltip: {
    backgroundColor: '#1e293b',
    borderColor: '#334155',
    textStyle: { color: '#e2e8f0' },
  },
}

function deepMerge(base: Record<string, unknown>, override: Record<string, unknown>): Record<string, unknown> {
  const result = { ...base }
  for (const key of Object.keys(override)) {
    const baseVal = result[key]
    const overrideVal = override[key]
    if (
      overrideVal !== null &&
      typeof overrideVal === 'object' &&
      !Array.isArray(overrideVal) &&
      baseVal !== null &&
      typeof baseVal === 'object' &&
      !Array.isArray(baseVal)
    ) {
      result[key] = deepMerge(
        baseVal as Record<string, unknown>,
        overrideVal as Record<string, unknown>
      )
    } else {
      result[key] = overrideVal
    }
  }
  return result
}

export default function EChartRenderer({
  option,
  height = 320,
  title,
  className = '',
}: EChartRendererProps) {
  const chartRef = useRef<ReactECharts | null>(null)

  const mergedOption = deepMerge(DARK_THEME_DEFAULTS as Record<string, unknown>, option)

  const handleExport = useCallback(() => {
    const instance = chartRef.current?.getEchartsInstance()
    if (!instance) return
    const url = instance.getDataURL({ type: 'png', pixelRatio: 2, backgroundColor: '#0f172a' })
    const link = document.createElement('a')
    link.href = url
    link.download = `${title ?? 'chart'}.png`
    link.click()
  }, [title])

  return (
    <div className={`relative group ${className}`}>
      {title && (
        <div className="flex items-center justify-between mb-2">
          <h3 className="text-sm font-medium text-slate-300">{title}</h3>
          <button
            onClick={handleExport}
            className="opacity-0 group-hover:opacity-100 transition-opacity text-xs text-slate-400 hover:text-slate-200 flex items-center gap-1 px-2 py-1 rounded bg-slate-700 hover:bg-slate-600"
          >
            <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
            </svg>
            PNG
          </button>
        </div>
      )}
      {!title && (
        <button
          onClick={handleExport}
          className="absolute top-2 right-2 z-10 opacity-0 group-hover:opacity-100 transition-opacity text-xs text-slate-400 hover:text-slate-200 flex items-center gap-1 px-2 py-1 rounded bg-slate-800/80 hover:bg-slate-700"
        >
          <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
              d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
          </svg>
          PNG
        </button>
      )}
      <ChartErrorBoundary>
        <React.Suspense fallback={<SkeletonChart height={height} />}>
          <ReactECharts
            ref={chartRef}
            option={mergedOption as EChartsOption}
            style={{ height, width: '100%' }}
            opts={{ renderer: 'canvas' }}
            notMerge={true}
            lazyUpdate={false}
          />
        </React.Suspense>
      </ChartErrorBoundary>
    </div>
  )
}
