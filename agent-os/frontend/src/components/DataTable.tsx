import React, { useState, useMemo } from 'react'
import {
  useReactTable,
  getCoreRowModel,
  getSortedRowModel,
  getFilteredRowModel,
  getPaginationRowModel,
  flexRender,
  type ColumnDef,
  type SortingState,
  type ColumnFiltersState,
} from '@tanstack/react-table'

interface DataTableProps<T extends object> {
  data: T[]
  columns: ColumnDef<T, unknown>[]
  loading?: boolean
  pageSize?: number
  globalFilter?: boolean
  className?: string
}

function SkeletonRow({ cols }: { cols: number }) {
  return (
    <tr>
      {Array.from({ length: cols }).map((_, i) => (
        <td key={i} className="px-4 py-3">
          <div className="h-4 bg-slate-700 rounded animate-pulse" style={{ width: `${60 + Math.random() * 30}%` }} />
        </td>
      ))}
    </tr>
  )
}

function exportToCSV<T extends object>(data: T[], columns: ColumnDef<T, unknown>[]) {
  const headers = columns
    .map(col => {
      if (typeof col.header === 'string') return col.header
      const key = (col as { accessorKey?: string }).accessorKey
      return key ?? ''
    })
    .filter(Boolean)

  const rows = data.map(row =>
    columns.map(col => {
      const key = (col as { accessorKey?: string }).accessorKey
      if (!key) return ''
      const val = (row as Record<string, unknown>)[key]
      if (val === null || val === undefined) return ''
      const str = String(val)
      return str.includes(',') || str.includes('"') || str.includes('\n')
        ? `"${str.replace(/"/g, '""')}"`
        : str
    }).join(',')
  )

  const csv = [headers.join(','), ...rows].join('\n')
  const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' })
  const url = URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = url
  link.download = 'data-export.csv'
  link.click()
  URL.revokeObjectURL(url)
}

export default function DataTable<T extends object>({
  data,
  columns,
  loading = false,
  pageSize = 10,
  globalFilter = true,
  className = '',
}: DataTableProps<T>) {
  const [sorting, setSorting] = useState<SortingState>([])
  const [columnFilters, setColumnFilters] = useState<ColumnFiltersState>([])
  const [globalFilterValue, setGlobalFilterValue] = useState('')
  const [pagination, setPagination] = useState({ pageIndex: 0, pageSize })

  const table = useReactTable({
    data,
    columns,
    state: { sorting, columnFilters, globalFilter: globalFilterValue, pagination },
    onSortingChange: setSorting,
    onColumnFiltersChange: setColumnFilters,
    onGlobalFilterChange: setGlobalFilterValue,
    onPaginationChange: setPagination,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
    getFilteredRowModel: getFilteredRowModel(),
    getPaginationRowModel: getPaginationRowModel(),
  })

  const pageCount = table.getPageCount()
  const currentPage = table.getState().pagination.pageIndex
  const totalRows = table.getFilteredRowModel().rows.length

  return (
    <div className={`flex flex-col gap-3 ${className}`}>
      {/* Toolbar */}
      <div className="flex items-center justify-between gap-3">
        {globalFilter && (
          <div className="relative">
            <svg className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
            </svg>
            <input
              type="text"
              placeholder="Search..."
              value={globalFilterValue}
              onChange={e => setGlobalFilterValue(e.target.value)}
              className="input pl-9 text-sm w-56"
            />
          </div>
        )}
        <div className="flex items-center gap-2 ml-auto">
          <span className="text-xs text-slate-400">
            {totalRows} row{totalRows !== 1 ? 's' : ''}
          </span>
          <button
            onClick={() => exportToCSV(data, columns)}
            className="btn-secondary text-sm flex items-center gap-2"
          >
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
            </svg>
            Export CSV
          </button>
        </div>
      </div>

      {/* Table */}
      <div className="overflow-x-auto rounded-xl border border-slate-700">
        <table className="min-w-full divide-y divide-slate-700">
          <thead className="bg-slate-800/60">
            {table.getHeaderGroups().map(headerGroup => (
              <tr key={headerGroup.id}>
                {headerGroup.headers.map(header => (
                  <th
                    key={header.id}
                    onClick={header.column.getToggleSortingHandler()}
                    className={`px-4 py-3 text-left text-xs font-semibold text-slate-400 uppercase tracking-wider select-none ${header.column.getCanSort() ? 'cursor-pointer hover:text-slate-200' : ''}`}
                  >
                    <div className="flex items-center gap-1.5">
                      {flexRender(header.column.columnDef.header, header.getContext())}
                      {header.column.getCanSort() && (
                        <span className="text-slate-600">
                          {header.column.getIsSorted() === 'asc' ? '↑' :
                           header.column.getIsSorted() === 'desc' ? '↓' : '↕'}
                        </span>
                      )}
                    </div>
                  </th>
                ))}
              </tr>
            ))}
          </thead>
          <tbody className="divide-y divide-slate-700/50 bg-slate-800/20">
            {loading ? (
              Array.from({ length: pageSize }).map((_, i) => (
                <SkeletonRow key={i} cols={columns.length} />
              ))
            ) : table.getRowModel().rows.length === 0 ? (
              <tr>
                <td colSpan={columns.length} className="px-4 py-12 text-center text-slate-500">
                  No data to display
                </td>
              </tr>
            ) : (
              table.getRowModel().rows.map(row => (
                <tr key={row.id} className="hover:bg-slate-700/20 transition-colors">
                  {row.getVisibleCells().map(cell => (
                    <td key={cell.id} className="px-4 py-3 text-sm text-slate-300 whitespace-nowrap">
                      {flexRender(cell.column.columnDef.cell, cell.getContext())}
                    </td>
                  ))}
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>

      {/* Pagination */}
      {pageCount > 1 && (
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <button
              onClick={() => table.firstPage()}
              disabled={!table.getCanPreviousPage()}
              className="btn-secondary text-xs px-2 py-1.5 disabled:opacity-40"
            >
              «
            </button>
            <button
              onClick={() => table.previousPage()}
              disabled={!table.getCanPreviousPage()}
              className="btn-secondary text-xs px-2 py-1.5 disabled:opacity-40"
            >
              ‹
            </button>
            <span className="text-xs text-slate-400">
              Page {currentPage + 1} of {pageCount}
            </span>
            <button
              onClick={() => table.nextPage()}
              disabled={!table.getCanNextPage()}
              className="btn-secondary text-xs px-2 py-1.5 disabled:opacity-40"
            >
              ›
            </button>
            <button
              onClick={() => table.lastPage()}
              disabled={!table.getCanNextPage()}
              className="btn-secondary text-xs px-2 py-1.5 disabled:opacity-40"
            >
              »
            </button>
          </div>
          <select
            value={table.getState().pagination.pageSize}
            onChange={e => table.setPageSize(Number(e.target.value))}
            className="input text-xs py-1 w-24"
          >
            {[10, 20, 50, 100].map(size => (
              <option key={size} value={size}>Show {size}</option>
            ))}
          </select>
        </div>
      )}
    </div>
  )
}
