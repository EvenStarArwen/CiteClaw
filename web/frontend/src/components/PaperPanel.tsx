import { useAppStore } from "../lib/store"
import { useQuery } from "@tanstack/react-query"
import { SOURCE_COLORS } from "../hooks/useSigmaGraph"

interface PaperData {
  paperId?: string
  title?: string
  abstract?: string
  venue?: string
  year?: number
  authors?: Array<{ authorId?: string; name?: string }>
  citationCount?: number
  referenceCount?: number
  influentialCitationCount?: number
  fieldsOfStudy?: string[]
  publicationTypes?: string[]
  externalIds?: Record<string, string>
  source?: string
  openAccessPdf?: { url?: string }
}

async function fetchPaper(paperId: string): Promise<PaperData> {
  const res = await fetch(`/api/papers/${encodeURIComponent(paperId)}`)
  if (!res.ok) throw new Error(`Paper not found: ${paperId}`)
  return res.json()
}

function SourceBadge({ source }: { source: string }) {
  const color = SOURCE_COLORS[source] ?? "#9ca3af"
  return (
    <span
      className="inline-block px-2 py-0.5 rounded text-xs font-medium"
      style={{ backgroundColor: color + "22", color, border: `1px solid ${color}44` }}
    >
      {source}
    </span>
  )
}

function MetricRow({ label, value }: { label: string; value: string | number | undefined | null }) {
  if (value == null) return null
  return (
    <div className="flex justify-between text-sm py-1">
      <span className="text-gray-400">{label}</span>
      <span className="text-gray-200 font-mono">{value.toLocaleString()}</span>
    </div>
  )
}

export function PaperPanel() {
  const selectedPaperId = useAppStore((s) => s.selectedPaperId)

  const { data: paper, isLoading, error } = useQuery<PaperData, Error>({
    queryKey: ["paper", selectedPaperId],
    queryFn: () => fetchPaper(selectedPaperId!),
    enabled: !!selectedPaperId,
    staleTime: 5 * 60 * 1000,
  })

  if (!selectedPaperId) {
    return (
      <p className="text-sm text-gray-500">
        Select a node in the graph to view paper details.
      </p>
    )
  }

  if (isLoading) {
    return <p className="text-sm text-gray-400 animate-pulse">Loading paper...</p>
  }

  if (error || !paper) {
    return (
      <div>
        <p className="text-sm text-red-400">
          {error?.message ?? "Paper data unavailable."}
        </p>
        <p className="text-xs text-gray-500 mt-1 font-mono break-all">{selectedPaperId}</p>
      </div>
    )
  }

  const s2Url = paper.paperId
    ? `https://www.semanticscholar.org/paper/${paper.paperId}`
    : null

  return (
    <div className="space-y-4">
      {/* Title */}
      <h3 className="text-base font-semibold text-gray-100 leading-snug">
        {paper.title ?? "Untitled"}
      </h3>

      {/* Source + Year + Venue */}
      <div className="flex flex-wrap items-center gap-2 text-sm">
        {paper.source && <SourceBadge source={paper.source} />}
        {paper.year && <span className="text-gray-400">{paper.year}</span>}
        {paper.venue && (
          <span className="text-gray-400 truncate max-w-[180px]" title={paper.venue}>
            {paper.venue}
          </span>
        )}
      </div>

      {/* Authors */}
      {paper.authors && paper.authors.length > 0 && (
        <div className="flex flex-wrap gap-1.5">
          {paper.authors.map((a, i) => (
            <span
              key={a.authorId ?? i}
              className="inline-block px-2 py-0.5 bg-gray-800 rounded text-xs text-gray-300 hover:bg-gray-700 transition-colors cursor-default"
              title={a.authorId ? `S2 Author: ${a.authorId}` : undefined}
            >
              {a.name ?? "Unknown"}
            </span>
          ))}
        </div>
      )}

      {/* Abstract */}
      {paper.abstract && (
        <div>
          <h4 className="text-xs font-semibold text-gray-400 uppercase tracking-wide mb-1">
            Abstract
          </h4>
          <p className="text-sm text-gray-300 leading-relaxed max-h-48 overflow-y-auto">
            {paper.abstract}
          </p>
        </div>
      )}

      {/* Citation Metrics */}
      <div className="border-t border-gray-800 pt-3">
        <h4 className="text-xs font-semibold text-gray-400 uppercase tracking-wide mb-1">
          Metrics
        </h4>
        <MetricRow label="Citations" value={paper.citationCount} />
        <MetricRow label="Influential citations" value={paper.influentialCitationCount} />
        <MetricRow label="References" value={paper.referenceCount} />
      </div>

      {/* Fields of Study */}
      {paper.fieldsOfStudy && paper.fieldsOfStudy.length > 0 && (
        <div>
          <h4 className="text-xs font-semibold text-gray-400 uppercase tracking-wide mb-1">
            Fields
          </h4>
          <div className="flex flex-wrap gap-1">
            {paper.fieldsOfStudy.map((f) => (
              <span key={f} className="px-1.5 py-0.5 bg-gray-800 rounded text-xs text-gray-400">
                {f}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Links */}
      <div className="border-t border-gray-800 pt-3 flex flex-col gap-1.5">
        {s2Url && (
          <a
            href={s2Url}
            target="_blank"
            rel="noopener noreferrer"
            className="text-sm text-indigo-400 hover:text-indigo-300 hover:underline"
          >
            Open on Semantic Scholar
          </a>
        )}
        {paper.openAccessPdf?.url && (
          <a
            href={paper.openAccessPdf.url}
            target="_blank"
            rel="noopener noreferrer"
            className="text-sm text-emerald-400 hover:text-emerald-300 hover:underline"
          >
            Open Access PDF
          </a>
        )}
        {paper.externalIds?.DOI && (
          <a
            href={`https://doi.org/${paper.externalIds.DOI}`}
            target="_blank"
            rel="noopener noreferrer"
            className="text-sm text-gray-400 hover:text-gray-300 hover:underline"
          >
            DOI: {paper.externalIds.DOI}
          </a>
        )}
      </div>

      {/* Paper ID */}
      <p className="text-xs text-gray-600 font-mono break-all">{selectedPaperId}</p>
    </div>
  )
}
