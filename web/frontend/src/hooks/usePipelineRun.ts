import { useEffect, useRef } from "react"
import { useAppStore } from "../lib/store"

/**
 * WebSocket hook that subscribes to a pipeline run's event stream.
 *
 * Connects to `ws://<host>/api/runs/{runId}/stream` and dispatches
 * incoming events (step_start, step_end, paper_added, shape_table_update)
 * to the Zustand store so Graph and other components react in real time.
 */
export function usePipelineRun(runId: string | null) {
  const wsRef = useRef<WebSocket | null>(null)

  const startRun = useAppStore((s) => s.startRun)
  const resetRun = useAppStore((s) => s.resetRun)
  const stepStart = useAppStore((s) => s.stepStart)
  const stepEnd = useAppStore((s) => s.stepEnd)
  const paperAdded = useAppStore((s) => s.paperAdded)
  const shapeTableUpdate = useAppStore((s) => s.shapeTableUpdate)
  const hitlRequest = useAppStore((s) => s.hitlRequest)

  useEffect(() => {
    if (!runId) {
      resetRun()
      return
    }

    startRun(runId)

    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:"
    const host = window.location.host
    const url = `${protocol}//${host}/api/runs/${runId}/stream`

    const ws = new WebSocket(url)
    wsRef.current = ws

    ws.onmessage = (event) => {
      let msg: { type: string; [key: string]: unknown }
      try {
        msg = JSON.parse(event.data as string)
      } catch {
        return
      }

      switch (msg.type) {
        case "step_start":
          stepStart(
            msg.idx as number,
            msg.name as string,
            msg.description as string,
          )
          break

        case "step_end":
          stepEnd(
            msg.idx as number,
            msg.name as string,
            msg.in_count as number,
            msg.out_count as number,
            msg.delta_collection as number,
            (msg.stats as Record<string, unknown>) ?? {},
          )
          break

        case "paper_added":
          paperAdded(msg.paper_id as string, msg.source as string)
          break

        case "shape_table_update":
          shapeTableUpdate(msg.rendered_shape as string)
          break

        case "hitl_request":
          hitlRequest(
            msg.papers as { paper_id: string; title: string; venue: string; year: number | null; abstract: string }[],
          )
          break
      }
    }

    ws.onclose = () => {
      wsRef.current = null
    }

    return () => {
      ws.close()
      wsRef.current = null
    }
  }, [runId, startRun, resetRun, stepStart, stepEnd, paperAdded, shapeTableUpdate, hitlRequest])
}
