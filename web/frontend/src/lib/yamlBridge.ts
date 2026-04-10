/**
 * Convert between React Flow node/edge representations and CiteClaw YAML pipeline configs.
 */

import yaml from "js-yaml"
import type { Node, Edge } from "@xyflow/react"
import { STEP_TYPE_MAP } from "./pipelineSchema"

/* ---- Flow → YAML ---- */

interface StepNodeData {
  stepType: string
  label: string
  config: Record<string, unknown>
  [key: string]: unknown
}

/**
 * Convert a React Flow graph into a CiteClaw YAML config string.
 * Nodes are sorted top-to-bottom by Y position to determine pipeline order.
 */
export function flowToYaml(
  nodes: Node<StepNodeData>[],
  _edges: Edge[],
  globalConfig: Record<string, unknown>,
): string {
  const sorted = [...nodes].sort((a, b) => (a.position.y ?? 0) - (b.position.y ?? 0))

  const pipeline = sorted.map((node) => {
    const entry: Record<string, unknown> = { step: node.data.stepType }
    const typeDef = STEP_TYPE_MAP[node.data.stepType]
    if (!typeDef) return entry

    for (const field of typeDef.fields) {
      const val = node.data.config[field.name]
      if (val === undefined || val === "" || val === field.default) continue
      if (field.type === "json" && typeof val === "string") {
        try {
          entry[field.name] = JSON.parse(val)
        } catch {
          entry[field.name] = val
        }
      } else {
        entry[field.name] = val
      }
    }
    return entry
  })

  const config: Record<string, unknown> = { ...globalConfig, pipeline }
  return yaml.dump(config, { noRefs: true, sortKeys: false, lineWidth: 120 })
}

/* ---- YAML → Flow ---- */

const NODE_SPACING_Y = 120

/**
 * Parse a CiteClaw YAML config and produce React Flow nodes + edges.
 */
export function yamlToFlow(
  yamlStr: string,
): { nodes: Node<StepNodeData>[]; edges: Edge[]; globalConfig: Record<string, unknown> } {
  const config = yaml.load(yamlStr) as Record<string, unknown> | null
  if (!config) return { nodes: [], edges: [], globalConfig: {} }

  const pipelineRaw = (config.pipeline ?? []) as Record<string, unknown>[]
  const globalConfig: Record<string, unknown> = {}
  for (const [k, v] of Object.entries(config)) {
    if (k !== "pipeline" && k !== "blocks") globalConfig[k] = v
  }

  const nodes: Node<StepNodeData>[] = []
  const edges: Edge[] = []

  pipelineRaw.forEach((stepDict, i) => {
    const stepType = stepDict.step as string
    const typeDef = STEP_TYPE_MAP[stepType]
    const nodeConfig: Record<string, unknown> = {}

    if (typeDef) {
      for (const field of typeDef.fields) {
        const raw = stepDict[field.name]
        if (raw === undefined) continue
        if (field.type === "json" && typeof raw === "object") {
          nodeConfig[field.name] = JSON.stringify(raw)
        } else {
          nodeConfig[field.name] = raw
        }
      }
    }

    const id = `step-${i}`
    nodes.push({
      id,
      type: "stepNode",
      position: { x: 250, y: i * NODE_SPACING_Y + 40 },
      data: {
        stepType,
        label: typeDef?.label ?? stepType,
        config: nodeConfig,
      },
    })

    if (i > 0) {
      edges.push({
        id: `e-${i - 1}-${i}`,
        source: `step-${i - 1}`,
        target: id,
        type: "smoothstep",
      })
    }
  })

  return { nodes, edges, globalConfig }
}

/**
 * Parse raw YAML string from a config object fetched via the API.
 */
export function configJsonToYaml(config: Record<string, unknown>): string {
  return yaml.dump(config, { noRefs: true, sortKeys: false, lineWidth: 120 })
}
