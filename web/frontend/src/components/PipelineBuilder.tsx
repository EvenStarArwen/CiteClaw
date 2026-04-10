/**
 * React Flow pipeline builder — drag step blocks onto a canvas, configure them,
 * and save/load as CiteClaw YAML configs.
 */

import { useCallback, useState, useMemo, type MouseEvent } from "react"
import {
  ReactFlow,
  Background,
  Controls,
  MiniMap,
  addEdge,
  useNodesState,
  useEdgesState,
  type Connection,
  type Edge,
  type Node,
  type NodeProps,
  Handle,
  Position,
} from "@xyflow/react"
import "@xyflow/react/dist/style.css"

import { STEP_TYPES, STEP_TYPE_MAP, CATEGORIES, type StepTypeDef, type StepField } from "../lib/pipelineSchema"
import { flowToYaml, yamlToFlow, configJsonToYaml } from "../lib/yamlBridge"

/* ---- Custom Step Node ---- */

interface StepNodeData {
  stepType: string
  label: string
  config: Record<string, unknown>
  [key: string]: unknown
}

function StepNode({ data, selected }: NodeProps<Node<StepNodeData>>) {
  const typeDef = STEP_TYPE_MAP[data.stepType]
  const color = typeDef?.color ?? "#6b7280"

  return (
    <div
      className={`rounded-lg border-2 shadow-md px-4 py-2 min-w-[180px] bg-gray-900 ${
        selected ? "ring-2 ring-blue-400" : ""
      }`}
      style={{ borderColor: color }}
    >
      <Handle type="target" position={Position.Top} className="!bg-gray-500 !w-3 !h-3" />
      <div className="flex items-center gap-2">
        <div className="w-3 h-3 rounded-full flex-shrink-0" style={{ backgroundColor: color }} />
        <span className="text-sm font-semibold text-gray-100 truncate">
          {data.label}
        </span>
      </div>
      {typeDef?.description && (
        <p className="text-xs text-gray-500 mt-1 line-clamp-2">{typeDef.description}</p>
      )}
      <Handle type="source" position={Position.Bottom} className="!bg-gray-500 !w-3 !h-3" />
    </div>
  )
}

const nodeTypes = { stepNode: StepNode }

/* ---- Field Editor ---- */

function FieldEditor({
  field,
  value,
  onChange,
}: {
  field: StepField
  value: unknown
  onChange: (name: string, val: unknown) => void
}) {
  const id = `field-${field.name}`

  switch (field.type) {
    case "boolean":
      return (
        <label className="flex items-center gap-2 text-sm">
          <input
            type="checkbox"
            checked={Boolean(value ?? field.default)}
            onChange={(e) => onChange(field.name, e.target.checked)}
            className="accent-blue-500"
          />
          {field.label}
        </label>
      )
    case "number":
      return (
        <label className="block text-sm">
          <span className="text-gray-400">{field.label}</span>
          <input
            id={id}
            type="number"
            value={value !== undefined ? String(value) : String(field.default ?? "")}
            onChange={(e) => onChange(field.name, e.target.value ? Number(e.target.value) : undefined)}
            placeholder={field.placeholder}
            className="mt-1 block w-full rounded bg-gray-800 border border-gray-700 px-2 py-1 text-sm text-gray-200"
          />
        </label>
      )
    case "select":
      return (
        <label className="block text-sm">
          <span className="text-gray-400">{field.label}</span>
          <select
            id={id}
            value={String(value ?? field.default ?? "")}
            onChange={(e) => onChange(field.name, e.target.value)}
            className="mt-1 block w-full rounded bg-gray-800 border border-gray-700 px-2 py-1 text-sm text-gray-200"
          >
            {field.options?.map((o) => (
              <option key={o} value={o}>
                {o}
              </option>
            ))}
          </select>
        </label>
      )
    case "json":
      return (
        <label className="block text-sm">
          <span className="text-gray-400">{field.label}</span>
          <textarea
            id={id}
            value={typeof value === "string" ? value : JSON.stringify(value ?? field.default ?? {}, null, 2)}
            onChange={(e) => onChange(field.name, e.target.value)}
            placeholder={field.placeholder}
            rows={3}
            className="mt-1 block w-full rounded bg-gray-800 border border-gray-700 px-2 py-1 text-sm text-gray-200 font-mono"
          />
        </label>
      )
    default: // string
      return (
        <label className="block text-sm">
          <span className="text-gray-400">
            {field.label}
            {field.required && <span className="text-red-400 ml-1">*</span>}
          </span>
          <input
            id={id}
            type="text"
            value={String(value ?? "")}
            onChange={(e) => onChange(field.name, e.target.value)}
            placeholder={field.placeholder}
            className="mt-1 block w-full rounded bg-gray-800 border border-gray-700 px-2 py-1 text-sm text-gray-200"
          />
        </label>
      )
  }
}

/* ---- Settings Sidebar ---- */

function SettingsSidebar({
  node,
  onUpdate,
  onDelete,
  onClose,
}: {
  node: Node<StepNodeData>
  onUpdate: (id: string, config: Record<string, unknown>) => void
  onDelete: (id: string) => void
  onClose: () => void
}) {
  const typeDef = STEP_TYPE_MAP[node.data.stepType]
  if (!typeDef) return null

  const handleChange = (name: string, val: unknown) => {
    onUpdate(node.id, { ...node.data.config, [name]: val })
  }

  return (
    <div className="w-72 border-l border-gray-800 bg-gray-950 p-4 overflow-y-auto flex flex-col gap-3">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-bold text-gray-200">{typeDef.label}</h3>
        <button
          onClick={onClose}
          className="text-gray-500 hover:text-gray-300 text-lg leading-none"
        >
          &times;
        </button>
      </div>
      <p className="text-xs text-gray-500">{typeDef.description}</p>
      <hr className="border-gray-800" />
      {typeDef.fields.length === 0 ? (
        <p className="text-xs text-gray-600 italic">No configurable fields.</p>
      ) : (
        typeDef.fields.map((f) => (
          <FieldEditor
            key={f.name}
            field={f}
            value={node.data.config[f.name]}
            onChange={handleChange}
          />
        ))
      )}
      <hr className="border-gray-800 mt-auto" />
      <button
        onClick={() => onDelete(node.id)}
        className="text-xs text-red-400 hover:text-red-300 self-start"
      >
        Remove step
      </button>
    </div>
  )
}

/* ---- Block Library (left drawer) ---- */

function BlockLibrary({ onAdd }: { onAdd: (typeDef: StepTypeDef) => void }) {
  return (
    <div className="w-56 border-r border-gray-800 bg-gray-950 p-3 overflow-y-auto">
      <h3 className="text-xs font-bold text-gray-400 uppercase tracking-wider mb-3">
        Block Library
      </h3>
      {CATEGORIES.map((cat) => {
        const steps = STEP_TYPES.filter((s) => s.category === cat.key)
        if (steps.length === 0) return null
        return (
          <div key={cat.key} className="mb-3">
            <p className="text-xs font-semibold text-gray-500 mb-1">{cat.label}</p>
            {steps.map((s) => (
              <button
                key={s.type}
                onClick={() => onAdd(s)}
                className="flex items-center gap-2 w-full text-left px-2 py-1.5 rounded text-sm
                           hover:bg-gray-800 text-gray-300 transition-colors"
              >
                <div
                  className="w-2.5 h-2.5 rounded-full flex-shrink-0"
                  style={{ backgroundColor: s.color }}
                />
                {s.label}
              </button>
            ))}
          </div>
        )
      })}
    </div>
  )
}

/* ---- Toolbar ---- */

function Toolbar({
  configName,
  onConfigNameChange,
  onSave,
  onLoad,
  saving,
}: {
  configName: string
  onConfigNameChange: (v: string) => void
  onSave: () => void
  onLoad: () => void
  saving: boolean
}) {
  return (
    <div className="flex items-center gap-3 px-3 py-2 border-b border-gray-800 bg-gray-950">
      <label className="flex items-center gap-2 text-sm text-gray-400">
        Config:
        <input
          type="text"
          value={configName}
          onChange={(e) => onConfigNameChange(e.target.value)}
          className="rounded bg-gray-800 border border-gray-700 px-2 py-1 text-sm text-gray-200 w-52"
          placeholder="config_my_pipeline.yaml"
        />
      </label>
      <button
        onClick={onLoad}
        className="px-3 py-1 text-sm rounded bg-gray-800 hover:bg-gray-700 text-gray-200 border border-gray-700"
      >
        Load
      </button>
      <button
        onClick={onSave}
        disabled={saving}
        className="px-3 py-1 text-sm rounded bg-blue-600 hover:bg-blue-500 text-white disabled:opacity-50"
      >
        {saving ? "Saving..." : "Save"}
      </button>
    </div>
  )
}

/* ---- Main PipelineBuilder ---- */

export function PipelineBuilder({ configName: initialName }: { configName?: string }) {
  const [nodes, setNodes, onNodesChange] = useNodesState<Node<StepNodeData>>([])
  const [edges, setEdges, onEdgesChange] = useEdgesState<Edge>([])
  const [selectedNode, setSelectedNode] = useState<string | null>(null)
  const [configName, setConfigName] = useState(initialName ?? "config_new.yaml")
  const [saving, setSaving] = useState(false)
  const [globalConfig, setGlobalConfig] = useState<Record<string, unknown>>({})

  const onConnect = useCallback(
    (conn: Connection) => setEdges((eds) => addEdge({ ...conn, type: "smoothstep" }, eds)),
    [setEdges],
  )

  const handleNodeClick = useCallback((_: MouseEvent, node: Node) => {
    setSelectedNode(node.id)
  }, [])

  const handlePaneClick = useCallback(() => {
    setSelectedNode(null)
  }, [])

  /* Add a step from the library */
  const handleAddStep = useCallback(
    (typeDef: StepTypeDef) => {
      const id = `step-${Date.now()}`
      const maxY = nodes.reduce((m, n) => Math.max(m, n.position.y), -80)
      const newNode: Node<StepNodeData> = {
        id,
        type: "stepNode",
        position: { x: 250, y: maxY + 120 },
        data: {
          stepType: typeDef.type,
          label: typeDef.label,
          config: {},
        },
      }

      setNodes((nds) => {
        const updated = [...nds, newNode]
        // Auto-connect to the previous last node
        if (nds.length > 0) {
          const lastNode = [...nds].sort((a, b) => b.position.y - a.position.y)[0]
          setEdges((eds) =>
            addEdge(
              { id: `e-${lastNode.id}-${id}`, source: lastNode.id, target: id, type: "smoothstep" },
              eds,
            ),
          )
        }
        return updated
      })
      setSelectedNode(id)
    },
    [nodes, setNodes, setEdges],
  )

  /* Update node config from settings sidebar */
  const handleUpdateConfig = useCallback(
    (nodeId: string, config: Record<string, unknown>) => {
      setNodes((nds) =>
        nds.map((n) =>
          n.id === nodeId ? { ...n, data: { ...n.data, config } } : n,
        ),
      )
    },
    [setNodes],
  )

  /* Delete a node */
  const handleDeleteNode = useCallback(
    (nodeId: string) => {
      setNodes((nds) => nds.filter((n) => n.id !== nodeId))
      setEdges((eds) => eds.filter((e) => e.source !== nodeId && e.target !== nodeId))
      setSelectedNode(null)
    },
    [setNodes, setEdges],
  )

  /* Save → POST /api/configs/{name} */
  const handleSave = useCallback(async () => {
    setSaving(true)
    try {
      const yamlStr = flowToYaml(nodes, edges, globalConfig)
      const { load } = await import("js-yaml")
      const parsed = load(yamlStr) as Record<string, unknown>
      const resp = await fetch(`/api/configs/${configName}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(parsed),
      })
      if (!resp.ok) throw new Error(await resp.text())
    } catch (err) {
      console.error("Save failed:", err)
    } finally {
      setSaving(false)
    }
  }, [nodes, edges, configName, globalConfig])

  /* Load ← GET /api/configs/{name} */
  const handleLoad = useCallback(async () => {
    try {
      const resp = await fetch(`/api/configs/${configName}`)
      if (!resp.ok) throw new Error(await resp.text())
      const data = await resp.json()
      const yamlStr = configJsonToYaml(data)
      const { nodes: newNodes, edges: newEdges, globalConfig: gc } = yamlToFlow(yamlStr)
      setNodes(newNodes)
      setEdges(newEdges)
      setGlobalConfig(gc)
      setSelectedNode(null)
    } catch (err) {
      console.error("Load failed:", err)
    }
  }, [configName, setNodes, setEdges])

  const activeNode = useMemo(
    () => nodes.find((n) => n.id === selectedNode) ?? null,
    [nodes, selectedNode],
  )

  return (
    <div className="flex flex-col h-full w-full">
      <Toolbar
        configName={configName}
        onConfigNameChange={setConfigName}
        onSave={handleSave}
        onLoad={handleLoad}
        saving={saving}
      />
      <div className="flex flex-1 min-h-0">
        <BlockLibrary onAdd={handleAddStep} />
        <div className="flex-1 relative">
          <ReactFlow
            nodes={nodes}
            edges={edges}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            onConnect={onConnect}
            onNodeClick={handleNodeClick}
            onPaneClick={handlePaneClick}
            nodeTypes={nodeTypes}
            fitView
            colorMode="dark"
          >
            <Background gap={20} size={1} />
            <Controls />
            <MiniMap
              nodeStrokeWidth={3}
              pannable
              zoomable
              className="!bg-gray-900 !border-gray-800"
            />
          </ReactFlow>
        </div>
        {activeNode && (
          <SettingsSidebar
            node={activeNode}
            onUpdate={handleUpdateConfig}
            onDelete={handleDeleteNode}
            onClose={() => setSelectedNode(null)}
          />
        )}
      </div>
    </div>
  )
}
