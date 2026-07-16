import type { Catalog, CredentialStatus, Credentials, RunSnapshot } from "./types"

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    ...init,
    headers: { "Content-Type": "application/json", ...(init?.headers || {}) },
  })
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`
    try {
      const body = await response.json() as { detail?: string }
      detail = body.detail || detail
    } catch { /* response was not JSON */ }
    throw new Error(detail)
  }
  return response.json() as Promise<T>
}

export const api = {
  catalog: () => request<Catalog>("/api/catalog"),
  credentialStatus: () => request<CredentialStatus>("/api/runs/credentials/status"),
  configs: () => request<Array<{ name: string }>>("/api/configs"),
  config: (name: string) => request<{ name: string; yaml: string }>(`/api/configs/${encodeURIComponent(name)}`),
  validate: (yaml: string) => request<{ valid: boolean; summary: Record<string, unknown> }>("/api/configs/validate/yaml", {
    method: "POST", body: JSON.stringify({ yaml }),
  }),
  save: (name: string, yaml: string) => request<{ status: string; name: string }>(`/api/configs/${encodeURIComponent(name)}`, {
    method: "PUT", body: JSON.stringify({ yaml }),
  }),
  run: (configName: string, configYaml: string, credentials: Credentials) => request<RunSnapshot>("/api/runs", {
    method: "POST", body: JSON.stringify({ config_name: configName, config_yaml: configYaml, credentials }),
  }),
  hitl: (runId: string, labels: Record<string, boolean>, stopRequested = false) => request<{ status: string }>(`/api/runs/${runId}/hitl`, {
    method: "POST", body: JSON.stringify({ labels, stop_requested: stopRequested }),
  }),
}
