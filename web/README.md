# CiteClaw Web UI

## Stack

- **Frontend**: React 18 + Vite + TypeScript + Tailwind CSS v4 + shadcn/ui (planned) + sigma.js + React Flow
- **Backend**: FastAPI + WebSockets (Python)
- **State**: Zustand (client), TanStack Query (server)

## Quick start

### Backend

```bash
cd web/backend
python3 -m uvicorn main:app --port 9999
# Health check: curl http://localhost:9999/health
```

### Frontend

```bash
cd web/frontend
pnpm install
pnpm dev
# Opens at http://localhost:5173
```
