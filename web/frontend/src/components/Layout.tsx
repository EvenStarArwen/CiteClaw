import { Outlet } from "react-router-dom"
import { TopBar } from "./TopBar"
import {
  ResizablePanelGroup,
  ResizablePanel,
  ResizableHandle,
} from "./ResizablePanel"

export function Layout() {
  return (
    <div className="flex flex-col h-screen bg-gray-950 text-gray-100">
      <TopBar />
      <ResizablePanelGroup orientation="horizontal" className="flex-1">
        <ResizablePanel defaultSize={25} minSize={15} id="paper-detail">
          <div className="h-full overflow-y-auto p-4">
            <h2 className="text-sm font-semibold text-gray-400 uppercase tracking-wide mb-3">
              Paper Detail
            </h2>
            <p className="text-sm text-gray-500">
              Select a node in the graph to view paper details.
            </p>
          </div>
        </ResizablePanel>
        <ResizableHandle />
        <ResizablePanel defaultSize={50} minSize={30} id="graph">
          <div className="h-full flex items-center justify-center">
            <Outlet />
          </div>
        </ResizablePanel>
        <ResizableHandle />
        <ResizablePanel defaultSize={25} minSize={15} id="controls">
          <div className="h-full overflow-y-auto p-4">
            <h2 className="text-sm font-semibold text-gray-400 uppercase tracking-wide mb-3">
              Controls
            </h2>
            <p className="text-sm text-gray-500">
              Configure and run pipelines here.
            </p>
          </div>
        </ResizablePanel>
      </ResizablePanelGroup>
    </div>
  )
}
