import {
  Group,
  Panel,
  Separator,
} from "react-resizable-panels"
import { cn } from "../lib/utils"

export function ResizablePanelGroup({
  className,
  ...props
}: React.ComponentProps<typeof Group>) {
  return (
    <Group
      className={cn("flex h-full w-full", className)}
      {...props}
    />
  )
}

export function ResizablePanel({
  className,
  ...props
}: React.ComponentProps<typeof Panel>) {
  return <Panel className={cn("", className)} {...props} />
}

export function ResizableHandle({
  className,
  ...props
}: React.ComponentProps<typeof Separator> & { className?: string }) {
  return (
    <Separator
      className={cn(
        "relative flex w-px items-center justify-center bg-gray-700 after:absolute after:inset-y-0 after:-left-1 after:-right-1 hover:bg-blue-500 transition-colors",
        className
      )}
      {...props}
    />
  )
}
