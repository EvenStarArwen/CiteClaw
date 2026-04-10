import { useParams } from "react-router-dom"

export function ConfigView() {
  const { name } = useParams()
  return (
    <div className="text-center">
      <h1 className="text-2xl font-bold tracking-tight mb-2">Config: {name}</h1>
      <p className="text-sm text-gray-500">
        Pipeline builder will render here.
      </p>
    </div>
  )
}
