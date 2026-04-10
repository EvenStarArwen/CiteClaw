import { useParams } from "react-router-dom"
import { PipelineBuilder } from "../components/PipelineBuilder"

export function ConfigView() {
  const { name } = useParams()
  return <PipelineBuilder configName={name} />
}
