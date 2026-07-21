PipelineBlock — one step on the Build pipeline canvas ("specimen" card: category · № · name · id · filter count). Selected shows the green ring.

```jsx
<PipelineBlock kind="fwd" num="02" name="Forward screener" id="FWD-02" filters={5} selected />
<PipelineBlock kind="rerank" num="04" name="Diversified rerank" id="RRK-04" />
<PipelineBlock kind="fwd" num="05" name="Forward screener" id="FWD-05" filters={5} sync="⇄ FWD-02" />
```
