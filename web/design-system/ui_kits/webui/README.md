# CiteClaw WebUI — UI kit

Full-screen recreations of the CiteClaw local WebUI, built on the CiteClaw Design System tokens + components. One product, three modes:

- **Build** — compose the snowball screening pipeline. Left: seed search. Center: the pipeline canvas (parallel rounds of forward/backward screeners + diversified reranks). Right: the selected step's parameters and filter pipeline (click a filter to open its config page). `index.html` renders this screen.
- **Run** — watch the pipeline execute live. Left: step progress + current activity + log. Center: the growing citation network over a dot-grid, with a metrics dashboard below (Overview / Rejections / Cost). Right: the accepted/rejected paper stream.
- **Explore** — the citation network as a full page. Left: sortable/filterable paper list. Center: the graph with a Citation↔Authors switch and a graph-settings panel. Right: selected-paper details.

## Fully interactive prototype
A clickable, stateful version of all three modes (mode switching, step selection, filter config pages, run step detail, dashboard tabs, Explore graph-settings) lives at the project root: **`CiteClaw Prototype.dc.html`**. Static option explorations: **`CiteClaw Refined System.dc.html`**.

## Composed from
`components/app` (PaperCard, PipelineBlock, FilterRow) and `components/core` (Button, Badge, SegmentedToggle, PanelHeader, MetricStat, ProgressBar), on `styles.css` tokens. Graphs are representative static SVG (the product uses sigma.js/WebGL).
