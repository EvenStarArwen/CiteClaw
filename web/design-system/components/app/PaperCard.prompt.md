PaperCard — the single card for every paper/seed list across the app. The 18px lead column is fixed, so seeded and non-seeded titles share one left edge.

```jsx
<PaperCard lead="seed" title="Autonomous chemical research…" meta="Boiko et al. · 2023 · Nature" trailing="1,094 cites" />
<PaperCard lead="star" title="ChemCrow…" meta="Bran et al. · 2024 · Nat. Mach. Intel." trailing="480 cites" />
<PaperCard trailingKind="score" trailing="0.94" title="…" meta="…" selected />
```

lead: none / seed (green dot) / star (gold, "star to accept") / reject (clay ✕). trailingKind: cites (plain mono) or score (green chip).
