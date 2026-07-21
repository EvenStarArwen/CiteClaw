SegmentedToggle — the mode switch (Build/Run/Explore) and dashboard tabs. Active segment raises to a cream card with green text.

```jsx
<SegmentedToggle value={mode} onChange={setMode}
  options={[{value:"build",label:"Build"},{value:"run",label:"Run"},{value:"explore",label:"Explore"}]} />
```
