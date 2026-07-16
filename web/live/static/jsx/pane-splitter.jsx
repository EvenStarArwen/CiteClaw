/* eslint-disable */
// Draggable horizontal splitter used in Build mode's main column to let the
// user re-allocate space between the pipeline canvas (top) and the block-config
// panel (bottom). Lives as its own grid row so `.main.build` becomes a 3-row
// grid: [top pane] [6px splitter] [bottom pane].

function PaneSplitter({ value, onChange }) {
  const [dragging, setDragging] = React.useState(false);

  const onMouseDown = (e) => {
    e.preventDefault();
    setDragging(true);
    const main = e.currentTarget.parentElement;
    if (!main) return;
    const rect = main.getBoundingClientRect();
    const move = (ev) => {
      const y = ev.clientY - rect.top;
      const frac = Math.max(0.15, Math.min(0.85, y / rect.height));
      onChange(Number(frac.toFixed(3)));
    };
    const up = () => {
      setDragging(false);
      window.removeEventListener("mousemove", move);
      window.removeEventListener("mouseup", up);
      document.body.style.cursor = "";
      document.body.style.userSelect = "";
    };
    window.addEventListener("mousemove", move);
    window.addEventListener("mouseup", up);
    document.body.style.cursor = "row-resize";
    document.body.style.userSelect = "none";
  };

  return (
    <div
      className={"pane-splitter" + (dragging ? " is-dragging" : "")}
      onMouseDown={onMouseDown}
      role="separator"
      aria-orientation="horizontal"
      title="Drag to resize"
    >
      <span className="pane-splitter-grip" />
    </div>
  );
}

Object.assign(window, { PaneSplitter });
