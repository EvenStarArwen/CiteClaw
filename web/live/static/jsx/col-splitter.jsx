/* eslint-disable */
// Vertical column splitters for the app frame. Drag to resize the left
// sidebar (seeds / progress) or right sidebar (config / accepted) widths.
function ColSplitter({ side, value, onChange, min = 200, max = 520 }) {
  const [dragging, setDragging] = React.useState(false);
  const onMouseDown = (e) => {
    e.preventDefault();
    setDragging(true);
    const startX = e.clientX;
    const startVal = value;
    const onMove = (ev) => {
      const dx = ev.clientX - startX;
      const delta = side === "left" ? dx : -dx;
      const next = Math.max(min, Math.min(max, startVal + delta));
      onChange(next);
    };
    const onUp = () => {
      setDragging(false);
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
    };
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
  };
  return (
    <div
      className={"col-splitter col-splitter-" + side + (dragging ? " is-dragging" : "")}
      onMouseDown={onMouseDown}
      role="separator"
      aria-orientation="vertical"
      title="Drag to resize column"
    />
  );
}
Object.assign(window, { ColSplitter });
