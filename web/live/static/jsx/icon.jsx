/* eslint-disable */
// Lucide icon wrapper — renders a named icon at any size.
function Icon({ name, size = 14, className = "", style = {} }) {
  const { useEffect, useRef } = React;
  const ref = useRef(null);
  useEffect(() => {
    if (!window.lucide || !ref.current) return;
    ref.current.innerHTML = "";
    const i = document.createElement("i");
    i.setAttribute("data-lucide", name);
    ref.current.appendChild(i);
    window.lucide.createIcons({
      attrs: { width: size, height: size, "stroke-width": 1.75 },
    });
  }, [name, size]);
  return (
    <span
      ref={ref}
      className={"lc " + className}
      style={{ width: size, height: size, ...style }}
    />
  );
}

Object.assign(window, { Icon });
