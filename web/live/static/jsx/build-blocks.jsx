/* eslint-disable */
// Section E (Build mode) — Foundational building blocks palette.
// Categorized list: Sources / Expanders / Rerankers / Sinks.

function BuildBlocks({ onAdd }) {
  const [q, setQ] = React.useState("");
  const cats = window.BLOCK_CATALOG.map(g => ({
    ...g,
    items: g.items.filter(it =>
      (it.name + " " + it.hint).toLowerCase().includes(q.toLowerCase())
    ),
  })).filter(g => g.items.length);

  return (
    <aside className="panel panel-right">
      <div className="ph">
        <span className="ph-title">Building blocks</span>
        <span className="ph-count">6 total</span>
      </div>
      <div className="searchbox">
        <Icon name="search" size={12} />
        <input
          value={q}
          onChange={e => setQ(e.target.value)}
          placeholder="Search blocks…"
        />
      </div>

      <div className="pb-scroll">
        <div className="blocks-list">
          {cats.map(cat => (
            <React.Fragment key={cat.cat}>
              <div className="blocks-cat">
                <span>{cat.cat}</span>
                <span className="blocks-cat-n">{cat.items.length}</span>
              </div>
              {cat.items.map(it => (
                <div
                  key={it.kind}
                  className="block-item"
                  draggable
                  title="Drag onto the canvas to add"
                  onClick={() => onAdd && onAdd(it)}
                >
                  <span className="block-icon">
                    <Icon name={it.icon} size={12} />
                  </span>
                  <div className="block-body">
                    <span className="block-name">{it.name}</span>
                    <span className="block-hint">{it.hint}</span>
                  </div>
                  <span className="block-add">
                    <Icon name="plus" size={13} />
                  </span>
                </div>
              ))}
            </React.Fragment>
          ))}
        </div>
      </div>

      <div className="blocks-foot">
        {/* foot intentionally empty — panel is self-explanatory */}
      </div>
    </aside>
  );
}

Object.assign(window, { BuildBlocks });
