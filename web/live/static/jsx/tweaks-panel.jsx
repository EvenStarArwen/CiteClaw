/* eslint-disable */
// Tweaks panel — toggled by the host toolbar. Exposes a few visual variables,
// notably the pipeline block style (card / chip / ghost / tag).

function TweaksPanel({ visible, tweaks, setTweak }) {
  if (!visible) return null;

  const StyleOpts = ({ id, options, cols = 2 }) => (
    <div className="tweak-opts" style={{ gridTemplateColumns: `repeat(${cols}, 1fr)` }}>
      {options.map(o => (
        <button
          key={String(o.value)}
          className={"tweak-opt" + (tweaks[id] === o.value ? " on" : "")}
          onClick={() => setTweak(id, o.value)}
        >
          {o.label}
        </button>
      ))}
    </div>
  );

  return (
    <div className="tweaks">
      <div className="tweaks-head">
        <span className="tweaks-title">Tweaks</span>
        <Icon name="sliders-horizontal" size={13} style={{ color: "var(--cc-ink-3)" }} />
      </div>
      <div className="tweaks-body">
        <div className="tweak-row">
          <div className="tweak-label">Mode</div>
          <StyleOpts id="mode" options={[
            { value: "build", label: "Build" },
            { value: "run",   label: "Run" },
          ]} />
        </div>
        <div className="tweak-row">
          <div className="tweak-label">Status bar</div>
          <StyleOpts id="showBottomBar" options={[
            { value: true,  label: "Show" },
            { value: false, label: "Hide" },
          ]} />
        </div>

        <div className="tweak-row" style={{ marginTop: 6, paddingTop: 10, borderTop: "1px dashed var(--cc-rule)" }}>
          <div className="tweak-label" style={{ fontSize: 10, letterSpacing: "0.1em", textTransform: "uppercase", color: "var(--cc-ink-3)" }}>Color scheme</div>
        </div>
        <div className="tweak-row">
          <div className="tweak-label">Canvas</div>
          <StyleOpts id="monoCanvas" cols={3} options={[
            { value: "paper", label: "Cream"  },
            { value: "snow",  label: "Snow"   },
            { value: "fog",   label: "Fog"    },
          ]} />
        </div>
        <div className="tweak-row">
          <div className="tweak-label">Accent</div>
          <StyleOpts id="monoAccent" cols={3} options={[
            { value: "ink",       label: "Ink"        },
            { value: "anthropic", label: "Anthropic"  },
            { value: "indigo",    label: "Indigo"     },
            { value: "cobalt",    label: "Cobalt"     },
            { value: "crimson",   label: "Crimson"    },
          ]} />
        </div>

        <div className="tweak-row" style={{ marginTop: 6, paddingTop: 10, borderTop: "1px dashed var(--cc-rule)" }}>
          <div className="tweak-label" style={{ fontSize: 10, letterSpacing: "0.1em", textTransform: "uppercase", color: "var(--cc-ink-3)" }}>Seeds</div>
        </div>
        <div className="tweak-row">
          <div className="tweak-label">Seed selected</div>
          <StyleOpts id="monoSeedFill" cols={2} options={[
            { value: "orange-card",      label: "Solid orange"   },
            { value: "orange-wash",      label: "Orange wash"    },
            { value: "orange-bar",       label: "Orange bar"     },
            { value: "orange-ring",      label: "Orange ring"    },
            { value: "orange-underline", label: "Orange rule"    },
            { value: "orange-chip",      label: "Orange chip"    },
          ]} />
        </div>
      </div>
    </div>
  );
}

Object.assign(window, { TweaksPanel });
