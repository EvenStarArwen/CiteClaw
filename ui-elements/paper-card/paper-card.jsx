/* eslint-disable */
/* ==========================================================================
   PAPER CARD  ·  <PaperCard>  ·  CiteClaw UI element
   --------------------------------------------------------------------------
   One reusable card for every paper list (Build seeds, Run accepted / rejected,
   Explore). Renders the .pcard structure that paper-card.css styles. Behaviour
   (which handler fires) and the lead / trail glyphs stay with the caller — the
   card only owns layout + look — so this component has NO icon dependency and
   drops cleanly into the app's Babel block or the standalone demo.

   PROPS
     title        string           the paper title (clamped to 2 lines)
     venue        string           journal / venue        (own row; "—" if empty)
     authors      string           author byline          (own row; "—" if empty)
     year         number|string    publication year       ┐ combined into the
     cites        number           citation count         ┘ "year · N cites" row
     rows         node[]           OPTIONAL — replace the authors / year·cites
                                   rows with custom nodes (author-mode, an
                                   off-network glyph, anything). venue stays.
     titlePrefix  node             OPTIONAL — rendered before the title (seed dot)
     lead         node             OPTIONAL — leading control slot (star / × / mark)
     trail        node             OPTIONAL — trailing chip slot (depth / category)
     footer       node             OPTIONAL — below the meta (reject reason)
     selected     bool             black-invert selected state
     fresh        bool             one-shot flash (newly-accepted row)
     offnet       bool             dimmed + off-network treatment (Explore)
     onClick      fn               row click
     tooltip      string           title="" hover tooltip
     className    string           extra classes on the root

   USAGE
     <PaperCard title={p.title} venue={p.venue} authors={p.authors}
                year={p.year} cites={p.cites}
                lead={<button className="pcard-starbtn">…</button>}
                onClick={() => open(p.id)} />
   ========================================================================== */

function PaperCard(props) {
  const {
    title, venue, authors, year, cites,
    rows, titlePrefix, lead, trail, footer,
    selected, fresh, offnet,
    onClick, tooltip, className, paperId,
  } = props;

  const cls = ["pcard"]
    .concat(selected ? ["is-selected"] : [])
    .concat(fresh ? ["is-fresh"] : [])
    .concat(offnet ? ["is-offnet"] : [])
    .concat(className ? [className] : [])
    .join(" ");

  // Default meta rows: authors, then "year · N cites". Callers needing custom
  // content (author-mode, off-network glyph, …) pass `rows` instead.
  let metaRows = rows;
  if (metaRows == null) {
    const yc = [
      year != null && year !== "" ? String(year) : null,
      cites != null ? PaperCard.fmtCites(cites) : null,
    ].filter(Boolean).join(" · ") || "—";
    metaRows = [authors || "—", yc];
  }

  return (
    <div className={cls} onClick={onClick} title={tooltip} data-paper-id={paperId}>
      {lead != null && <span className="pcard-lead">{lead}</span>}
      <div className="pcard-body">
        <div className="pcard-headrow">
          <div className="pcard-title">{titlePrefix}{title}</div>
          {trail != null && <span className="pcard-trail">{trail}</span>}
        </div>
        <div className="pcard-meta pcard-venue">{venue || "—"}</div>
        {metaRows.map((r, i) => <div className="pcard-meta" key={i}>{r}</div>)}
        {footer}
      </div>
    </div>
  );
}

// Default citation formatter — grouped digits + " cites". Override the whole
// row via the `rows` prop when a list needs different wording (e.g. "N papers").
PaperCard.fmtCites = function (n) {
  return (Number(n) || 0).toLocaleString() + " cites";
};

if (typeof window !== "undefined") { Object.assign(window, { PaperCard }); }
