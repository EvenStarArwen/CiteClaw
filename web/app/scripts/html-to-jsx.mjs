/**
 * html-to-jsx.mjs — mechanical HTML -> JSX converter for the demo templates.
 *
 * The rewrite's first rule is that markup is TRANSPLANTED, not authored. This
 * file is what makes that checkable: the JSX under src/screens and
 * src/components/shared is machine-produced from the demo's own templates, so
 * "did someone quietly improve the markup?" is answered by re-running the
 * generator and diffing, not by reading 200 kB of JSX.
 *
 * The only transformations are the ones JSX requires:
 *   - `class` -> `className`, `for` -> `htmlFor`, and the rest of React's
 *     attribute spellings (including the SVG set, which is separate);
 *   - inline `style` strings -> style objects, custom properties quoted;
 *   - EVERY attribute value emitted as `attr={"…"}` and EVERY text node as
 *     `{"…"}`, with non-ASCII escaped. That is not stylistic: the templates
 *     contain 48 literal U+00A0 non-breaking spaces and inter-element
 *     whitespace that JSX's own text handling would trim, and both are visible.
 *   - `style-hover="…"` -> a generated class + a rule in
 *     src/styles/demo-style-hover.css (see hoverClass below).
 *
 * Nothing here decides anything about the design. If the output looks wrong,
 * the template is what it is — file an escalation, do not patch the generator.
 */
import fs from 'node:fs';
import crypto from 'node:crypto';
import { parseFragment } from 'parse5';

/**
 * `style-hover` is not an HTML attribute — it is a dc-runtime directive.
 * support.js strips it at compile time, hands the element a generated class
 * (`scp0`, `scp1`, …, deduped by declaration text across every screen in the
 * document) and inserts `.scpN:hover { <decls> !important }` into the CSSOM.
 * Verified in the running demo: `.tb-project` renders as
 * `class="tb-project scp0"` with `.scp0:hover { background: var(--row-hover)
 * !important; }`, and carries no style-hover attribute.
 *
 * We reproduce the mechanism, not support.js's counter: the class name is a
 * hash of the declaration text, so the same declaration gets the same class in
 * every screen without a shared mutable counter. The emitted rule keeps the
 * per-declaration `!important` — without it the hover never wins against the
 * element's own inline `style`.
 */
export const HOVER_RULES = new Map(); // className -> declaration text

function hoverClass(decls) {
  const norm = decls.trim().replace(/;\s*$/, '');
  const name = 'dsh-' + crypto.createHash('sha1').update(norm).digest('hex').slice(0, 8);
  if (!HOVER_RULES.has(name)) HOVER_RULES.set(name, norm);
  return name;
}

export function hoverStylesheet() {
  const lines = [...HOVER_RULES.entries()].sort(([a], [b]) => (a < b ? -1 : 1)).map(
    ([cls, decls]) =>
      `.${cls}:hover{${decls
        .split(';')
        .map((d) => d.trim())
        .filter(Boolean)
        .map((d) => d + ' !important')
        .join(';')};}`,
  );
  return lines.join('\n') + '\n';
}

const SVG_ATTR = new Map(Object.entries({
  'stroke-width': 'strokeWidth', 'stroke-linecap': 'strokeLinecap', 'stroke-linejoin': 'strokeLinejoin',
  'stroke-dasharray': 'strokeDasharray', 'stroke-dashoffset': 'strokeDashoffset', 'stroke-opacity': 'strokeOpacity',
  'stroke-miterlimit': 'strokeMiterlimit', 'fill-rule': 'fillRule', 'fill-opacity': 'fillOpacity',
  'clip-rule': 'clipRule', 'clip-path': 'clipPath', 'text-anchor': 'textAnchor',
  'dominant-baseline': 'dominantBaseline', 'font-family': 'fontFamily', 'font-size': 'fontSize',
  'font-weight': 'fontWeight', 'letter-spacing': 'letterSpacing', 'stop-color': 'stopColor',
  'stop-opacity': 'stopOpacity', 'marker-end': 'markerEnd', 'marker-start': 'markerStart',
  'vector-effect': 'vectorEffect', 'shape-rendering': 'shapeRendering', 'paint-order': 'paintOrder',
  'transform-origin': 'transformOrigin', 'xlink:href': 'xlinkHref', 'xmlns:xlink': 'xmlnsXlink',
  'color-interpolation-filters': 'colorInterpolationFilters', 'flood-color': 'floodColor',
  'flood-opacity': 'floodOpacity', 'stroke': 'stroke', 'fill': 'fill',
}));

const HTML_ATTR = new Map(Object.entries({
  class: 'className', for: 'htmlFor', tabindex: 'tabIndex', readonly: 'readOnly',
  maxlength: 'maxLength', minlength: 'minLength', autocomplete: 'autoComplete',
  contenteditable: 'contentEditable', spellcheck: 'spellCheck', colspan: 'colSpan',
  rowspan: 'rowSpan', srcset: 'srcSet', crossorigin: 'crossOrigin', autofocus: 'autoFocus',
  novalidate: 'noValidate', enctype: 'encType', 'accept-charset': 'acceptCharset',
  'http-equiv': 'httpEquiv', datetime: 'dateTime', usemap: 'useMap', inputmode: 'inputMode',
  autoplay: 'autoPlay', playsinline: 'playsInline', allowfullscreen: 'allowFullScreen',
  frameborder: 'frameBorder', cellpadding: 'cellPadding', cellspacing: 'cellSpacing',
}));

const BOOL_ATTR = new Set(['multiple', 'disabled', 'checked', 'selected', 'hidden', 'readonly', 'required', 'autofocus', 'novalidate', 'default', 'reversed', 'loop', 'muted', 'controls', 'open']);

const VOID = new Set(['area', 'base', 'br', 'col', 'embed', 'hr', 'img', 'input', 'link', 'meta', 'param', 'source', 'track', 'wbr']);

function jsStr(s) {
  let out = '';
  for (const ch of s) {
    const c = ch.codePointAt(0);
    if (ch === '"') out += '\\"';
    else if (ch === '\\') out += '\\\\';
    else if (ch === '\n') out += '\\n';
    else if (ch === '\r') out += '\\r';
    else if (ch === '\t') out += '\\t';
    else if (c < 0x20 || c === 0x7f) out += '\\u' + c.toString(16).padStart(4, '0');
    else if (c > 0x7e) out += '\\u' + c.toString(16).padStart(4, '0');
    else out += ch;
  }
  return '"' + out + '"';
}

function camelProp(p) {
  if (p.startsWith('--')) return null; // keep verbatim, needs quoting
  return p.replace(/-([a-z])/g, (_, c) => c.toUpperCase());
}

function styleToObject(css) {
  const decls = [];
  let depth = 0, cur = '';
  for (const ch of css) {
    if (ch === '(') depth++;
    if (ch === ')') depth--;
    if (ch === ';' && depth === 0) { decls.push(cur); cur = ''; } else cur += ch;
  }
  if (cur.trim()) decls.push(cur);
  const parts = [];
  for (const d of decls) {
    const i = d.indexOf(':');
    if (i < 0) continue;
    const prop = d.slice(0, i).trim();
    const val = d.slice(i + 1).trim();
    if (!prop) continue;
    const camel = camelProp(prop);
    parts.push(`${camel === null ? jsStr(prop) : camel}: ${jsStr(val)}`);
  }
  return '{ ' + parts.join(', ') + ' }';
}

const OVERRIDES = new Map(); // "attrName=rawValue" -> literal JSX expression
OVERRIDES.set('data-pstyle={{ pipelineStyle }}', 'data-pstyle={pipelineStyle}');
OVERRIDES.set('data-cardlayout={{ cardLayout }}', 'data-cardlayout={cardLayout}');

function attrToJsx(name, value, ns) {
  if (name === 'ref' && /\{\{\s*rootRef\s*\}\}/.test(value)) return 'ref={rootRef}';
  if (name === 'style') return `style={${styleToObject(value)}}`;
  if (name === 'data-pstyle' && value.includes('{{')) return 'data-pstyle={pipelineStyle}';
  if (name === 'data-cardlayout' && value.includes('{{')) return 'data-cardlayout={cardLayout}';
  let jsxName = name;
  if (ns === 'svg' && SVG_ATTR.has(name)) jsxName = SVG_ATTR.get(name);
  else if (HTML_ATTR.has(name)) jsxName = HTML_ATTR.get(name);
  if (name.startsWith('data-') || name.startsWith('aria-')) jsxName = name;
  if (BOOL_ATTR.has(name) && (value === '' || value === name)) return `${jsxName}={true}`;
  return `${jsxName}={${jsStr(value)}}`;
}

function isSvgCtx(node) {
  let n = node;
  while (n) { if (n.tagName === 'svg') return true; n = n.parentNode; }
  return false;
}

function emit(node, indent, out) {
  const pad = '  '.repeat(indent);
  if (node.nodeName === '#text') {
    const t = node.value;
    if (t === '') return;
    out.push(pad + '{' + jsStr(t) + '}');
    return;
  }
  if (node.nodeName === '#comment') return;
  const tag = node.tagName;
  const ns = isSvgCtx(node) ? 'svg' : 'html';
  const rawAttrs = (node.attrs || []).filter((a) => a.name !== 'style-hover');
  const hoverAttr = (node.attrs || []).find((a) => a.name === 'style-hover');
  const extraClass = hoverAttr ? hoverClass(hoverAttr.value) : null;
  const attrs = rawAttrs.map((a) => {
    const nm = a.prefix ? `${a.prefix}:${a.name}` : a.name;
    if (extraClass && nm === 'class') return attrToJsx(nm, a.value + ' ' + extraClass, ns);
    return attrToJsx(nm, a.value, ns);
  });
  if (extraClass && !rawAttrs.some((a) => a.name === 'class')) {
    attrs.unshift(attrToJsx('class', extraClass, ns));
  }
  const kids = (node.childNodes || []).filter((c) => !(c.nodeName === '#text' && c.value === ''));
  const openParts = attrs.length ? `<${tag}\n${attrs.map((a) => pad + '  ' + a).join('\n')}\n${pad}` : `<${tag}`;
  if (VOID.has(tag) || kids.length === 0) {
    out.push(attrs.length ? `${pad}${openParts}/>` : `${pad}<${tag} />`);
    return;
  }
  out.push(attrs.length ? `${pad}${openParts}>` : `${pad}<${tag}>`);
  for (const k of kids) emit(k, indent + 1, out);
  out.push(`${pad}</${tag}>`);
}

export function convert(html, indent = 0) {
  const frag = parseFragment(html);
  const out = [];
  for (const n of frag.childNodes) emit(n, indent, out);
  return out.join('\n');
}

if (process.argv[1].endsWith('convert.mjs')) {
  const src = fs.readFileSync(process.argv[2], 'utf8');
  process.stdout.write(convert(src, Number(process.argv[3] || 0)) + '\n');
}
