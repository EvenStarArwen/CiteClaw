// Static file server for web/design-reference.
//
// The Claude-Design demo is shipped as a self-contained bundle, but it is still
// served over HTTP rather than file:// so that (a) relative sibling paths behave
// exactly as they would in a real deployment and (b) network traffic can be
// observed and asserted on.

import http from 'node:http';
import fs from 'node:fs';
import fsp from 'node:fs/promises';
import path from 'node:path';
import { DESIGN_REFERENCE_DIR } from './config.mjs';

const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.csv': 'text/csv; charset=utf-8',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.svg': 'image/svg+xml',
  '.woff': 'font/woff',
  '.woff2': 'font/woff2',
  '.ttf': 'font/ttf',
  '.md': 'text/markdown; charset=utf-8',
  '.graphml': 'application/xml; charset=utf-8',
};

/**
 * @param {object} opts
 * @param {string} [opts.root]  directory to serve (default: web/design-reference)
 * @param {number} [opts.port]  0 = ephemeral
 * @returns {Promise<{url: string, port: number, root: string, close: () => Promise<void>}>}
 */
export async function startStaticServer({ root = DESIGN_REFERENCE_DIR, port = 0, quiet = true } = {}) {
  const absRoot = path.resolve(root);
  await fsp.access(absRoot);

  const server = http.createServer(async (req, res) => {
    try {
      const url = new URL(req.url, 'http://localhost');
      let rel = decodeURIComponent(url.pathname);
      if (rel.endsWith('/')) rel += 'index.html';
      const filePath = path.resolve(absRoot, '.' + rel);
      // Directory-traversal guard.
      if (filePath !== absRoot && !filePath.startsWith(absRoot + path.sep)) {
        res.writeHead(403).end('Forbidden');
        return;
      }
      const stat = await fsp.stat(filePath).catch(() => null);
      if (!stat || !stat.isFile()) {
        if (!quiet) console.warn(`[parity:server] 404 ${rel}`);
        res.writeHead(404, { 'content-type': 'text/plain' }).end('Not found');
        return;
      }
      res.writeHead(200, {
        'content-type': MIME[path.extname(filePath).toLowerCase()] ?? 'application/octet-stream',
        'content-length': stat.size,
        // Deterministic captures must never be served a stale/varying cache entry.
        'cache-control': 'no-store',
      });
      fs.createReadStream(filePath).pipe(res);
    } catch (err) {
      res.writeHead(500, { 'content-type': 'text/plain' }).end(String(err && err.message));
    }
  });

  await new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(port, '127.0.0.1', resolve);
  });
  const actualPort = server.address().port;

  return {
    url: `http://127.0.0.1:${actualPort}`,
    port: actualPort,
    root: absRoot,
    close: () => new Promise((resolve) => server.close(() => resolve())),
  };
}

/** URL of the design demo entry page on a running server. */
export function demoUrl(serverUrl, file = 'KnowledgeLab iPad Demo.html') {
  return `${serverUrl}/${encodeURIComponent(file)}`;
}
