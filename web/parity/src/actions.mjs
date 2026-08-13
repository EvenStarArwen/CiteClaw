// JSON action-script runner.
//
// One script, two targets: the same file drives the demo and the rewrite, which
// is what makes "did the rebuild land on the same screen?" a mechanical question
// instead of a judgement call.
//
// Script shape:
// {
//   "name": "build-screen",
//   "description": "...",
//   "steps": [
//     { "action": "waitFor",  "selector": "[data-page]", "state": "visible" },
//     { "action": "click",    "selector": "nav button", "text": "Build", "note": "..." },
//     { "action": "hover",    "selector": ".pipeline-node" },
//     { "action": "type",     "selector": "input.query", "value": "moead" },
//     { "action": "press",    "key": "Enter" },
//     { "action": "scroll",   "selector": ".panel", "y": 400 },
//     { "action": "wait",     "ms": 800 },
//     { "action": "settle" },
//     { "action": "shot",     "name": "after-build" }
//   ]
// }
//
// Selector resolution: `selector` is a CSS selector; when `text` is also given,
// the element is narrowed to the one whose trimmed text matches (exact first,
// then substring). This keeps scripts readable and portable across a rewrite
// whose class names will not match the demo's.

import { warmUpVirtualClock, settleAndShoot } from './determinism.mjs';

const DEFAULT_TIMEOUT = 15000;

async function resolve(page, step) {
  const { selector, text, nth } = step;
  if (!selector && !text) throw new Error(`step "${step.action}" needs a selector or text`);
  let locator = selector ? page.locator(selector) : page.locator('body *');
  if (text != null) {
    const exact = locator.filter({ hasText: new RegExp(`^\\s*${escapeRe(text)}\\s*$`) });
    if (await exact.count()) locator = exact;
    else locator = locator.filter({ hasText: text });
  }
  if (typeof nth === 'number') locator = locator.nth(nth);
  else locator = locator.first();
  return locator;
}

function escapeRe(s) {
  return String(s).replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

/**
 * Advance virtual time by `ms`. Uses the fake clock so a "wait" is exactly as
 * long on both targets regardless of machine speed.
 */
async function virtualWait(page, ms) {
  const chunk = 100;
  for (let left = ms; left > 0; left -= chunk) {
    await page.clock.runFor(Math.min(chunk, left));
    await new Promise((r) => setTimeout(r, 8));
  }
}

/**
 * Execute a script against an already-open page.
 * @returns {Promise<Array<{index:number, action:string, name:string, buffer:Buffer, stable:boolean}>>}
 */
export async function runScript(page, script, { shotOptions = {}, determinism = {}, onStep } = {}) {
  const shots = [];
  const steps = script.steps ?? [];

  for (let i = 0; i < steps.length; i++) {
    const step = steps[i];
    const label = step.name || `${String(i).padStart(2, '0')}-${step.action}`;
    const timeout = step.timeout ?? DEFAULT_TIMEOUT;

    // `skipIfVisible` lets one script cover a screen that legitimately renders
    // differently at some viewports — e.g. the demo replaces the whole app with
    // a "Rotate to landscape" gate in portrait, where the nav is not clickable.
    // The step is skipped, but the screenshot is still taken, so the gate itself
    // becomes part of the responsive baseline instead of an error.
    if (step.skipIfVisible) {
      const gate = page.locator(step.skipIfVisible).first();
      const gateVisible = await gate.isVisible().catch(() => false);
      if (gateVisible) {
        if (step.shot !== false) {
          const { buffer, stable, polls } = await settleAndShoot(page, shotOptions, determinism);
          shots.push({ index: i, action: 'skipped:' + step.action, name: label, buffer, stable, polls, note: step.note, skipped: true });
        }
        if (onStep) await onStep({ index: i, step, label, skipped: true });
        continue;
      }
    }

    switch (step.action) {
      case 'goto':
        await page.goto(step.url, { waitUntil: 'load', timeout });
        await warmUpVirtualClock(page, determinism);
        break;
      case 'click': {
        const el = await resolve(page, step);
        await el.waitFor({ state: 'visible', timeout });
        await el.click({ timeout, force: step.force ?? false });
        break;
      }
      case 'hover': {
        const el = await resolve(page, step);
        await el.waitFor({ state: 'visible', timeout });
        await el.hover({ timeout });
        break;
      }
      case 'type': {
        const el = await resolve(page, step);
        await el.waitFor({ state: 'visible', timeout });
        await el.click({ timeout });
        await el.type(String(step.value ?? ''), { delay: 0 });
        break;
      }
      case 'fill': {
        const el = await resolve(page, step);
        await el.fill(String(step.value ?? ''), { timeout });
        break;
      }
      case 'press':
        await page.keyboard.press(step.key ?? 'Enter');
        break;
      case 'moveMouse':
        // Park the cursor. A click leaves the pointer on the clicked element, so
        // without this a hover state gets baked into the baseline.
        await page.mouse.move(step.x ?? 2, step.y ?? 2);
        break;
      case 'blur':
        await page.evaluate(() => document.activeElement && document.activeElement.blur());
        break;
      case 'scroll': {
        if (step.selector) {
          const el = await resolve(page, step);
          await el.evaluate((node, { x, y }) => node.scrollTo(x ?? 0, y ?? 0), { x: step.x, y: step.y });
        } else {
          await page.evaluate(({ x, y }) => window.scrollTo(x ?? 0, y ?? 0), { x: step.x, y: step.y });
        }
        break;
      }
      case 'waitFor': {
        const el = await resolve(page, step);
        await el.waitFor({ state: step.state ?? 'visible', timeout });
        break;
      }
      case 'waitForText':
        await page.waitForFunction(
          (t) => document.body && document.body.innerText.includes(t),
          step.text,
          { timeout },
        );
        break;
      case 'wait':
        await virtualWait(page, step.ms ?? 500);
        break;
      case 'settle':
        await warmUpVirtualClock(page, determinism);
        break;
      case 'shot':
        break; // handled by the screenshot below
      default:
        throw new Error(`Unknown action "${step.action}" at step ${i}`);
    }

    // Screenshot after EVERY step — the point of the runner is a comparable
    // filmstrip, not just a final frame.
    if (step.shot !== false) {
      const { buffer, stable, polls } = await settleAndShoot(page, shotOptions, determinism);
      shots.push({ index: i, action: step.action, name: label, buffer, stable, polls, note: step.note });
    }
    if (onStep) await onStep({ index: i, step, label });
  }

  return shots;
}
