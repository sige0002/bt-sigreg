const { test, expect } = require('@playwright/test');

const url = process.env.BT_VIEWER_URL;
test.skip(!url, 'Set BT_VIEWER_URL to a generated PushT viewer (HTTP or file URL)');

test('playback, frame scores, seeking, filtering and rapid case switching', async ({ page }) => {
  const errors = [];
  page.on('pageerror', e => errors.push(e.message));
  page.on('response', r => { if (r.status() >= 400 && !r.url().endsWith('/favicon.ico')) errors.push(`${r.status()} ${r.url()}`); });
  await page.goto(url);
  await expect(page.locator('#case-title')).toContainText('1回目');
  await page.waitForFunction(() => document.querySelector('#video').readyState >= 2);
  const initial = await page.locator('#video').evaluate(v => ({ width: v.videoWidth, height: v.videoHeight, duration: v.duration, source: v.dataset.source }));
  expect(initial.width).toBe(448);
  expect(initial.height).toBe(304);
  expect(initial.source).toContain('videos/trial_001.mp4');
  expect(initial.duration).toBeGreaterThan(0);
  await page.locator('#video').evaluate(v => v.play());
  await page.waitForFunction(() => document.querySelector('#video').currentTime > .4);
  await page.locator('#video').evaluate(v => v.pause());
  const score = await page.evaluate(() => {
    const data = JSON.parse(document.querySelector('#data').textContent);
    const v = document.querySelector('#video'), c = data.cases[0];
    return c.scores[Math.min(c.scores.length - 1, Math.floor(v.currentTime * c.fps + 1e-6))];
  });
  await expect(page.locator('#frame-title')).toContainText(`step ${score.step}`);
  await expect(page.locator('#position')).toHaveText(score.position.toFixed(3));
  await page.locator('#jump').click();
  await page.waitForFunction(() => !document.querySelector('#video').seeking);
  const selected = await page.evaluate(() => JSON.parse(document.querySelector('#data').textContent).cases[0].selected_step);
  await expect(page.locator('#frame-title')).toContainText(`step ${selected}`);
  await page.locator('#overlay-toggle').check();
  await expect(page.locator('#overlay')).toBeVisible();
  await page.locator('#filter').selectOption('success');
  await page.locator('#case-list button').first().click();
  await expect(page.locator('#outcome')).toContainText('成功');
  await page.locator('#filter').selectOption('failure');
  await page.locator('#case-list button').first().click();
  await expect(page.locator('#outcome')).toContainText('失敗');
  await page.locator('#filter').selectOption('all');
  await page.locator('#case-list button').first().click();
  // Exercise changes before the previous video's metadata/decode has completed.
  await page.evaluate(() => {
    for (let i = 0; i < 20; i++) document.querySelector('#next').click();
    for (let i = 0; i < 19; i++) document.querySelector('#prev').click();
  });
  await expect(page.locator('#case-title')).toContainText('2回目');
  await page.waitForFunction(() => document.querySelector('#video').readyState >= 2);
  await page.locator('#video').evaluate(v => v.play());
  await page.waitForFunction(() => document.querySelector('#video').currentTime > .2);
  await page.locator('#video').evaluate(v => v.pause());
  expect(await page.locator('#video').evaluate(v => v.error)).toBeNull();
  await expect(page.locator('#download-video')).toHaveAttribute('href', 'videos/trial_002.mp4');
  expect(errors).toEqual([]);
});

test('filter keeps selected trial when it still matches, and switching does not jump the page', async ({ page }) => {
  await page.goto(url);
  await page.locator('#case-list button').nth(3).click();
  await expect(page.locator('#case-title')).toContainText('4回目');
  const matches = await page.locator('#outcome').textContent();
  await page.locator('#filter').selectOption(matches.startsWith('成功') ? 'success' : 'failure');
  await expect(page.locator('#case-title')).toContainText('4回目');
  await page.locator('#filter').selectOption('all');
  await expect(page.locator('#case-title')).toContainText('4回目');
  await page.locator('#video').scrollIntoViewIfNeeded();
  const before = await page.evaluate(() => window.scrollY);
  await page.evaluate(() => document.querySelector('#next').click());
  await page.waitForFunction(() => document.querySelector('#video').readyState >= 2);
  const after = await page.evaluate(() => window.scrollY);
  expect(Math.abs(after - before)).toBeLessThan(8);
});

test('reload and browser history preserve the trial instead of falling back to trial 1', async ({ page }) => {
  await page.goto(url);
  await page.locator('#case-list button').nth(26).click();
  await expect(page.locator('#case-title')).toContainText('27回目');
  await page.reload();
  await expect(page.locator('#case-title')).toContainText('27回目');
  await page.locator('#next').click();
  await expect(page.locator('#case-title')).toContainText('28回目');
  await page.goBack();
  await expect(page.locator('#case-title')).toContainText('27回目');
  await page.goForward();
  await expect(page.locator('#case-title')).toContainText('28回目');
  await page.waitForFunction(() => document.querySelector('#video').readyState >= 2);
  await expect(page.locator('#video')).toHaveAttribute('data-source', 'videos/trial_028.mp4');
});

test('other trial playback ends and replays without changing its video, goal or identity', async ({ page }) => {
  await page.goto(url);
  for (const n of [4, 27, 50]) {
    await page.locator(`#case-list button[data-case="${n - 1}"]`).click();
    await page.waitForFunction(() => document.querySelector('#video').readyState >= 2);
    for (let replay = 0; replay < 2; replay++) {
      await page.locator('#video').evaluate(replay => {
        const v = document.querySelector('#video');
        v.currentTime = 0; v.playbackRate = replay ? 4 : 1; return v.play();
      }, replay);
      await page.waitForFunction(() => document.querySelector('#video').ended);
      await expect(page.locator('#case-title')).toHaveText(`${n}回目（env_${n - 1}.mp4）`);
      await expect(page.locator('#video')).toHaveAttribute('data-source', `videos/trial_${String(n).padStart(3, '0')}.mp4`);
      await expect(page.locator('#goal')).toHaveAttribute('src', `assets/case_${n - 1}_goal.png`);
      await expect(page.locator(`#case-list button[data-case="${n - 1}"]`)).toHaveAttribute('aria-current', 'true');
      await expect(page).toHaveURL(new RegExp(`#trial=${n}&filter=all$`));
      const last = await page.evaluate(i => JSON.parse(document.querySelector('#data').textContent).cases[i].scores.length, n - 1);
      await expect(page.locator('#frame-title')).toContainText(`step ${last}`);
    }
  }
});

test('excluded and missing trials never silently select the first case', async ({ page }) => {
  await page.goto(url);
  await page.locator('#case-list button').nth(26).click();
  const outcome = await page.locator('#outcome').textContent();
  const opposite = outcome.startsWith('成功') ? 'failure' : 'success';
  await page.locator('#filter').selectOption(opposite);
  await expect(page.locator('#case-title')).toContainText('27回目');
  await expect(page.locator('#selection-status')).toContainText('表示条件の対象外');
  await page.reload();
  await expect(page.locator('#case-title')).toContainText('27回目');
  await expect(page.locator('#filter')).toHaveValue(opposite);
  await page.locator('#filter').selectOption('all');
  await expect(page.locator('#case-title')).toContainText('27回目');
  for (const trial of ['999', '0', '-1', 'abc', '1.5', '']) {
    await page.goto(`${url}#trial=${trial}&filter=all`);
    await expect(page.locator('#selection-status')).toContainText('指定された試行が見つかりません');
    await expect(page.locator('#viewer')).toBeHidden();
    await page.locator('#case-list button').nth(3).click();
    await expect(page.locator('#case-title')).toContainText('4回目');
  }
});

test('mobile layout, end frame and all media references', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto(url);
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  await page.waitForFunction(() => document.querySelector('#video').readyState >= 2);
  await page.locator('#video').evaluate(v => { v.currentTime = v.duration; });
  await page.waitForFunction(() => !document.querySelector('#video').seeking);
  const last = await page.evaluate(() => JSON.parse(document.querySelector('#data').textContent).cases[0].scores.length);
  await expect(page.locator('#frame-title')).toContainText(`step ${last}`);
  const failures = await page.evaluate(async () => {
    if (location.protocol === 'file:') return []; // File URLs are checked by decoded playback above.
    const data = JSON.parse(document.querySelector('#data').textContent), failed = [];
    for (const c of data.cases) {
      for (const path of [c.goal_video, `assets/case_${c.case}_goal.png`, `assets/case_${c.case}_evidence.png`]) {
        const r = await fetch(path, { method: 'HEAD' });
        if (!r.ok) failed.push(path);
      }
    }
    return failed;
  });
  expect(failures).toEqual([]);
});

test('delayed loads cannot replace the selected video and failures recover', async ({ page }) => {
  test.skip(url.startsWith('file:'), 'HTTP fetch cancellation/failure test');
  await page.addInitScript(() => {
    window.activeVideoBlobs = new Set();
    const create = URL.createObjectURL.bind(URL), revoke = URL.revokeObjectURL.bind(URL);
    URL.createObjectURL = value => { const u = create(value); activeVideoBlobs.add(u); return u; };
    URL.revokeObjectURL = value => { activeVideoBlobs.delete(value); return revoke(value); };
  });
  await page.goto(url);
  await page.waitForFunction(() => document.querySelector('#video').readyState >= 2);
  await page.route('**/videos/trial_002.mp4', async route => {
    await new Promise(resolve => setTimeout(resolve, 350));
    await route.continue().catch(() => {});
  });
  await page.evaluate(() => { document.querySelector('#next').click(); document.querySelector('#next').click(); });
  await page.waitForFunction(() => document.querySelector('#video').readyState >= 2);
  await expect(page.locator('#case-title')).toContainText('3回目');
  await expect(page.locator('#video')).toHaveAttribute('data-source', 'videos/trial_003.mp4');
  expect(await page.evaluate(() => activeVideoBlobs.size)).toBe(1);
  await page.unroute('**/videos/trial_002.mp4');
  await page.route('**/videos/trial_002.mp4', route => route.fulfill({ status: 404, body: 'missing video fixture' }));
  await page.locator('#prev').click();
  await expect(page.locator('#video-status')).toContainText('HTTP 404');
  await expect(page.locator('#jump')).toBeDisabled();
  await page.locator('#prev').click();
  await page.waitForFunction(() => document.querySelector('#video').readyState >= 2);
  await expect(page.locator('#video-status')).toBeEmpty();
  expect(await page.evaluate(() => activeVideoBlobs.size)).toBe(1);
});
