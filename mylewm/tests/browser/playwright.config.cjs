const { defineConfig } = require('@playwright/test');

module.exports = defineConfig({
  testDir: __dirname,
  testMatch: 'pusht_viewer.spec.cjs',
  workers: 1,
  timeout: 30000,
  outputDir: process.env.BT_PLAYWRIGHT_OUTPUT || '/tmp/bt-viewer-playwright-results',
  use: { browserName: 'chromium', headless: true, viewport: { width: 1280, height: 1000 },
         launchOptions: { args: ['--disable-gpu'] }, trace: 'retain-on-failure' },
});
