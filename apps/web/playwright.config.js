const { defineConfig } = require('@playwright/test');

module.exports = defineConfig({
  testDir: './tests',
  timeout: 20000,
  retries: 0,
  workers: 1,
  reporter: [['list'], ['json', { outputFile: 'tests/results.json' }]],
  use: {
    baseURL: 'http://localhost:4028',
    headless: true,
    viewport: { width: 1280, height: 800 },
    ignoreHTTPSErrors: true,
    screenshot: 'only-on-failure',
    video: 'off',
  },
});
