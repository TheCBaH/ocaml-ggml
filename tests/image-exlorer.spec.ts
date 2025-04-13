import { test, expect } from '@playwright/test';

test('graph', async ({ page }) => {
  await page.goto('http://127.0.0.1:8080/dist/');
  await page.getByText('op node count').click();
  await expect(page).toHaveScreenshot();
});

