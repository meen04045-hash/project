// Local-only LIFF UI mocks; no browser, LINE, or Firebase request.
const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

const html = fs.readFileSync('templates/risk_demo_contact.html', 'utf8');
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1]
  .replace('{{ liff_id|tojson }}', '"synthetic-liff"');

async function run({ loggedIn = true, token = 'synthetic-token', initError = false,
                     apiStatus = 200 } = {}) {
  const elements = new Map();
  const calls = { login: 0, fetch: 0 };
  const document = { getElementById(id) {
    if (!elements.has(id)) elements.set(id, {
      disabled: true, value: '', checked: false, textContent: '',
      addEventListener() {},
    });
    return elements.get(id);
  } };
  const liff = {
    init: async () => { if (initError) throw new Error('synthetic init failure'); },
    isLoggedIn: () => loggedIn,
    login: () => { calls.login++; },
    getIDToken: () => token,
  };
  const fetch = async (path, options) => {
    calls.fetch++;
    assert.equal(path, '/demo/api/contact');
    assert.equal(options.method, 'GET');
    assert.equal(options.headers.Authorization, `Bearer ${token}`);
    return { ok: apiStatus === 200, status: apiStatus,
             json: async () => ({ consent: false }) };
  };
  vm.runInNewContext(script, { document, liff, fetch, crypto: {}, console: {} });
  await new Promise(setImmediate);
  return { calls, status: elements.get('status').textContent,
           saveDisabled: elements.get('save').disabled };
}

test('in-LINE login with allowed API enables form', async () => {
  const result = await run();
  assert.equal(result.calls.fetch, 1);
  assert.equal(result.saveDisabled, false);
});

test('external browser starts login redirect without API call', async () => {
  const result = await run({ loggedIn: false });
  assert.equal(result.calls.login, 1);
  assert.equal(result.calls.fetch, 0);
  assert.equal(result.saveDisabled, true);
});

test('missing token remains disabled and reports token stage', async () => {
  const result = await run({ token: null });
  assert.match(result.status, /LINE ID token/);
  assert.equal(result.calls.fetch, 0);
  assert.equal(result.saveDisabled, true);
});

test('LIFF initialization error remains disabled and reports LIFF stage', async () => {
  const result = await run({ initError: true });
  assert.match(result.status, /TEST LIFF/);
  assert.equal(result.calls.fetch, 0);
  assert.equal(result.saveDisabled, true);
});

test('deny-all 403 is reported as access denial, not LIFF failure', async () => {
  const result = await run({ apiStatus: 403 });
  assert.match(result.status, /ยังไม่ได้รับอนุญาต/);
  assert.doesNotMatch(result.status, /token|LIFF/i);
  assert.equal(result.saveDisabled, true);
});

test('invalid token 401 and API outage remain distinct', async () => {
  const invalid = await run({ apiStatus: 401 });
  const outage = await run({ apiStatus: 503 });
  assert.match(invalid.status, /การยืนยันบัญชี LINE ไม่สำเร็จ/);
  assert.match(outage.status, /ติดต่อระบบข้อมูลติดต่อไม่สำเร็จ/);
  assert.equal(invalid.saveDisabled, true);
  assert.equal(outage.saveDisabled, true);
});
