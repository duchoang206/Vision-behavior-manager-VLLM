const { test } = require('node:test');
const assert = require('node:assert/strict');
const { readFileSync, mkdtempSync, writeFileSync, rmSync } = require('node:fs');
const { tmpdir } = require('node:os');
const { join } = require('node:path');
const typescript = require('typescript');

const directory = mkdtempSync(join(tmpdir(), 'rsky-comm-test-'));
const compiled = join(directory, 'comm-gateway.cjs');
writeFileSync(compiled, typescript.transpileModule(readFileSync(join(__dirname, '../lib/comm-gateway.ts'), 'utf8'), {
  compilerOptions: { target: typescript.ScriptTarget.ES2022, module: typescript.ModuleKind.CommonJS },
}).outputText);
const { renderTemplate, fmsPreview, sampleContext } = require(compiled);
test.after(() => rmSync(directory, { recursive: true, force: true }));

const GATE5 = '{\n  "station": "ST_{slot_id}",\n  "occupied": {is_occupied},\n  "carrier": "{occupant_label}",\n  "cam": "{camera_name}"\n}';

test('live preview matches the backend renderer for the Gate 5 template', () => {
  const result = renderTemplate(GATE5, sampleContext('5', 'Car Full'));
  assert.equal(result.ok, true);
  assert.deepEqual(JSON.parse(result.value), { station: 'ST_5', occupied: true, carrier: 'rack #400', cam: 'Cam 4' });
});

test('booleans stay native and strings are escaped', () => {
  const result = renderTemplate('{"cam": "{camera_name}", "occ": {is_occupied}}', { camera_name: 'Kho "A"', is_occupied: false });
  assert.deepEqual(JSON.parse(result.value), { cam: 'Kho "A"', occ: false });
});

test('syntax errors are reported instead of throwing', () => {
  const result = renderTemplate('{"slot_id": "{slot_id}"', sampleContext());
  assert.equal(result.ok, false);
  assert.match(result.error, /JSON không hợp lệ/);
});

test('text payloads substitute plain values', () => {
  const result = renderTemplate('SLOT={slot_id};S={status}', sampleContext('7', 'Empty'), 'text');
  assert.equal(result.value, 'SLOT=7;S=EMPTY');
});

test('FMS preview uses the WCS slots document and rejects non integer ids', () => {
  const ok = fmsPreview('5', 'Car Full');
  assert.deepEqual(JSON.parse(ok.value), { slot_id: '5', state: 'Car Full', slots: [{ slot_id: '5', state: 'Car Full' }] });
  assert.equal(fmsPreview('A1', 'Car Full').ok, false);
});
