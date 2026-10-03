const { test } = require('node:test');
const assert = require('node:assert/strict');
const { readFileSync, mkdtempSync, writeFileSync, rmSync } = require('node:fs');
const { tmpdir } = require('node:os');
const { join } = require('node:path');
const typescript = require('typescript');

const directory = mkdtempSync(join(tmpdir(), 'rsky-roi-rules-test-'));
const compiled = join(directory, 'roi-rules.cjs');
writeFileSync(compiled, typescript.transpileModule(readFileSync(join(__dirname, '../lib/roi-rules.ts'), 'utf8'), {
  compilerOptions: { target: typescript.ScriptTarget.ES2022, module: typescript.ModuleKind.CommonJS },
}).outputText);
const {
  DEFAULT_INSPECTION_CONFIG, applyPreset, articleFitsCell, normalizeInspectionConfig, verdictSummary, INSPECTION_PRESETS,
  INSPECTION_MODES, orderClockwise, baselineMatchesRoi, toDatetimeLocal, fromDatetimeLocal, occupancyText, occupancyColor,
} = require(compiled);
test.after(() => rmSync(directory, { recursive: true, force: true }));

test('defaults follow the plan: hybrid carton 800x600, ±30 mm, 50 mm, 5°', () => {
  assert.equal(DEFAULT_INSPECTION_CONFIG.mode, 'HYBRID');
  assert.equal(DEFAULT_INSPECTION_CONFIG.width_mm, 800);
  assert.equal(DEFAULT_INSPECTION_CONFIG.height_mm, 600);
  assert.equal(DEFAULT_INSPECTION_CONFIG.tolerance_w_mm, 30);
  assert.equal(DEFAULT_INSPECTION_CONFIG.max_center_offset_mm, 50);
  assert.equal(DEFAULT_INSPECTION_CONFIG.max_rotation_deg, 5);
  assert.equal(DEFAULT_INSPECTION_CONFIG.temporal_window_frames, 3);
});

test('presets set the nominal size and keep the tolerances', () => {
  const pallet = applyPreset({ ...DEFAULT_INSPECTION_CONFIG, tolerance_w_mm: 12 }, 'pallet_1100x1100');
  assert.equal(pallet.width_mm, 1100);
  assert.equal(pallet.height_mm, 1100);
  assert.equal(pallet.tolerance_w_mm, 12);
  const custom = applyPreset({ ...DEFAULT_INSPECTION_CONFIG, width_mm: 640 }, 'custom');
  assert.equal(custom.width_mm, 640);
  assert.deepEqual(Object.keys(INSPECTION_PRESETS).sort(), ['carton_800x600', 'custom', 'pallet_1100x1100', 'pallet_1200x1000']);
});

test('stored configurations are normalised on reload (F5)', () => {
  const restored = normalizeInspectionConfig({ mode: 'CV_ONLY', max_rotation_deg: 3.5, width_mm: 'x', target_preset: 'bogus' });
  assert.equal(restored.mode, 'CV_ONLY');
  assert.equal(restored.max_rotation_deg, 3.5);
  assert.equal(restored.width_mm, 800);
  assert.equal(restored.target_preset, 'custom');
  assert.equal(normalizeInspectionConfig(null).mode, 'HYBRID');
});

test('verdict summary reads like target.md Test 4.3', () => {
  const summary = verdictSummary({
    status: 'OK', measurement: { width_mm: 808, height_mm: 602, offset_mm: 14, rotation_deg: 1.5 },
    checks: { width: { ok: true }, height: { ok: true }, center_offset: { ok: true }, rotation: { ok: true } },
    ai: { available: true, detections: [] }, ai_class: 'Thùng chuẩn', ai_confidence: 0.98,
  });
  assert.equal(summary, 'Chiều rộng: 808mm (PASS) | Chiều dài: 602mm (PASS) | Lệch tâm: 14mm (PASS) | Góc: 1.5° (PASS) | AI: Thùng chuẩn 98% => KẾT QUẢ: OK');
});

test('the floor cell size is configured per station, 1 m x 1 m only by default', () => {
  assert.equal(DEFAULT_INSPECTION_CONFIG.roi_width_mm, 1000);
  assert.equal(DEFAULT_INSPECTION_CONFIG.roi_height_mm, 1000);
  const pallet = applyPreset(DEFAULT_INSPECTION_CONFIG, 'pallet_1200x1000');
  assert.equal(articleFitsCell(pallet), false);
  assert.equal(articleFitsCell({ ...pallet, roi_width_mm: 1400, roi_height_mm: 1200 }), true);
  assert.equal(normalizeInspectionConfig({ roi_width_mm: 1400, roi_height_mm: 1200 }).roi_height_mm, 1200);
  assert.equal(normalizeInspectionConfig({}).roi_width_mm, 1000);
});

test('each mode states what it answers to "does the cell hold goods"', () => {
  const byMode = Object.fromEntries(INSPECTION_MODES.map(mode => [mode.value, mode.outcome]));
  assert.match(byMode.CV_ONLY, /CÓ HÀNG/);
  assert.match(byMode.HYBRID, /KHÔNG XÁC ĐỊNH/);
  assert.match(byMode.AI_ONLY, /đối tượng áp dụng/);
});

test('monitor display options default on and survive a reload', () => {
  assert.equal(DEFAULT_INSPECTION_CONFIG.monitor_overlay, true);
  const restored = normalizeInspectionConfig({ monitor_label: false });
  assert.deepEqual([restored.monitor_overlay, restored.monitor_label, restored.monitor_object], [true, false, true]);
});

test('occupancy text and colours match the storage-slot chips', () => {
  assert.equal(occupancyText({ state: 'CARFULL', label: 'Rack', confidence: 0.914 }), 'CÓ HÀNG · Rack 91%');
  assert.equal(occupancyText({ state: 'UNKNOWN', label: null, confidence: null }), 'KHÔNG XÁC ĐỊNH');
  assert.equal(occupancyText(null), 'ĐANG CHỜ');
  assert.equal(occupancyColor('CARFULL'), '#f43f5e');
  assert.equal(occupancyColor('EMPTY'), '#10b981');
  assert.equal(occupancyColor(undefined), '#f59e0b');
});

test('a baseline belongs to the drawn ROI and cell size, whatever the click direction', () => {
  const roi = [[0.36, 0.23], [0.67, 0.25], [0.75, 0.8], [0.27, 0.77]];
  const counter = [roi[0], roi[3], roi[2], roi[1]];
  assert.deepEqual(orderClockwise(counter), roi);
  const entry = { points: roi, cell_mm: [1400, 1200] };
  assert.equal(baselineMatchesRoi(entry, counter, [1400, 1200]), true);
  assert.equal(baselineMatchesRoi(entry, roi, [1000, 1000]), false);
  assert.equal(baselineMatchesRoi(entry, roi.map(([x, y]) => [x + 0.01, y]), [1400, 1200]), false);
  assert.equal(baselineMatchesRoi(entry, roi.slice(0, 3), [1400, 1200]), false);
});

test('recorded-video moments round-trip through the datetime picker', () => {
  const moment = Math.floor(Date.now() / 1000) - 3 * 3600;
  assert.equal(fromDatetimeLocal(toDatetimeLocal(moment)), moment);
  assert.equal(fromDatetimeLocal('2026-10-03T06:10'), new Date(2026, 9, 3, 6, 10, 0).getTime() / 1000);
  assert.equal(fromDatetimeLocal(''), null);
  assert.equal(fromDatetimeLocal('06:10'), null);
});
