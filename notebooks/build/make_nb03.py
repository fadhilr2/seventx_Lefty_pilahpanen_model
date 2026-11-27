"""Generates notebooks/03_export.ipynb (design: plans/03_export.md).
Embeds notebooks/lib/preprocess.py and preprocess.js, so the Colab kernel writes the exact same files."""
import sys
from pathlib import Path
import nbformat as nbf

out = Path(sys.argv[1])
lib = Path(__file__).resolve().parent.parent / 'lib'   # notebooks/lib, wherever the output goes
PY_SRC = (lib / 'preprocess.py').read_text(encoding='utf-8')
JS_SRC = (lib / 'preprocess.js').read_text(encoding='utf-8')
assert "'''" not in PY_SRC and "'''" not in JS_SRC

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md(r"""
# 03 · Export

Confirms the chosen model after quantization, measures its speed, sets the shipped threshold, and writes everything both demos need. Design: `plans/03_export.md`.

- **Exports EfficientNet-B0**, the chosen model. Add `'mnv3s'` to `MODELS` to also convert MobileNetV3-Small for comparison.
- **Reads** (Drive): `data/manifest.csv`, `dataset.zip`, and notebook 02's models and results in `models/crop_lr1e-4/` and `results/crop_lr1e-4/`.
- **Writes** (Drive):
  - **`models/export/`:** the shipped `.tflite` (used by both the Android app and the web demo), all variants, `labels.txt`, `metadata.json`, `preprocess.py`, `preprocess.js` and `test_vectors/`.
  - **`results/crop_lr1e-4/`:** `export.md` and `export_runs.csv`.
- **Afterwards:** download `models/export/` from Drive into this project's `models/export/`. The kernel can't write to your laptop.

**Rules.** Seed, quantization variant and threshold are chosen on **val** only. Test scores are printed after the choice, for reporting.

**Run:** connect a Colab kernel (GPU helps but isn't needed), then run all cells.
""")

code(r"""
import csv, json, math, random, shutil, subprocess, time, zipfile, collections, importlib.util, datetime as dt
from pathlib import Path
import numpy as np

# --- Settings -------------------------------------------------------------------
SOURCE_VARIANT = 'crop_lr1e-4'   # notebook 02 folder with the frozen settings
MODELS = ['effb0']               # the chosen model; add 'mnv3s' to also convert the MobileNet fallback
SHIP_MODEL = 'effb0'
SHIP_VARIANT = 'float16'         # decided 2026-11-27: same val scores as float32, half the size. CPU timing on free
                                 # Colab is too noisy to rank float16 vs float32 (they swapped between runs). None = rule only
TOLERANCE = 0.01                 # a variant may lose at most ~1 point of val acc and val damaged recall
SPEED_LIMIT_MS = 150             # Colab-CPU ms/image: slower than this -> warning (and fallback, if in MODELS)
RECALL_TARGET = 0.95             # shipped threshold: largest one with >= 95% damaged recall on val
N_REPRESENTATIVE = 200           # train images used to calibrate full-int8 quantization
TIMING_WARMUP, TIMING_RUNS = 5, 50
RUN_WEB_CHECK = True             # Node runs preprocess.js on the test vectors; its output goes through the shipped .tflite
EXPECTED_MANIFEST_ROWS = 1123

# --- Where things live (same as notebooks 01-02) ---------------------------------
try:
    import google.colab  # noqa: F401
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

if IN_COLAB:
    from google.colab import drive
    drive.mount('/content/drive')
    PROJECT = Path('/content/drive/MyDrive/seventx_prototype')  # change if you put the folder elsewhere
    ROOT = Path('/content/seventx')
    WORK = Path('/content/export_work')                         # big temporary files stay off Drive
    if not (ROOT / 'dataset').exists():
        if (PROJECT / 'dataset.zip').exists():
            with zipfile.ZipFile(PROJECT / 'dataset.zip') as z:
                z.extractall(ROOT)
        else:
            shutil.copytree(PROJECT / 'dataset', ROOT / 'dataset')
else:
    PROJECT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()
    ROOT = PROJECT
    WORK = PROJECT / 'export_work'

SRC_MODELS = PROJECT / 'models' / SOURCE_VARIANT
SRC_RESULTS = PROJECT / 'results' / SOURCE_VARIANT
EXPORT = PROJECT / 'models' / 'export'
for d in (EXPORT / 'tflite_variants', EXPORT / 'test_vectors', WORK):
    d.mkdir(parents=True, exist_ok=True)
print('reading', SRC_MODELS, '| writing', EXPORT)
""")

code(r"""
import tensorflow as tf
import keras
tf.get_logger().setLevel('ERROR')
print('TensorFlow', tf.__version__, '| Keras', keras.__version__, '| GPU:', bool(tf.config.list_physical_devices('GPU')))
""")

md(r"""
## 1. Preprocessing: one file for training, export and the apps

`preprocess.py` (the reference) and `preprocess.js` (the web port) are written to `models/export/`, and this notebook imports the Python one from there. Their source lives in the project at `notebooks/lib/`. If you change them, regenerate this notebook. Local checks already confirmed two things:
- `preprocess.fit_pad` is byte-identical to notebook 02's on all 1,123 images.
- The JS port gives the same crop boxes, with pixels within 2 levels.
""")

code("PREPROCESS_PY = r'''" + PY_SRC + "'''\n\nPREPROCESS_JS = r'''" + JS_SRC + "'''\n\n" + r"""
(EXPORT / 'preprocess.py').write_text(PREPROCESS_PY, encoding='utf-8')
(EXPORT / 'preprocess.js').write_text(PREPROCESS_JS, encoding='utf-8')
import sys
sys.dont_write_bytecode = True                      # no __pycache__ in the export folder
shutil.rmtree(EXPORT / '__pycache__', ignore_errors=True)
spec = importlib.util.spec_from_file_location('preprocess', EXPORT / 'preprocess.py')
pp = importlib.util.module_from_spec(spec); spec.loader.exec_module(pp)
print('preprocess.py and preprocess.js written; crop margin', pp.MARGIN, '| input size', pp.SIZE)
""")

md("## 2. Data: val, test and calibration images")

code(r"""
with open(PROJECT / 'data' / 'manifest.csv', newline='') as f:
    M = list(csv.DictReader(f))
assert len(M) == EXPECTED_MANIFEST_ROWS, f'manifest has {len(M)} rows, expected {EXPECTED_MANIFEST_ROWS}'
for r in M:
    r['y'] = 1 if r['label'] == 'damaged' else 0
    r['multi_berry'] = int(r['multi_berry'])

VAL = [i for i, r in enumerate(M) if r['split'] == 'val']
TEST = [i for i, r in enumerate(M) if r['split'] == 'test']
REP = sorted(random.Random(0).sample([i for i, r in enumerate(M) if r['split'] == 'train'], N_REPRESENTATIVE))
t0 = time.time()
ARR = {i: pp.fit_pad(ROOT / M[i]['path']) for i in VAL + TEST + REP}   # exactly as in training
print(f'{len(VAL)} val, {len(TEST)} test, {len(REP)} calibration images preprocessed in {time.time() - t0:.0f} s')
assert (len(VAL), len(TEST)) == (165, 161)

def batch(ix):
    return np.stack([ARR[i] for i in ix]).astype(np.float32)
""")

md(r"""
## 3. Which trained model to export

For each model, the E1 seed with the best **val** accuracy (from notebook 02's `runs.csv`), tie-broken by the lowest val loss.
""")

code(r"""
with open(SRC_RESULTS / 'runs.csv', newline='') as f:
    RUNS = list(csv.DictReader(f))
SEED = {}
for name in MODELS:
    cands = []
    for r in RUNS:
        if (r['exp'], r['model'], r['eval_on'], r['subset'], r['threshold']) == ('E1', name, 'val', 'all', '0.5'):
            hist = json.loads((SRC_RESULTS / 'history' / f"E1_{name}_s{r['seed']}.json").read_text())
            cands.append((float(r['acc']), -hist['best_val_loss'], int(r['seed'])))
    cands.sort(reverse=True)
    SEED[name] = cands[0][2]
    print(name, 'seeds (val acc, val loss):', [(s, round(a, 4), round(-l, 4)) for a, l, s in cands], '-> seed', SEED[name])
""")

md(r"""
## 4. Parity check: does the exported preprocessing match training?

Predict val and test with the float32 Keras model on images from `preprocess.py`, and compare with notebook 02's saved predictions.
""")

code(r"""
def read_probs(path):
    with open(path, newline='') as f:
        return {r['path']: float(r['prob_damaged']) for r in csv.DictReader(f)}

KERAS, P_KERAS = {}, {}
for name in MODELS:
    tag = f'E1_{name}_s{SEED[name]}'
    KERAS[name] = keras.models.load_model(str(SRC_MODELS / f'{tag}.keras'))
    worst = 0.0
    for part, ix, f in (('val', VAL, f'{tag}_val.csv'), ('test', TEST, f'{tag}.csv')):
        p = KERAS[name].predict(batch(ix), verbose=0).ravel()
        P_KERAS[(name, part)] = p
        saved = read_probs(SRC_RESULTS / 'preds' / f)
        worst = max(worst, max(abs(p[k] - saved[M[i]['path']]) for k, i in enumerate(ix)))
    print(f'{name}: max |dp| vs notebook 02 = {worst:.2e}')
    assert worst < 1e-3, 'preprocessing or model differs from training'
print('parity OK: export preprocessing = training preprocessing')
""")

md(r"""
## 5. TFLite conversion

Four variants. Input and output stay float32 in all of them, so the app contract never changes.

| Variant | How |
| --- | --- |
| `float32` | no optimisation (reference) |
| `float16` | weights stored as float16 |
| `dynamic_int8` | weights int8, activations float |
| `int8` | weights and activations int8, calibrated on 200 train images; ops without int8 kernels stay float |
""")

code(r"""
VARIANTS = ['float32', 'float16', 'dynamic_int8', 'int8']

def representative():
    for i in REP:
        yield [ARR[i][None].astype(np.float32)]

def convert(saved_model_dir, variant):
    c = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
    if variant == 'float16':
        c.optimizations = [tf.lite.Optimize.DEFAULT]
        c.target_spec.supported_types = [tf.float16]
    elif variant == 'dynamic_int8':
        c.optimizations = [tf.lite.Optimize.DEFAULT]
    elif variant == 'int8':
        c.optimizations = [tf.lite.Optimize.DEFAULT]
        c.representative_dataset = representative
    return c.convert()

TFLITE = {}
for name in MODELS:
    sm = WORK / f'savedmodel_{name}'
    if sm.exists():
        shutil.rmtree(sm)
    KERAS[name].export(str(sm))
    for v in VARIANTS:
        path = EXPORT / 'tflite_variants' / f'{name}_{v}.tflite'
        try:
            t0 = time.time()
            path.write_bytes(convert(sm, v))
            TFLITE[(name, v)] = path
            print(f'{name:6s} {v:13s} {path.stat().st_size / 1e6:6.2f} MB  ({time.time() - t0:.0f} s)')
        except Exception as e:
            print(f'{name:6s} {v:13s} FAILED: {type(e).__name__}: {str(e)[:200]}')
""")

md(r"""
## 6. Accuracy and speed of every variant

`tf.lite.Interpreter` on the CPU, one image at a time, as an app does. Speed is the median of 50 runs after 5 warm-up runs. Free Colab has only 2 CPU cores, so treat ms/image as a rough reference; phone timing comes from the apps.
""")

code(r"""
def tflite_predict(path, x):
    it = tf.lite.Interpreter(model_path=str(path), num_threads=4)
    it.allocate_tensors()
    inp, out = it.get_input_details()[0], it.get_output_details()[0]
    probs = np.empty(len(x), dtype=np.float32)
    for k in range(len(x)):
        it.set_tensor(inp['index'], x[k:k + 1])
        it.invoke()
        probs[k] = it.get_tensor(out['index']).ravel()[0]
    times = []
    for k in range(TIMING_WARMUP + TIMING_RUNS):
        it.set_tensor(inp['index'], x[k % len(x):k % len(x) + 1])
        t0 = time.perf_counter(); it.invoke(); times.append(time.perf_counter() - t0)
    return probs, 1000 * float(np.median(times[TIMING_WARMUP:]))

def scores(y, p, thr):
    pred = (p >= thr).astype(int)
    tp, fn = int(((pred == 1) & (y == 1)).sum()), int(((pred == 0) & (y == 1)).sum())
    tn, fp = int(((pred == 0) & (y == 0)).sum()), int(((pred == 1) & (y == 0)).sum())
    div = lambda a, b: a / b if b else float('nan')
    return dict(acc=div(tp + tn, len(y)), damaged_recall=div(tp, tp + fn), good_precision=div(tn, tn + fn),
                damaged_precision=div(tp, tp + fp), tp=tp, fn=fn, tn=tn, fp=fp)

def recall_threshold(y, p, target=RECALL_TARGET):
    # Largest threshold that still gives damaged recall >= target (not capped at 0.5 here).
    d = np.sort(p[y == 1])
    return float(d[int(math.floor((1 - target) * len(d)))])

Y = {'val': np.array([M[i]['y'] for i in VAL]), 'test': np.array([M[i]['y'] for i in TEST])}
X = {'val': batch(VAL), 'test': batch(TEST)}
ROWS, P_TFLITE = [], {}
for (name, v), path in TFLITE.items():
    pv, ms = tflite_predict(path, X['val'])
    pt, _ = tflite_predict(path, X['test'])
    assert len(pv) == 165 and len(pt) == 161
    P_TFLITE[(name, v, 'val')], P_TFLITE[(name, v, 'test')] = pv, pt
    if v == 'float32':
        d = float(np.abs(pv - P_KERAS[(name, 'val')]).max())
        print(f'{name} float32 TFLite vs Keras: max |dp| = {d:.2e}')
        assert d < 1e-3, 'float32 TFLite does not match Keras'
    thr = recall_threshold(Y['val'], pv)
    row = dict(model=name, seed=SEED[name], variant=v, mb=round(path.stat().st_size / 1e6, 2), ms=round(ms, 1), threshold=round(thr, 4))
    for part, p in (('val', pv), ('test', pt)):
        for tname, t in (('05', 0.5), ('thr', thr)):
            for k, val in scores(Y[part], p, t).items():
                row[f'{part}_{tname}_{k}'] = val
    ROWS.append(row)
    print(f"{name:6s} {v:13s} {row['mb']:6.2f} MB {ms:7.1f} ms/img | val acc {row['val_05_acc']:.3f}, damaged recall {row['val_05_damaged_recall']:.3f}")

with open(SRC_RESULTS / 'export_runs.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, list(ROWS[0])); w.writeheader(); w.writerows(ROWS)
""")

md(r"""
## 7. Choose what ships (val only)

**`SHIP_VARIANT` (float16)** if its val accuracy **and** val damaged recall at threshold 0.5 are within `TOLERANCE` of float32. Otherwise, the rule: among the variants that pass, the **fastest**; ties go to the smaller file. (Why it's pinned: free Colab's CPU timings are too noisy to rank variants that run at the same speed. Float16 and float32 swapped places between two runs, and the second run shipped float32 at 16 MB.) If none passes, float16. (The first run used "smallest" and picked dynamic int8: 4.6 MB but 4× slower than float16 at 92 ms, and with shifted predictions. Float16 is 8.1 MB and gives the same answers as float32.) If the model is slower than `SPEED_LIMIT_MS`, the notebook warns; with `'mnv3s'` added to `MODELS` it would ship MobileNet instead. The threshold is the chosen variant's val-tuned one.
""")

code(r"""
def choose_variant(name):
    rows = {r['variant']: r for r in ROWS if r['model'] == name}
    ref = rows['float32']
    ok = [r for r in rows.values()
          if r['val_05_acc'] >= ref['val_05_acc'] - TOLERANCE and r['val_05_damaged_recall'] >= ref['val_05_damaged_recall'] - TOLERANCE]
    if SHIP_VARIANT and SHIP_VARIANT in [r['variant'] for r in ok]:
        return rows[SHIP_VARIANT]
    if SHIP_VARIANT:
        print(f'WARNING: {SHIP_VARIANT} is not within {TOLERANCE * 100:.0f} point(s) of float32 on val; using the rule instead')
    return min(ok, key=lambda r: (r['ms'], r['mb'])) if ok else rows.get('float16', ref)

CHOICE = {name: choose_variant(name) for name in MODELS}
ship = CHOICE[SHIP_MODEL]
why = (f'{SHIP_MODEL} {ship["variant"]}: ' + ('SHIP_VARIANT, checked within ' if ship['variant'] == SHIP_VARIANT else 'fastest variant within ')
       + f'{TOLERANCE * 100:.0f} point(s) of float32 on val')
if ship['ms'] > SPEED_LIMIT_MS:
    fallback = next((n for n in MODELS if n != SHIP_MODEL), None)
    if fallback:
        why = f'{SHIP_MODEL} too slow ({ship["ms"]} ms > {SPEED_LIMIT_MS} ms): shipping {fallback} {CHOICE[fallback]["variant"]}'
        ship = CHOICE[fallback]
    else:
        print(f'WARNING: {SHIP_MODEL} takes {ship["ms"]} ms/image on this CPU (limit {SPEED_LIMIT_MS}). Shipping it anyway; '
              "measure on the phone. To compare with MobileNet, add 'mnv3s' to MODELS and rerun.")
SHIPPED = ship
THRESHOLD = ship['threshold']
shutil.copy(TFLITE[(ship['model'], ship['variant'])], EXPORT / 'strawberry_2class.tflite')
print('SHIPPED:', why)
print(f"  {ship['mb']} MB, {ship['ms']} ms/image (Colab CPU), threshold {THRESHOLD:.3f}")
print(f"  val @0.5: acc {ship['val_05_acc']:.3f}, damaged recall {ship['val_05_damaged_recall']:.3f}")
print(f"  val @{THRESHOLD:.3f}: acc {ship['val_thr_acc']:.3f}, damaged recall {ship['val_thr_damaged_recall']:.3f}, good precision {ship['val_thr_good_precision']:.3f}")
""")

code(r"""
# Test scores of the shipped model: printed AFTER the choice, for reporting only.
s = SHIPPED
print(f"TEST (161 images) @0.5:   acc {s['test_05_acc']:.3f}, damaged recall {s['test_05_damaged_recall']:.3f}, good precision {s['test_05_good_precision']:.3f}")
print(f"TEST @{THRESHOLD:.3f}: acc {s['test_thr_acc']:.3f}, damaged recall {s['test_thr_damaged_recall']:.3f}, good precision {s['test_thr_good_precision']:.3f}")
print(f"confusion @{THRESHOLD:.3f}: TN {s['test_thr_tn']}  FP {s['test_thr_fp']}  FN {s['test_thr_fn']}  TP {s['test_thr_tp']}")
""")

md(r"""
## 8. Test vectors for the app ports

Ten **Mendeley** photos (CC BY 4.0; Mahendra's GPL photos must not be redistributed): fresh singles, fresh piles, a crate, a dark mouldy berry, a mashed berry, rotten piles and a rotten single. For each one: the crop box, the 224×224 output's mean and std, and P(damaged) from the float32 model and the shipped `.tflite`, all on a **full-size decode**, as a phone or browser does. A port passes if its boxes agree within 1% of the photo's longer side and its P is within 0.02.
""")

code(r"""
import re
num = lambda r: int(re.search(r'\((\d+)\)', r['path']).group(1))
MEN = sorted((r for r in M if r['source'] == 'mendeley'), key=lambda r: (r['label'], num(r)))
want = [  # (description, condition, how many)
    ('fresh single', lambda r: r['label'] == 'good' and r['multi_berry'] == 0 and not 173 <= num(r) <= 177, 2),
    ('fresh pile', lambda r: r['label'] == 'good' and r['multi_berry'] == 1 and not 173 <= num(r) <= 177, 2),
    ('fresh crate', lambda r: r['label'] == 'good' and 173 <= num(r) <= 177, 1),
    ('rotten dark mouldy', lambda r: r['label'] == 'damaged' and 169 <= num(r) <= 197, 1),
    ('rotten mashed', lambda r: r['label'] == 'damaged' and 151 <= num(r) <= 166, 1),
    ('rotten pile', lambda r: r['label'] == 'damaged' and r['multi_berry'] == 1, 2),
    ('rotten single', lambda r: r['label'] == 'damaged' and r['multi_berry'] == 0 and num(r) < 151, 1),
]
rng = random.Random(1)
ship_keras = KERAS[SHIPPED['model']]
it = tf.lite.Interpreter(model_path=str(EXPORT / 'strawberry_2class.tflite')); it.allocate_tensors()
inp, outp = it.get_input_details()[0], it.get_output_details()[0]
VECTORS = []
for desc, cond, n in want:
    for r in rng.sample([r for r in MEN if cond(r)], n):
        src = ROOT / r['path']
        name = src.name.replace(' ', '_').replace('(', '').replace(')', '')
        shutil.copy(src, EXPORT / 'test_vectors' / name)
        im = pp.load_image(src)                       # full-size decode, upright
        arr, box = pp.fit_pad_image(im)
        x = arr.astype(np.float32)[None]
        p_keras = float(ship_keras.predict(x, verbose=0).ravel()[0])
        it.set_tensor(inp['index'], x); it.invoke()
        p_tfl = float(it.get_tensor(outp['index']).ravel()[0])
        p_train = float(ship_keras.predict(pp.fit_pad(src).astype(np.float32)[None], verbose=0).ravel()[0])
        VECTORS.append(dict(file=name, description=desc, label=r['label'], width=im.width, height=im.height,
                            box=list(box) if box else None, mean=round(float(arr.mean()), 3), std=round(float(arr.std()), 3),
                            p_damaged_float32=round(p_keras, 5), p_damaged_tflite=round(p_tfl, 5)))
        print(f'{desc:20s} {name:28s} box {box}  P float32 {p_keras:.3f} | tflite {p_tfl:.3f} | training-style decode {p_train:.3f}')
(EXPORT / 'test_vectors' / 'test_vectors.json').write_text(json.dumps(dict(
    pass_criteria=dict(box_tolerance='1% of the longer image side', p_tolerance=0.02),
    note='Decode at full size and apply EXIF rotation; feed RGB 0-255 float32, no /255.',
    vectors=VECTORS), indent=1))
print(len(VECTORS), 'test vectors written')
""")

md("## 9. `labels.txt` and `metadata.json`")

code(r"""
(EXPORT / 'labels.txt').write_text('good\ndamaged\n')

def run_score(model, exp):
    for r in RUNS:
        if (r['exp'], r['model'], r['eval_on'], r['subset'], r['threshold'], r['seed']) == (exp, model, 'test', 'all', '0.5', '42'):
            return round(float(r['acc']), 4)
    return None

s = SHIPPED
META = {
    'name': 'strawberry_2class', 'version': dt.date.today().isoformat(),
    'model': {'architecture': {'effb0': 'EfficientNet-B0', 'mnv3s': 'MobileNetV3-Small'}[s['model']],
              'trained_run': f"E1_{s['model']}_s{s['seed']} ({SOURCE_VARIANT})", 'tflite_variant': s['variant'],
              'file': 'strawberry_2class.tflite', 'size_mb': s['mb'], 'tf': tf.__version__, 'keras': keras.__version__},
    'input': {'shape': [1, pp.SIZE, pp.SIZE, 3], 'dtype': 'float32', 'color': 'RGB', 'range': [0, 255],
              'normalisation': 'none (do NOT divide by 255; the model rescales internally)',
              'preprocessing': 'preprocess.py / preprocess.js: EXIF-upright, crop to berry, fit inside 224x224 (shrink only), pad with median border colour'},
    'output': {'shape': [1, 1], 'meaning': 'P(damaged)'},
    'labels': ['good', 'damaged'],
    'threshold': {'value': THRESHOLD, 'rule': f'damaged if P >= threshold; largest threshold with >= {RECALL_TARGET:.0%} damaged recall on the validation set (165 images)'},
    'crop': {'mask_size': pp.MASK_SIZE, 'diff_min': pp.DIFF_MIN, 'sat_min': pp.SAT_MIN, 'line_frac': pp.LINE_FRAC,
             'percentiles': list(pp.PCT), 'margin': pp.MARGIN, 'min_pixels': pp.MIN_PIXELS, 'size': pp.SIZE},
    'scores': {
        'val_at_0.5': {k: round(s[f'val_05_{k}'], 4) for k in ('acc', 'damaged_recall', 'good_precision')},
        'val_at_threshold': {k: round(s[f'val_thr_{k}'], 4) for k in ('acc', 'damaged_recall', 'good_precision')},
        'test_at_0.5': {k: round(s[f'test_05_{k}'], 4) for k in ('acc', 'damaged_recall', 'good_precision')},
        'test_at_threshold': {k: round(s[f'test_thr_{k}'], 4) for k in ('acc', 'damaged_recall', 'good_precision')},
        'cross_dataset_float32_seed42': {'E2a_train_mendeley_test_mahendra_acc': run_score(s['model'], 'E2a'),
                                         'E2b_train_mahendra_test_mendeley_acc': run_score(s['model'], 'E2b')},
        'warning': 'Accuracy drops on photos from a new camera or setting (cross-dataset 63-75%). Collect and fine-tune on local photos before relying on it.',
    },
    'speed_ms_colab_cpu': s['ms'],
    'runtimes': {'android': 'TensorFlow Lite / LiteRT Interpreter, strawberry_2class.tflite',
                 'web': 'the same strawberry_2class.tflite with the LiteRT / TFLite web (WebAssembly) runtime; preprocess.js for the crop'},
    'attribution': [
        'Fresh and Rotten Fruits Dataset, Mendeley Data, DOI 10.17632/bdd69gyhv8.1, CC BY 4.0',
        'Mahendra et al. 2018 strawberry dataset (GPL), https://github.com/okamahendra123/dataset_strawberry',
    ],
}
(EXPORT / 'metadata.json').write_text(json.dumps(META, indent=1))
for key in ('model', 'input', 'output', 'labels', 'threshold', 'crop', 'scores', 'attribution'):
    assert key in json.loads((EXPORT / 'metadata.json').read_text())
print(json.dumps(META, indent=1)[:1500], '...')
""")

md(r"""
## 10. Web demo check

The web demo runs the **same** `strawberry_2class.tflite` in the browser with Google's LiteRT/TFLite web runtime (WebAssembly), so both demos share one model file. No TF.js conversion is needed; it failed on Colab's Python 3.13 anyway.

This cell checks the part the web demo adds, `preprocess.js`. Node runs it on the test-vector photos, then its 224×224 output goes through the shipped `.tflite` here and is compared with the Python reference:
- crop boxes within 1% of the photo's longer side;
- P(damaged) within 0.02.

Running the browser runtime itself is checked in the browser (see `export.md`).
""")

code(r"""
def sh(cmd, timeout=900):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    return r.returncode, (r.stdout + r.stderr)[-3000:]

WEB_CHECK = 'skipped (RUN_WEB_CHECK = False)'
if RUN_WEB_CHECK:
    if shutil.which('node') is None:
        WEB_CHECK = 'skipped: node not found on this machine'
    else:
        chk = WORK / 'web_check'
        chk.mkdir(exist_ok=True)
        cases = []
        for v in VECTORS:   # full-size decode in Python, so only preprocess.js is being tested
            im = pp.load_image(EXPORT / 'test_vectors' / v['file'])
            raw = chk / (v['file'] + '.rgb')
            raw.write_bytes(np.asarray(im).tobytes())
            cases.append(dict(file=v['file'], w=im.width, h=im.height, raw=str(raw)))
        (chk / 'cases.json').write_text(json.dumps(cases))
        (chk / 'run.js').write_text(r'''
const fs = require('fs'); const pp = require(process.argv[2]);
const out = [];
for (const c of JSON.parse(fs.readFileSync(process.argv[3]))) {
  const r = pp.preprocess({width: c.w, height: c.h, data: fs.readFileSync(c.raw)});
  fs.writeFileSync(c.raw + '.js.bin', Buffer.from(r.pixels));
  out.push({file: c.file, box: r.box});
}
console.log(JSON.stringify(out));
''')
        code_, log = sh(f"node {chk / 'run.js'} {EXPORT / 'preprocess.js'} {chk / 'cases.json'}")
        if code_ != 0:
            WEB_CHECK = 'node failed: ' + log[-500:]
        else:
            res = {o['file']: o for o in json.loads(log.strip().splitlines()[-1])}
            it = tf.lite.Interpreter(model_path=str(EXPORT / 'strawberry_2class.tflite')); it.allocate_tensors()
            inp, outp = it.get_input_details()[0], it.get_output_details()[0]
            worst_p, worst_box = 0.0, 0.0
            for v, c in zip(VECTORS, cases):
                px = np.frombuffer(Path(c['raw'] + '.js.bin').read_bytes(), dtype=np.uint8).reshape(pp.SIZE, pp.SIZE, 3)
                it.set_tensor(inp['index'], px.astype(np.float32)[None]); it.invoke()
                p = float(it.get_tensor(outp['index']).ravel()[0])
                worst_p = max(worst_p, abs(p - v['p_damaged_tflite']))
                jb = res[v['file']]['box']
                if v['box'] and jb:
                    worst_box = max(worst_box, max(abs(a - b) for a, b in zip(v['box'], jb)) / max(v['width'], v['height']))
                elif v['box'] != jb:
                    worst_box = 1.0
            WEB_CHECK = f'preprocess.js + shipped .tflite: max |dp| {worst_p:.4f} (limit 0.02), max box diff {100 * worst_box:.2f}% of side (limit 1%)'
            assert worst_p < 0.02 and worst_box <= 0.01, WEB_CHECK
        shutil.rmtree(chk, ignore_errors=True)   # the raw files are large
print('Web check:', WEB_CHECK)
""")

md("## 11. `export.md`")

code(r"""
pct = lambda v: f'{100 * v:.1f}'
L = ['# Export results', '',
     f'Generated by notebook 03 on {dt.date.today().isoformat()}. Choices use **val** only; test is reported after the choice.', '',
     f"**Shipped:** {SHIPPED['model']} `{SHIPPED['variant']}` from run E1_{SHIPPED['model']}_s{SHIPPED['seed']} "
     f"as `strawberry_2class.tflite`: {SHIPPED['mb']} MB, {SHIPPED['ms']} ms/image on the Colab CPU, threshold **{THRESHOLD:.3f}**.", '',
     '| Model | Variant | MB | ms/img | val acc | val damaged recall | Δ acc vs float32 | test acc | test damaged recall |',
     '| --- | --- | --- | --- | --- | --- | --- | --- | --- |']
for r in ROWS:
    ref = next(x for x in ROWS if x['model'] == r['model'] and x['variant'] == 'float32')
    L.append(f"| {r['model']} | {r['variant']} | {r['mb']} | {r['ms']} | {pct(r['val_05_acc'])} | {pct(r['val_05_damaged_recall'])} | "
             f"{100 * (r['val_05_acc'] - ref['val_05_acc']):+.1f} | {pct(r['test_05_acc'])} | {pct(r['test_05_damaged_recall'])} |")
s = SHIPPED
L += ['', 'Threshold 0.5 in the table. The shipped model at its val-tuned threshold:', '',
      '| | threshold | accuracy | damaged recall | good precision |', '| --- | --- | --- | --- | --- |',
      f"| val | {THRESHOLD:.3f} | {pct(s['val_thr_acc'])} | {pct(s['val_thr_damaged_recall'])} | {pct(s['val_thr_good_precision'])} |",
      f"| test | {THRESHOLD:.3f} | {pct(s['test_thr_acc'])} | {pct(s['test_thr_damaged_recall'])} | {pct(s['test_thr_good_precision'])} |", '',
      '## Checks', '',
      '- Preprocessing parity with training: max |Δp| < 1e-3 (asserted).',
      '- float32 TFLite vs Keras: max |Δp| < 1e-3 (asserted).',
      f'- Web check: {WEB_CHECK}.', '',
      'Both demos use `strawberry_2class.tflite`. In the browser, load it with the LiteRT / TFLite web runtime and check the test vectors there once.', '',
      'Speed is a Colab-CPU reference (2 cores); measure on the phone and in the browser for real numbers.']
(SRC_RESULTS / 'export.md').write_text('\n'.join(L) + '\n', encoding='utf-8')
print('\n'.join(L))
print('\nFiles in', EXPORT, ':', sorted(p.name for p in EXPORT.iterdir()))
print('Next: download models/export/ from Drive into the project folder for the app builders.')
""")

nb = nbf.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}, "accelerator": "GPU"})
nbf.write(nb, str(out))
print('wrote', out)
