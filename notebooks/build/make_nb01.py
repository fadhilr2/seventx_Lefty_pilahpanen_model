"""Generates notebooks/01_manifest_and_split.ipynb."""
import sys
import nbformat as nbf

out = sys.argv[1]
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md(r"""
# 01 · Manifest and split

Builds `data/manifest.csv`, the single file every later experiment reads. Design and reasons: `plans/01_data_and_splits.md`. In short:

1. **Mendeley images 201–737 are dropped.** In both `FreshStrawberry/` and `RottenStrawberry/`, images 1–200 are the original photos and 201–737 are rotated, 1020×1020 copies with black corners. They are augmentations of the originals, so keeping them would put copies of training photos in the test set. We do our own augmentation in training.
2. **Exact duplicates (MD5) are dropped.** Near-duplicates (perceptual hash of a crop around the berry) are kept, but forced into the same group so they land in the same split. This matters: e.g. Fresh 190–200 are re-shots of the same berries as 37–47.
3. **Images are split by group, not one by one.** The same berry (or pile) was photographed several times in a row:
   - Mahendra: photos of one label taken ≤ 30 s apart form one burst (one group).
   - Mendeley: hand-marked **photo series** (runs of file numbers showing the same berry or pile) form one group. Fixed blocks of 10 were tried first, but a series often crossed a block edge, e.g. rotten 10 (train) and 11–13 (test) are the same berry.

   This is deliberately cautious: it can merge different berries into one group, but it shouldn't split one berry across train and test.
4. **`multi_berry`**: 1 if the photo shows more than one berry. Tagged by hand from contact sheets (Mendeley originals). All Mahendra photos are single berries. Train on everything; report test scores on all images and on single-berry images only.
5. **Split 70 / 15 / 15** by group, stratified by source × label, seed 42.
6. **Purge.** After the split, any val or test image whose berry-crop hash is within 8 bits of an image in an earlier split (same source and label) is removed (`excluded.csv`). These are re-shots the grouping missed, e.g. a Mahendra berry photographed again 25 minutes later. Merging at 8 bits instead would chain unrelated photos into huge groups, so they are dropped, not merged.

Needs only Pillow and NumPy. Runs locally or on Google Colab. On Colab, put the `seventx_prototype` folder in `MyDrive` with `dataset.zip` inside (or the `dataset/` folder). The first cell mounts Drive, copies the images to `/content`, and writes outputs back to Drive.
""")

code(r"""
import csv, hashlib, re, random, collections, datetime as dt, shutil, zipfile
from pathlib import Path
import numpy as np
from PIL import Image

# --- Where things live -------------------------------------------------------
# PROJECT: the seventx_prototype folder. Outputs (data/, models/, results/) are written here.
#          On Colab this is your Google Drive, so outputs survive the session.
# ROOT:    where the images are read from. Paths in manifest.csv are relative to ROOT.
#          On Colab this is a local copy in /content, because reading images from Drive is slow.
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
    if not (ROOT / 'dataset').exists():
        if (PROJECT / 'dataset.zip').exists():   # fastest: one zip file on Drive
            with zipfile.ZipFile(PROJECT / 'dataset.zip') as z:
                z.extractall(ROOT)
        else:                                    # fallback: the dataset/ folder on Drive
            shutil.copytree(PROJECT / 'dataset', ROOT / 'dataset')
else:
    PROJECT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()
    ROOT = PROJECT

DATA = ROOT / 'dataset'
OUT = PROJECT / 'data'
OUT.mkdir(exist_ok=True)

SEED = 42
SPLIT = {'train': 0.70, 'val': 0.15, 'test': 0.15}
MAHENDRA_BURST_GAP_S = 30   # photos of one label closer than this = same group
MENDELEY_MAX_ORIGINAL = 200 # 201-737 are augmented copies
PHASH_SAME_GROUP = 6        # Hamming distance (of 64) to merge groups, same source and label only
PHASH_FLAG = 6              # closer than this across source/label = printed for a manual look
PHASH_PURGE = 8             # val/test image this close to an earlier split (same source and label) = removed
print('images from', DATA, '| outputs to', OUT)
""")

md(r"""
## Hand-made tags: Mendeley photos with more than one berry

From looking at every original (1–200) in each folder. Edit here if a tag is wrong.
""")

code(r"""
def ranges(*rs):
    s = set()
    for r in rs:
        a, b = (r, r) if isinstance(r, int) else r
        s.update(range(a, b + 1))
    return s

# Fresh: 88-150 are small piles, 173-177 are crates full of berries.
MULTI_FRESH = ranges((88, 150), (173, 177))
# Rotten: everything is a pile except these single-berry photos.
SINGLE_ROTTEN = ranges(1, 5, (10, 13), 21, (50, 52), (84, 87), 90, 92, 93, (95, 101), (151, 166), (169, 197))
MULTI_ROTTEN = set(range(1, MENDELEY_MAX_ORIGINAL + 1)) - SINGLE_ROTTEN
print(len(MULTI_FRESH), 'fresh multi;', len(MULTI_ROTTEN), 'rotten multi')

# Photo series: runs of consecutive file numbers that show the same berry or pile,
# as (start, end), from the same contact sheets. Where unsure, neighbouring runs are merged.
MENDELEY_SERIES = {
    'good': [(1, 8), (9, 33), (34, 41), (42, 69), (70, 87), (88, 113), (114, 127), (128, 150),
             (151, 160), (161, 172), (173, 177), (178, 194), (195, 200)],
    'damaged': [(1, 1), (2, 9), (10, 13), (14, 22), (23, 29), (30, 49), (50, 52), (53, 83), (84, 91),
                (92, 101), (102, 110), (111, 124), (125, 130), (131, 150), (151, 168), (169, 197), (198, 200)],
}
for label, series in MENDELEY_SERIES.items():  # every file number 1-200 in exactly one series
    covered = [n for a, b in series for n in range(a, b + 1)]
    assert sorted(covered) == list(range(1, MENDELEY_MAX_ORIGINAL + 1)), label
""")

md("## 1. Scan the folders")

code(r"""
FOLDERS = [  # (folder, source, label)
    ('color/good', 'mahendra', 'good'),
    ('color/damaged', 'mahendra', 'damaged'),
    ('FreshStrawberry', 'mendeley', 'good'),
    ('RottenStrawberry', 'mendeley', 'damaged'),
]

def crop_berry(im):
    # Crop to the berry (or pile): pixels that differ from the median border colour.
    # Hashing the whole photo fails on Mahendra, where a small berry sits on a big white background.
    a = np.asarray(im.convert('RGB').resize((256, 256)), dtype=np.int16)
    bg = np.median(np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]]), axis=0)
    ys, xs = np.where(np.abs(a - bg).sum(-1) > 60)
    if len(ys) < 50:
        return im
    (y0, y1), (x0, x1) = np.percentile(ys, [1, 99]), np.percentile(xs, [1, 99])
    W, H = im.size
    return im.crop((int(x0 / 256 * W), int(y0 / 256 * H), int(x1 / 256 * W) + 1, int(y1 / 256 * H) + 1))

def phash(im):
    # 64-bit perceptual hash: DCT of a 32x32 grey image, sign of the 8x8 low frequencies vs their median.
    g = np.asarray(im.convert('L').resize((32, 32), Image.LANCZOS), dtype=np.float64)
    n = np.arange(32)
    D = np.cos(np.pi * (2 * n[None, :] + 1) * n[:, None] / 64)
    low = (D @ g @ D.T)[:8, :8].flatten()
    return low > np.median(low[1:])

rows, excluded = [], []
for folder, source, label in FOLDERS:
    for p in sorted((DATA / folder).glob('*.jpg')):
        rel = p.relative_to(ROOT).as_posix()
        r = dict(path=rel, source=source, label=label)
        if source == 'mendeley':
            r['num'] = int(re.search(r'\((\d+)\)', p.name).group(1))
            if r['num'] > MENDELEY_MAX_ORIGINAL:
                excluded.append(dict(path=rel, reason='augmented copy (Mendeley 201-737)'))
                continue
            multi = MULTI_FRESH if label == 'good' else MULTI_ROTTEN
            r['multi_berry'] = int(r['num'] in multi)
        else:
            r['time'] = dt.datetime.strptime(p.stem[6:25], '%Y-%m-%d_%H-%M-%S')
            r['multi_berry'] = 0
        with Image.open(p) as im:
            r['width'], r['height'] = im.size
            im.draft('RGB', (256, 256))
            r['phash_bits'] = phash(crop_berry(im))
        r['md5'] = hashlib.md5(p.read_bytes()).hexdigest()
        rows.append(r)

print(len(rows), 'kept,', len(excluded), 'excluded')
print(collections.Counter((r['source'], r['label']) for r in rows))
""")

md("## 2. Exact duplicates")

code(r"""
seen = {}
kept = []
for r in rows:
    if r['md5'] in seen:
        excluded.append(dict(path=r['path'], reason=f"exact duplicate of {seen[r['md5']]}"))
    else:
        seen[r['md5']] = r['path']
        kept.append(r)
print(len(rows) - len(kept), 'exact duplicates removed')
rows = kept
""")

md("## 3. Groups: bursts, series, near-duplicates")

code(r"""
parent = list(range(len(rows)))
def find(i):
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i
def union(i, j):
    parent[find(i)] = find(j)

# Mahendra: link consecutive photos of the same label taken close together.
for label in ('good', 'damaged'):
    idx = sorted((i for i, r in enumerate(rows) if r['source'] == 'mahendra' and r['label'] == label),
                 key=lambda i: rows[i]['time'])
    for a, b in zip(idx, idx[1:]):
        if (rows[b]['time'] - rows[a]['time']).total_seconds() <= MAHENDRA_BURST_GAP_S:
            union(a, b)

# Mendeley: hand-marked photo series.
series_of = {(label, n): k for label, series in MENDELEY_SERIES.items()
             for k, (a, b) in enumerate(series) for n in range(a, b + 1)}
members = collections.defaultdict(list)
for i, r in enumerate(rows):
    if r['source'] == 'mendeley':
        members[(r['label'], series_of[(r['label'], r['num'])])].append(i)
for m in members.values():
    for j in m[1:]:
        union(m[0], j)

# Near-duplicates anywhere: same group if same source and label, flag if not.
H = np.array([r['phash_bits'] for r in rows])
dist = (H[:, None, :] != H[None, :, :]).sum(-1)
ii, jj = np.where(np.triu(dist <= PHASH_SAME_GROUP, 1))
suspicious = []
for i, j in zip(ii, jj):
    a, b = rows[i], rows[j]
    if (a['source'], a['label']) == (b['source'], b['label']):
        union(i, j)
    elif dist[i, j] <= PHASH_FLAG:
        suspicious.append((a['path'], b['path'], int(dist[i, j])))
print(len(ii), 'similar pairs;', len(suspicious), 'very similar across source/label (check these by eye):')
for s in suspicious[:20]:
    print('  ', s)

roots = {}
for i, r in enumerate(rows):
    root = find(i)
    roots.setdefault(root, f"{r['source']}-{r['label']}-{len(roots):03d}")
    r['group'] = roots[root]

sizes = collections.Counter(r['group'] for r in rows)
for key in sorted({(r['source'], r['label']) for r in rows}):
    gs = [n for g, n in sizes.items() if g.startswith('-'.join(key) + '-')]
    print(key, f'{len(gs)} groups, images per group: median {int(np.median(gs))}, max {max(gs)}')
""")

md(r"""
## 4. Split by group

Groups are big and uneven (a Mendeley series can be 40 photos, a Mahendra session 87), so one greedy pass can miss the 70/15/15 target badly. Within each source × label stratum, we try `SPLIT_TRIES` group orders (seeded, so the result is repeatable). Each order is filled greedily: a group goes to the split furthest below its target. We keep the order whose splits are closest to the targets in **both** image count and single-berry count, so every split gets a fair mix of single berries and piles. Labels other than the stratum are never looked at.
""")

code(r"""
SPLIT_TRIES = 5000

def greedy(order, size):
    total = sum(size[g] for g in order)
    count, assign = dict.fromkeys(SPLIT, 0), {}
    for g in order:
        s = max(SPLIT, key=lambda s: SPLIT[s] * total - count[s])
        assign[g] = s
        count[s] += size[g]
    return assign

rng = random.Random(SEED)
for key in sorted({(r['source'], r['label']) for r in rows}):
    members = collections.defaultdict(list)
    for r in rows:
        if (r['source'], r['label']) == key:
            members[r['group']].append(r)
    size = {g: len(v) for g, v in members.items()}
    single = {g: sum(1 for r in v if r['multi_berry'] == 0) for g, v in members.items()}
    total, total_single = sum(size.values()), sum(single.values())

    def score(assign):
        err = 0.0
        for s, frac in SPLIT.items():
            n = sum(size[g] for g, a in assign.items() if a == s)
            k = sum(single[g] for g, a in assign.items() if a == s)
            err += ((n - frac * total) / total) ** 2 + ((k - frac * total_single) / max(total_single, 1)) ** 2
        return err

    best = None
    for _ in range(SPLIT_TRIES):
        order = sorted(members)
        rng.shuffle(order)
        assign = greedy(order, size)
        if best is None or score(assign) < score(best):
            best = assign
    for g, s in best.items():
        for r in members[g]:
            r['split'] = s
    n = {s: sum(size[g] for g, a in best.items() if a == s) for s in SPLIT}
    k = {s: sum(single[g] for g, a in best.items() if a == s) for s in SPLIT}
    print(key, f'{len(members)} groups |', {s: f'{n[s]} ({n[s] / total:.0%}), {k[s]} single' for s in SPLIT})
""")

md(r"""
## 5. Purge near-duplicates across splits

Val is checked against train; test against train and val. Only within the same source and label: across labels, similar hashes are different berries.
""")

code(r"""
def purge(target, against):
    keep, removed = [], []
    ref = [r for r in rows if r['split'] in against]
    for r in rows:
        if r['split'] != target:
            keep.append(r)
            continue
        close = [q for q in ref if (q['source'], q['label']) == (r['source'], r['label'])
                 and int((q['phash_bits'] != r['phash_bits']).sum()) <= PHASH_PURGE]
        if close:
            excluded.append(dict(path=r['path'], reason=f"near-duplicate of {close[0]['split']} image {close[0]['path']}"))
            removed.append(r)
        else:
            keep.append(r)
    return keep, removed

rows, rm_val = purge('val', {'train'})
rows, rm_test = purge('test', {'train', 'val'})
print(f'removed {len(rm_val)} val and {len(rm_test)} test images:')
for r in rm_val + rm_test:
    print('  ', r['split'], r['path'])
""")

md("## 6. Write `data/manifest.csv` and `data/excluded.csv`")

code(r"""
COLS = ['path', 'source', 'label', 'split', 'group', 'multi_berry', 'width', 'height', 'md5', 'phash']
for r in rows:
    r['phash'] = f"{int(''.join('1' if b else '0' for b in r['phash_bits']), 2):016x}"
rows.sort(key=lambda r: r['path'])
with open(OUT / 'manifest.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, COLS, extrasaction='ignore')
    w.writeheader()
    w.writerows(rows)
with open(OUT / 'excluded.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, ['path', 'reason'])
    w.writeheader()
    w.writerows(sorted(excluded, key=lambda r: r['path']))
print('wrote', OUT / 'manifest.csv', len(rows), 'rows;', OUT / 'excluded.csv', len(excluded), 'rows')
""")

md("## 7. Checks")

code(r"""
# No group may appear in two splits.
by_group = collections.defaultdict(set)
for r in rows:
    by_group[r['group']].add(r['split'])
assert all(len(s) == 1 for s in by_group.values()), 'a group spans splits'

# No exact copy in two splits.
by_md5 = collections.defaultdict(set)
for r in rows:
    by_md5[r['md5']].add(r['split'])
assert all(len(s) == 1 for s in by_md5.values()), 'an image is in two splits'

# No val/test image within PHASH_PURGE bits of an earlier split (same source and label).
order = {'train': 0, 'val': 1, 'test': 2}
Hb = np.array([r['phash_bits'] for r in rows])
D = (Hb[:, None, :] != Hb[None, :, :]).sum(-1)
for i, a in enumerate(rows):
    for j, b in enumerate(rows):
        if order[a['split']] > order[b['split']] and (a['source'], a['label']) == (b['source'], b['label']):
            assert D[i, j] > PHASH_PURGE, f"{a['path']} ({a['split']}) is too close to {b['path']} ({b['split']})"

# Every image file is either in the manifest or in excluded.csv.
all_files = {p.relative_to(ROOT).as_posix() for f, _, _ in FOLDERS for p in (DATA / f).glob('*.jpg')}
assert all_files == {r['path'] for r in rows} | {e['path'] for e in excluded}

table = collections.Counter((r['split'], r['source'], r['label'], r['multi_berry']) for r in rows)
print(f"{'split':6} {'source':9} {'label':8} {'single':>6} {'multi':>6} {'total':>6}")
for split in SPLIT:
    for source in ('mahendra', 'mendeley'):
        for label in ('good', 'damaged'):
            s, m = table[(split, source, label, 0)], table[(split, source, label, 1)]
            print(f'{split:6} {source:9} {label:8} {s:6} {m:6} {s + m:6}')
tot = collections.Counter(r['label'] for r in rows)
print('\nTotal:', dict(tot), '| all checks passed')
""")

nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}})
nbf.write(nb, out)
print("wrote", out)
