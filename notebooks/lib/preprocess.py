"""Strawberry grader preprocessing: crop to the berry, fit inside 224x224, pad with the border colour.

This is the reference implementation. Training (notebook 02), export (notebook 03), the web demo
(preprocess.js) and the Android app must all do exactly this, or accuracy drops.

Model input: float32, shape (1, 224, 224, 3), RGB, values 0-255. NO normalisation (no /255):
EfficientNet-B0 and MobileNetV3 rescale inside the model. Output: P(damaged).
"""
import numpy as np
from PIL import Image, ImageOps

SIZE = 224          # model input size
MASK_SIZE = 256     # the berry mask is computed on a copy resized to 256x256
DIFF_MIN = 60       # |pixel - median border colour| summed over R, G, B must exceed this...
SAT_MIN = 70        # ...and the pixel must be coloured: HSV saturation (0-255) above this
LINE_FRAC = 0.8     # mask rows/columns more than 80% full are lines (e.g. a webcam edge), not berry
PCT = (1, 99)       # the box spans the 1st to 99th percentile of mask pixels
MARGIN = 0.15       # margin around the box, as a fraction of its size; then made square
MIN_PIXELS = 50     # fewer mask pixels than this: no crop


def berry_box(a, margin=MARGIN):
    """Box (x0, y0, x1, y1) around the berry or pile in RGB array `a`, or None if none is found."""
    h, w = a.shape[:2]
    img = Image.fromarray(a).resize((MASK_SIZE, MASK_SIZE))
    small = np.asarray(img, dtype=np.int16)
    sat = np.asarray(img.convert('HSV'), dtype=np.int16)[..., 1]
    bg = np.median(np.concatenate([small[0], small[-1], small[:, 0], small[:, -1]]), axis=0)
    mask = (np.abs(small - bg).sum(-1) > DIFF_MIN) & (sat > SAT_MIN)
    rows, cols = mask.mean(1) > LINE_FRAC, mask.mean(0) > LINE_FRAC
    mask[rows, :] = False
    mask[:, cols] = False
    ys, xs = np.where(mask)
    if len(ys) < MIN_PIXELS:
        return None
    (y0, y1) = np.percentile(ys, PCT) * h / MASK_SIZE
    (x0, x1) = np.percentile(xs, PCT) * w / MASK_SIZE
    side = max(x1 - x0, y1 - y0) * (1 + 2 * margin)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    box = (max(0, cx - side / 2), max(0, cy - side / 2), min(w, cx + side / 2), min(h, cy + side / 2))
    return tuple(int(round(v)) for v in box)


def fit_pad_image(im, size=SIZE, crop=True):
    """Upright RGB PIL image -> (uint8 array size x size x 3, crop box or None)."""
    a = np.asarray(im)
    border = np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]])
    bg = tuple(int(v) for v in np.median(border, axis=0))
    box = berry_box(a) if crop else None
    if box is not None:
        im = im.crop(box)
    im = im.copy()
    im.thumbnail((size, size), Image.LANCZOS)   # shrinks only, never enlarges
    canvas = Image.new('RGB', (size, size), bg)
    canvas.paste(im, ((size - im.width) // 2, (size - im.height) // 2))
    return np.asarray(canvas, dtype=np.uint8), box


def load_image(path, draft=False):
    """Open a photo upright (EXIF rotation applied). draft=True decodes big JPEGs at reduced size (training)."""
    with Image.open(path) as im:
        if draft:
            im.draft('RGB', (SIZE * 4, SIZE * 4))
        return ImageOps.exif_transpose(im).convert('RGB')


def fit_pad(path, size=SIZE, crop=True):
    """Exactly what training used (notebook 02): reduced-size JPEG decode, crop, fit, pad. uint8 array."""
    return fit_pad_image(load_image(path, draft=True), size, crop)[0]


def preprocess(image):
    """For apps and checks: a path or an upright RGB PIL image -> float32 model input (1, 224, 224, 3).
    Decodes at full size, as a phone or browser would."""
    im = load_image(image) if isinstance(image, (str, bytes)) or hasattr(image, '__fspath__') else image.convert('RGB')
    arr, _ = fit_pad_image(im)
    return arr.astype(np.float32)[None]
