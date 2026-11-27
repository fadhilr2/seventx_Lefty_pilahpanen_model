/*
 * Strawberry grader preprocessing for the web demo: a port of preprocess.py.
 * Crop to the berry, fit inside 224x224 (shrink only), pad with the median border colour.
 *
 * Input:  an upright image as {width, height, data}, where data is RGBA (canvas ImageData) or RGB bytes.
 *         Browsers apply EXIF rotation when decoding, so draw the photo to a canvas and pass getImageData().
 * Output: {input: Float32Array(224*224*3) RGB 0-255 (NO /255), box: [x0,y0,x1,y1] or null, pixels: Uint8Array}
 *         Feed `input` to the model as shape [1, 224, 224, 3].
 *
 * Resampling copies Pillow's algorithm (antialiased convolution, fixed-point coefficients, two passes,
 * and thumbnail's pre-reduction), so the result matches the Python reference within a level or two.
 * Works in the browser (window.StrawberryPreprocess) and in Node (require).
 */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else root.StrawberryPreprocess = factory();
}(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  const P = { SIZE: 224, MASK_SIZE: 256, DIFF_MIN: 60, SAT_MIN: 70, LINE_FRAC: 0.8,
              PCT_LO: 1, PCT_HI: 99, MARGIN: 0.15, MIN_PIXELS: 50 };
  const PRECISION_BITS = 32 - 8 - 2;   // Pillow's fixed-point precision for 8-bit images

  function toRGB(img) {
    const w = img.width, h = img.height, d = img.data, ch = d.length / (w * h);
    const px = new Uint8Array(w * h * 3);
    for (let i = 0, j = 0; i < w * h; i++, j += ch) { px[3 * i] = d[j]; px[3 * i + 1] = d[j + 1]; px[3 * i + 2] = d[j + 2]; }
    return { w, h, px };
  }

  // ---- Pillow-compatible resampling -------------------------------------------------------------
  const FILTERS = {
    bicubic: { support: 2, fn: (x) => { const a = -0.5; x = Math.abs(x);
      if (x < 1) return ((a + 2) * x - (a + 3)) * x * x + 1;
      if (x < 2) return (((x - 5) * x + 8) * x - 4) * a; return 0; } },
    lanczos: { support: 3, fn: (x) => (x >= -3 && x < 3) ? sinc(x) * sinc(x / 3) : 0 },
  };
  function sinc(x) { if (x === 0) return 1; x *= Math.PI; return Math.sin(x) / x; }

  function coeffs(inSize, in0, in1, outSize, filter) {
    const scale = (in1 - in0) / outSize, filterscale = Math.max(scale, 1);
    const support = filter.support * filterscale, ksize = Math.ceil(support) * 2 + 1;
    const bounds = new Int32Array(outSize * 2), kk = new Float64Array(outSize * ksize);
    for (let xx = 0; xx < outSize; xx++) {
      const center = in0 + (xx + 0.5) * scale, ss = 1 / filterscale;
      let xmin = Math.trunc(center - support + 0.5); if (xmin < 0) xmin = 0;
      let xmax = Math.trunc(center + support + 0.5); if (xmax > inSize) xmax = inSize;
      xmax -= xmin;
      let ww = 0;
      for (let x = 0; x < xmax; x++) { const w = filter.fn((x + xmin - center + 0.5) * ss); kk[xx * ksize + x] = w; ww += w; }
      for (let x = 0; x < xmax; x++) {
        let k = ww !== 0 ? kk[xx * ksize + x] / ww : kk[xx * ksize + x];
        kk[xx * ksize + x] = Math.trunc(k * (1 << PRECISION_BITS) + (k < 0 ? -0.5 : 0.5));   // fixed point, as Pillow
      }
      bounds[2 * xx] = xmin; bounds[2 * xx + 1] = xmax;
    }
    return { ksize, bounds, kk };
  }
  const clip8 = (ss) => { const v = Math.floor(ss / (1 << PRECISION_BITS)); return v < 0 ? 0 : v > 255 ? 255 : v; };

  // Resize RGB image {w,h,px} to (ow, oh) from source box [x0,y0,x1,y1] (floats allowed), like Image.resize.
  function resample(src, ow, oh, filter, box) {
    box = box || [0, 0, src.w, src.h];
    const needH = ow !== src.w || box[0] !== 0 || box[2] !== src.w;
    const needV = oh !== src.h || box[1] !== 0 || box[3] !== src.h;
    const ch = coeffs(src.w, box[0], box[2], ow, filter), cv = coeffs(src.h, box[1], box[3], oh, filter);
    const half = 1 << (PRECISION_BITS - 1);
    let img = src;
    let vb = cv.bounds;
    if (needH) {
      const yFirst = cv.bounds[0], yLast = cv.bounds[2 * oh - 2] + cv.bounds[2 * oh - 1];
      vb = Int32Array.from(cv.bounds); for (let i = 0; i < oh; i++) vb[2 * i] -= yFirst;
      const th = yLast - yFirst, out = new Uint8Array(ow * th * 3);
      for (let y = 0; y < th; y++) {
        const row = (y + yFirst) * src.w;
        for (let xx = 0; xx < ow; xx++) {
          const xmin = ch.bounds[2 * xx], xmax = ch.bounds[2 * xx + 1], k = xx * ch.ksize;
          let s0 = half, s1 = half, s2 = half;
          for (let x = 0; x < xmax; x++) {
            const p = (row + x + xmin) * 3, c = ch.kk[k + x];
            s0 += src.px[p] * c; s1 += src.px[p + 1] * c; s2 += src.px[p + 2] * c;
          }
          const o = (y * ow + xx) * 3; out[o] = clip8(s0); out[o + 1] = clip8(s1); out[o + 2] = clip8(s2);
        }
      }
      img = { w: ow, h: th, px: out };
    }
    if (needV) {
      const out = new Uint8Array(img.w * oh * 3);
      for (let yy = 0; yy < oh; yy++) {
        const ymin = vb[2 * yy], ymax = vb[2 * yy + 1], k = yy * cv.ksize;
        for (let x = 0; x < img.w; x++) {
          let s0 = half, s1 = half, s2 = half;
          for (let y = 0; y < ymax; y++) {
            const p = ((y + ymin) * img.w + x) * 3, c = cv.kk[k + y];
            s0 += img.px[p] * c; s1 += img.px[p + 1] * c; s2 += img.px[p + 2] * c;
          }
          const o = (yy * img.w + x) * 3; out[o] = clip8(s0); out[o + 1] = clip8(s1); out[o + 2] = clip8(s2);
        }
      }
      img = { w: img.w, h: oh, px: out };
    }
    return img;
  }

  // Box-average downscale by integer factors (Image.reduce); edge blocks average what is there.
  function reduce(src, fx, fy) {
    const ow = Math.ceil(src.w / fx), oh = Math.ceil(src.h / fy), out = new Uint8Array(ow * oh * 3);
    for (let oy = 0; oy < oh; oy++) for (let ox = 0; ox < ow; ox++) {
      const y1 = Math.min((oy + 1) * fy, src.h), x1 = Math.min((ox + 1) * fx, src.w);
      let s0 = 0, s1 = 0, s2 = 0, n = 0;
      for (let y = oy * fy; y < y1; y++) for (let x = ox * fx; x < x1; x++) {
        const p = (y * src.w + x) * 3; s0 += src.px[p]; s1 += src.px[p + 1]; s2 += src.px[p + 2]; n++;
      }
      const o = (oy * ow + ox) * 3, a = Math.floor(n / 2);
      out[o] = Math.floor((s0 + a) / n); out[o + 1] = Math.floor((s1 + a) / n); out[o + 2] = Math.floor((s2 + a) / n);
    }
    return { w: ow, h: oh, px: out };
  }

  // Image.thumbnail((size, size), LANCZOS) on a non-JPEG image: shrink only, keep aspect, reducing_gap 2.
  function thumbnail(src, size) {
    if (size >= src.w && size >= src.h) return src;
    const aspect = src.w / src.h;
    let x = size, y = size;
    const roundAspect = (n, key) => Math.max([Math.floor(n), Math.ceil(n)].reduce((a, b) => key(b) < key(a) ? b : a), 1);
    if (x / y >= aspect) x = roundAspect(y * aspect, (n) => Math.abs(aspect - n / y));
    else y = roundAspect(x / aspect, (n) => n === 0 ? 0 : Math.abs(aspect - x / n));
    if (x === src.w && y === src.h) return src;
    const fx = Math.trunc(src.w / x / 2) || 1, fy = Math.trunc(src.h / y / 2) || 1;
    let img = src, box = [0, 0, src.w, src.h];
    if (fx > 1 || fy > 1) { img = reduce(src, fx, fy); box = [0, 0, src.w / fx, src.h / fy]; }
    return resample(img, x, y, FILTERS.lanczos, box);
  }

  // ---- the crop ----------------------------------------------------------------------------------
  function median(vals) {   // numpy median: mean of the two middle values for even counts
    const s = Float64Array.from(vals).sort(), n = s.length;
    return n % 2 ? s[(n - 1) / 2] : (s[n / 2 - 1] + s[n / 2]) / 2;
  }
  function borderMedian(img) {   // np.median(concatenate([a[0], a[-1], a[:,0], a[:,-1]]), axis=0)
    const out = [];
    for (let c = 0; c < 3; c++) {
      const v = [];
      for (let x = 0; x < img.w; x++) v.push(img.px[x * 3 + c]);
      for (let x = 0; x < img.w; x++) v.push(img.px[((img.h - 1) * img.w + x) * 3 + c]);
      for (let y = 0; y < img.h; y++) v.push(img.px[(y * img.w) * 3 + c]);
      for (let y = 0; y < img.h; y++) v.push(img.px[(y * img.w + img.w - 1) * 3 + c]);
      out.push(median(v));
    }
    return out;
  }
  function percentile(sorted, q) {   // numpy 'linear' percentile of a sorted array
    const i = (q / 100) * (sorted.length - 1), lo = Math.floor(i), hi = Math.ceil(i);
    return sorted[lo] + (sorted[hi] - sorted[lo]) * (i - lo);
  }
  function roundHalfEven(v) {   // Python's round()
    const f = Math.floor(v), d = v - f;
    if (d > 0.5) return f + 1; if (d < 0.5) return f; return f % 2 === 0 ? f : f + 1;
  }

  function berryBox(img) {
    const S = P.MASK_SIZE, small = resample(img, S, S, FILTERS.bicubic);
    const bg = borderMedian(small), mask = new Uint8Array(S * S);
    const rowCount = new Int32Array(S), colCount = new Int32Array(S);
    for (let i = 0; i < S * S; i++) {
      const r = small.px[3 * i], g = small.px[3 * i + 1], b = small.px[3 * i + 2];
      const diff = Math.abs(r - bg[0]) + Math.abs(g - bg[1]) + Math.abs(b - bg[2]);
      const mx = Math.max(r, g, b), mn = Math.min(r, g, b);
      const sat = mx === mn ? 0 : Math.trunc(Math.fround((mx - mn) / mx) * 255);   // PIL HSV saturation
      if (diff > P.DIFF_MIN && sat > P.SAT_MIN) { mask[i] = 1; rowCount[Math.floor(i / S)]++; colCount[i % S]++; }
    }
    const ys = [], xs = [];
    for (let y = 0; y < S; y++) {
      if (rowCount[y] / S > P.LINE_FRAC) continue;
      for (let x = 0; x < S; x++) if (mask[y * S + x] && !(colCount[x] / S > P.LINE_FRAC)) { ys.push(y); xs.push(x); }
    }
    if (ys.length < P.MIN_PIXELS) return null;
    ys.sort((a, b) => a - b); xs.sort((a, b) => a - b);
    const y0 = percentile(ys, P.PCT_LO) * img.h / S, y1 = percentile(ys, P.PCT_HI) * img.h / S;
    const x0 = percentile(xs, P.PCT_LO) * img.w / S, x1 = percentile(xs, P.PCT_HI) * img.w / S;
    const side = Math.max(x1 - x0, y1 - y0) * (1 + 2 * P.MARGIN), cx = (x0 + x1) / 2, cy = (y0 + y1) / 2;
    return [Math.max(0, cx - side / 2), Math.max(0, cy - side / 2), Math.min(img.w, cx + side / 2), Math.min(img.h, cy + side / 2)]
      .map(roundHalfEven);
  }

  function crop(img, [x0, y0, x1, y1]) {
    const w = x1 - x0, h = y1 - y0, out = new Uint8Array(w * h * 3);
    for (let y = 0; y < h; y++) out.set(img.px.subarray(((y + y0) * img.w + x0) * 3, ((y + y0) * img.w + x0 + w) * 3), y * w * 3);
    return { w, h, px: out };
  }

  function preprocess(imageData, options) {
    const doCrop = !options || options.crop !== false;
    let img = toRGB(imageData);
    const bg = borderMedian(img).map(Math.trunc);
    const box = doCrop ? berryBox(img) : null;
    if (box) img = crop(img, box);
    img = thumbnail(img, P.SIZE);
    const S = P.SIZE, pixels = new Uint8Array(S * S * 3);
    for (let i = 0; i < S * S; i++) { pixels[3 * i] = bg[0]; pixels[3 * i + 1] = bg[1]; pixels[3 * i + 2] = bg[2]; }
    const ox = Math.floor((S - img.w) / 2), oy = Math.floor((S - img.h) / 2);
    for (let y = 0; y < img.h; y++) pixels.set(img.px.subarray(y * img.w * 3, (y + 1) * img.w * 3), ((y + oy) * S + ox) * 3);
    return { input: Float32Array.from(pixels), box, pixels };
  }

  return { preprocess, berryBox, PARAMS: P };
}));
