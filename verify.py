"""Label-free verifiers for evidence-verified RL.

  midline_x(img)            the body's midline on a rendered axial slice (448 x 448): centre of mass of the filled body
                            mask (non-background pixels), in pixel coordinates
  side_of(x, img)           the patient side a point lies on, radiological display (image left = patient's right)
  remove(img, p, r)         counterfactual image: the disc of radius r around p is replaced by the median of the ring
                            r..1.6r around it plus matched noise, so a lesion there disappears into its surroundings
  control_point(img, p, r)  a random point inside the body, at least 2.5r from p, for the matched control removal
"""
import numpy as np
from PIL import Image
from scipy import ndimage


def gray(img):
    return np.asarray(img.convert("L"), dtype=np.float32)


def body_mask(g):
    """Otsu split of the slice (air/background vs tissue), largest component, holes filled (lungs, ventricles)."""
    hist, edges = np.histogram(g, bins=256, range=(0, 256)); c = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(hist); w1 = w0[-1] - w0; m0 = np.cumsum(hist * c); mu0 = m0 / np.maximum(w0, 1); mu1 = (m0[-1] - m0) / np.maximum(w1, 1)
    t = c[int(np.argmax(w0 * w1 * (mu0 - mu1) ** 2))]
    m = ndimage.binary_opening(g > t, iterations=2)
    lab, n = ndimage.label(m)
    if n == 0:
        return m
    big = 1 + int(np.argmax(ndimage.sum(np.ones_like(lab), lab, range(1, n + 1))))
    return ndimage.binary_fill_holes(lab == big)


def midline_x(img):
    m = body_mask(gray(img))
    return float(np.nonzero(m)[1].mean()) if m.any() else img.size[0] / 2


def ap_center_y(img):
    """Front-back centre of the body on a rendered axial slice: centre of mass of the body mask, in pixel rows (anterior
    is up in the radiological display)."""
    m = body_mask(gray(img))
    return float(np.nonzero(m)[0].mean()) if m.any() else img.size[1] / 2


def side_of(x, img):
    return "right" if x < midline_x(img) else "left"


def remove(img, p, r=40, rng=None):
    """Counterfactual: the disc of radius r around p is inpainted from its surroundings (Telea), so whatever lay there is
    replaced by a smooth continuation of the neighbouring tissue."""
    import cv2
    g = np.asarray(img.convert("L"), dtype=np.uint8); m = np.zeros_like(g)
    cv2.circle(m, (int(round(p[0])), int(round(p[1]))), int(r), 255, -1)
    out = cv2.inpaint(g, m, 7, cv2.INPAINT_TELEA)
    return Image.fromarray(out).convert("RGB")


def control_point(img, p, r=40, rng=None):
    rng = rng or np.random.default_rng(0)
    m = body_mask(gray(img)); yy, xx = np.nonzero(m)
    ok = np.hypot(xx - p[0], yy - p[1]) >= 2.5 * r
    if not ok.any():
        return (img.size[0] - p[0], p[1])
    k = rng.integers(ok.sum()); return (float(xx[ok][k]), float(yy[ok][k]))
