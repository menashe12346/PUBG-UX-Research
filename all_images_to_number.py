import cv2
import numpy as np
from pathlib import Path
import shutil
from typing import Dict, Tuple, Optional, List

DIGITS_DIR = r"/home/menashe/image_proses/digits_font"        # התיקייה עם 0.png ... 9.png
#IMAGES_DIR = r"/home/menashe/image_proses/PUBG_Images" 
IMAGES_DIR = r"/home/menashe/image_proses/adb/sessions_screencap_pc_roi/run_20251221_200610" 
DIGIT_SIZE = (40, 64)  # (width, height) – גודל אחיד לכל הספרות

DEBUG_DIR = Path("/home/menashe/image_proses/debug_out")

# מחיקה מלאה של התיקייה אם קיימת
if DEBUG_DIR.exists():
    shutil.rmtree(DEBUG_DIR)

# יצירה מחדש
DEBUG_DIR.mkdir(parents=True, exist_ok=True)

def dbg_write(filename: str, img) -> str:
    out_path = DEBUG_DIR / filename
    cv2.imwrite(str(out_path), img)
    return str(out_path)

def sharpen(image: np.ndarray, strength: float = 1.5, radius: float = 1.0) -> np.ndarray:

    if image is None or image.size == 0:
        raise ValueError("image is empty or None")

    blurred = cv2.GaussianBlur(image, ksize=(0, 0), sigmaX=radius, sigmaY=radius)
    sharp = cv2.addWeighted(image, 1 + strength, blurred, -strength, 0)

    return sharp

def upscale_image(img: np.ndarray, scale: int = 8) -> np.ndarray:
    if img is None:
        raise ValueError("img is None")

    if scale <= 0:
        raise ValueError("scale must be positive")

    up = cv2.resize(
        img,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_CUBIC 
    )
    return up

def make_white_digit_mask(img_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    bw = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        21,
        -5
    )

    bw = 255 - bw

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel, iterations=1)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=1)

    return bw

def img_to_bw_digit(img: np.ndarray) -> np.ndarray:
    if img is None:
        raise ValueError("img_to_bw_digit got None")

    if len(img.shape) == 3 and img.shape[2] == 4:
        bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    elif len(img.shape) == 3 and img.shape[2] == 3:
        bgr = img
    else:
        bgr = None 

    if bgr is not None:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        unique_vals = np.unique(gray)

        if len(unique_vals) <= 4:
            bw = (gray > 128).astype(np.uint8) * 255
            if np.mean(bw) > 127:
                bw = 255 - bw
        else:
            bw = make_white_digit_mask(bgr)

    else:
        gray = img
        unique_vals = np.unique(gray)

        if len(unique_vals) <= 4:
            bw = (gray > 128).astype(np.uint8) * 255
            if np.mean(bw) > 127:
                bw = 255 - bw
        else:
            gray = cv2.GaussianBlur(gray, (3, 3), 0)
            bw = cv2.adaptiveThreshold(
                gray, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                21, -5
            )
            bw = 255 - bw

    return bw

def normalize_digit(bw: np.ndarray) -> np.ndarray:
    h, w = bw.shape[:2]
    if (w, h) != DIGIT_SIZE:
        bw = cv2.resize(bw, DIGIT_SIZE, interpolation=cv2.INTER_AREA)
    return bw

def load_digit_templates(base_dir: str) -> Dict[int, List[np.ndarray]]:
    base = Path(base_dir) / "0_9"
    templates: Dict[int, List[np.ndarray]] = {d: [] for d in range(10)}

    for d in range(10):
        digit_dir = base / str(d)
        if not digit_dir.exists():
            # print(f"[WARN] directory {digit_dir} not found")
            continue

        for img_path in digit_dir.iterdir():
            if not img_path.is_file():
                continue

            img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
            if img is None:
                # print(f"[WARN] cannot read {img_path}")
                continue

            try:
                bw = img_to_bw_digit(img)
                bw = normalize_digit(bw)
                templates[d].append(bw)
            except Exception as e:
                pass   
                # print(f"[ERR] processing {img_path}: {e}")

        if not templates[d]:
            pass   
            # print(f"[WARN] no valid templates loaded for digit {d}")

    return templates


def segment_bw_digits(bw_digits: np.ndarray,
                      min_area: int = 500,
                      min_height: int = 40,
                      min_width: int = 15) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:

    fg = (bw_digits < 128).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel, iterations=1)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        fg, connectivity=8
    )

    h_img, w_img = bw_digits.shape[:2]
    img_area = h_img * w_img

    components = []
    if num_labels <= 1:
        return []

    for label in range(1, num_labels):
        x, y, w, h, area = stats[label]

        if (
            area > 0.9 * img_area and
            w > 0.9 * w_img and
            h > 0.9 * h_img
        ):
            continue

        if area < min_area:
            continue
        if h < min_height:
            continue
        if w < min_width:
            continue

        if w > 1.4 * h: 
            child_boxes = split_wide_component(bw_digits, int(x), int(y), int(w), int(h),
                                               min_child_width=min_width)
            components.extend(child_boxes)
        else:
            components.append((int(x), int(y), int(w), int(h)))

    if not components:
        return []

    components.sort(key=lambda bb: bb[0])

    crops = []
    for (x, y, w, h) in components:
        x0 = max(x - 1, 0)
        y0 = max(y - 1, 0)
        x1 = min(x + w + 1, bw_digits.shape[1])
        y1 = min(y + h + 1, bw_digits.shape[0])
        crop = bw_digits[y0:y1, x0:x1]
        crops.append((crop, (x0, y0, x1 - x0, y1 - y0)))

    return crops

def crop_to_foreground(bw: np.ndarray, pad: int = 10) -> np.ndarray:

    ys, xs = np.where(bw < 128)
    if len(xs) == 0 or len(ys) == 0:
        return bw

    x0 = max(int(xs.min()) - pad, 0)
    x1 = min(int(xs.max()) + pad + 1, bw.shape[1])
    y0 = max(int(ys.min()) - pad, 0)
    y1 = min(int(ys.max()) + pad + 1, bw.shape[0])

    return bw[y0:y1, x0:x1]


def split_wide_component(
    bw_digits: np.ndarray,
    x: int, y: int, w: int, h: int,
    min_child_width: int = 15
) -> List[Tuple[int, int, int, int]]:

    roi = (bw_digits[y:y+h, x:x+w] < 128).astype(np.uint8)  
    col_sum = roi.sum(axis=0)  
    h_roi = roi.shape[0]
    max_sum = int(col_sum.max())
    if max_sum == 0:
        return [(x, y, w, h)]


    thresh = min(0.35 * max_sum, 0.25 * h_roi)
    gap_cols = np.where(col_sum < thresh)[0]
    if len(gap_cols) == 0:
        return [(x, y, w, h)]


    splits = []
    start = gap_cols[0]
    prev = gap_cols[0]
    for c in gap_cols[1:]:
        if c == prev + 1:
            prev = c
        else:
            splits.append((start, prev))
            start = c
            prev = c
    splits.append((start, prev))


    boundaries = [0]
    for s, e in splits:
        mid = (s + e) // 2
        boundaries.append(mid)
    boundaries.append(w)

    child_boxes: List[Tuple[int, int, int, int]] = []
    for i in range(len(boundaries) - 1):
        x0_c = boundaries[i]
        x1_c = boundaries[i + 1]
        cw = x1_c - x0_c
        if cw < min_child_width:
            continue
        child_boxes.append((x + x0_c, y, cw, h))

    if len(child_boxes) <= 1:
        return [(x, y, w, h)]
    return child_boxes

def digit_similarity(bw_query: np.ndarray, bw_template: np.ndarray) -> float:

    if bw_query.shape != bw_template.shape:
        bw_template = cv2.resize(
            bw_template,
            (bw_query.shape[1], bw_query.shape[0]),
            interpolation=cv2.INTER_AREA,
        )
    diff = cv2.absdiff(bw_query, bw_template)
    return float(np.mean(diff))


def recognize_single_digit(
    bw_digit: np.ndarray, templates: Dict[int, List[np.ndarray]]
) -> Tuple[int, Dict[int, float]]:

    bw_clean = trim_digit_bw(img_to_bw_digit(bw_digit))
    bw_norm = normalize_digit(bw_clean)


    scores: Dict[int, float] = {}
    for d in range(10):
        tmpls = templates.get(d, [])
        if not tmpls:
            continue

        best = float("inf")
        for t in tmpls:
            s = digit_similarity(bw_norm, t)
            if s < best:
                best = s

        scores[d] = best

    if not scores:
        raise RuntimeError("no templates available for recognition")

    best_digit = min(scores, key=scores.get)
    return best_digit, scores

def test_recognize_number_from_bw_mask(
    bw_digits: np.ndarray,
    templates: Dict[int, List[np.ndarray]],
    max_mean_diff: float = 90.0,
    min_digits: int = 2,
) -> Tuple[Optional[int], List[int]]:

    segments = segment_bw_digits(bw_digits)
    if not segments:
        return None, []

    boxes = [bbox for (_, bbox) in segments]
    max_h = max(h for (_, _, _, h) in boxes)

    filtered = []
    for i, (digit_bw, bbox) in enumerate(segments):
        x, y, w, h = bbox

        if h < 0.80 * max_h:
            continue

        ar = w / float(h)
        if ar < 0.15 or ar > 0.85:
            continue

        filtered.append((digit_bw, bbox))

    if not filtered:
        return None, []

    filtered.sort(key=lambda t: t[1][0])  # sort by x
    #filtered = filtered[:2]  # Choose how many digits to return

    recognized_digits: List[int] = []

    for i, (digit_bw, bbox) in enumerate(filtered):
        d, scores = recognize_single_digit(digit_bw, templates)
        best_score = scores[d]

        dbg_write(f"dbg_keep_{i}_pred_{d}_score_{best_score:.1f}.png", digit_bw)

        if best_score > max_mean_diff:
            return None, []

        recognized_digits.append(d)

    if len(recognized_digits) < min_digits:
        return None, []

    number = 0
    for d in recognized_digits:
        number = number * 10 + d

    return number, recognized_digits

def trim_digit_bw(bw: np.ndarray, pad: int = 2) -> np.ndarray:
    ys, xs = np.where(bw < 128)
    if len(xs) == 0 or len(ys) == 0:
        return bw
    x0 = max(int(xs.min()) - pad, 0)
    x1 = min(int(xs.max()) + pad + 1, bw.shape[1])
    y0 = max(int(ys.min()) - pad, 0)
    y1 = min(int(ys.max()) + pad + 1, bw.shape[0])
    return bw[y0:y1, x0:x1]

def test_recognize_number_from_screenshot(
    image: np.ndarray,
    templates: Dict[int, List[np.ndarray]]
) -> Tuple[Optional[int], List[int]]:

    crop_big = upscale_image(sharpen(image))

    crop_big = crop_big[:, :int(0.70 * crop_big.shape[1])]

    bw = make_white_digit_mask(crop_big)

    bw = crop_to_foreground(bw, pad=12)

    dbg_write("dbg_bw_white.png", bw)

    number, digits = test_recognize_number_from_bw_mask(
        bw,
        templates,
        max_mean_diff=80.0,
        min_digits=1,
    )

    return number, digits

def get_ping_region(image_path: str):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)

    X1, Y1 = 200, 1040
    X2, Y2 = 266, 1080

    crop = img[Y1:Y2, X1:X2]

    return crop

def iter_image_files(folder: str):
    exts = {".png", ".jpg", ".jpeg", ".webp"}
    p = Path(folder)
    files = [f for f in p.iterdir() if f.is_file() and f.suffix.lower() in exts]
    files.sort(key=lambda x: x.name.lower())
    return files
def run_on_folder(images_dir: str, templates):
    files = iter_image_files(images_dir)
    if not files:
        return

    results = []  # (path, number_or_none)

    for path in files:
        try:
            crop = get_ping_region(str(path))
            number, digits = test_recognize_number_from_screenshot(crop, templates)

            if number is None:
                results.append((path, None))
            else:
                results.append((path, int(number)))

        except Exception:
            results.append((path, None))

    # ---- sort files by recognized number ----
    # numbers first (ascending), undefined at the end
    def sort_key(item):
        path, num = item
        if num is None:
            return (1, 0, path.name.lower())  # undefined last
        return (0, num, path.name.lower())

    results_sorted = sorted(results, key=sort_key)

    # ---- print ONE LINE, comma-separated, SORTED ----
    out_values_sorted = [
        ("undefined" if num is None else str(num))
        for (_, num) in results_sorted
    ]
    print(",".join(out_values_sorted))

    # ---- rename with numeric prefix so folder listing becomes sorted ----
    p = Path(images_dir)

    # two-pass rename to avoid collisions
    temp_paths = []
    for i, (path, num) in enumerate(results_sorted, start=1):
        tmp = p / (f"__tmp__{i:06d}__{path.name}")
        path.rename(tmp)
        temp_paths.append((tmp, path.name, num))

    # final rename
    for idx, (tmp_path, original_name, num) in enumerate(temp_paths, start=1):
        if num is None:
            new_name = f"999999_undefined_{idx:06d}_{original_name}"
        else:
            new_name = f"{num:06d}_{idx:06d}_{original_name}"
        tmp_path.rename(p / new_name)

# ---- main ----
templates = load_digit_templates(DIGITS_DIR)
run_on_folder(IMAGES_DIR, templates)


