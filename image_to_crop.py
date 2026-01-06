from pathlib import Path
import cv2


def save_ping_region(image_path: str, output_path: str):
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(str(image_path))

    X1, Y1 = 200, 1040
    X2, Y2 = 266, 1080

    crop = img[Y1:Y2, X1:X2]
    if crop.size == 0:
        raise ValueError(f"Empty crop for: {image_path}")

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ok = cv2.imwrite(str(out_path), crop)
    if not ok:
        raise RuntimeError(f"cv2.imwrite failed for: {out_path}")


def crop_tree_to_new_root(src_root: str, dst_root: str):
    src = Path(src_root)
    dst = Path(dst_root)

    if not src.exists():
        raise FileNotFoundError(src_root)

    # אוספים PNG (כולל .PNG)
    png_files = list(src.rglob("*.png")) + list(src.rglob("*.PNG"))
    print(f"[+] Found {len(png_files)} PNG files under: {src}")

    for p in png_files:
        try:
            # שומר את אותה היררכיה: relative path מהמקור
            rel = p.relative_to(src)
            out_path = dst / rel

            save_ping_region(str(p), str(out_path))

        except Exception as e:
            print(f"[!] Failed on {p}: {type(e).__name__}: {e}")

    print("[+] Done.")
    print("[+] Source kept intact:", src)
    print("[+] Cropped output at   :", dst)


if __name__ == "__main__":
    # דוגמה:
    # המקור:  C:\data\sessions_screencap_pc_roi
    # יעד:    C:\data\sessions_screencap_pc_roi_CROPPED
    crop_tree_to_new_root(
        src_root=r"captures_old",
        dst_root=r"captures"
    )
