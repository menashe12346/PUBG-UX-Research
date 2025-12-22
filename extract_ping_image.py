import cv2

def save_ping_region(image_path: str, output_path: str = "test.png"):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)

    X1, Y1 = 200, 1040
    X2, Y2 = 266, 1080

    crop = img[Y1:Y2, X1:X2]

    cv2.imwrite(output_path, crop)


if __name__ == "__main__":
    save_ping_region("PUBG_Images/999999_undefined_000005_image.png")