import tensorflow as tf
import matplotlib.pyplot as plt

# ==============================
# SETTINGS 
# ==============================
# name = input("Enter the name of the image (without extension): ")
INPUT_PATH  = f"image.jpg"
OUTPUT_PATH = f"image_enhanced.png"
SKIN_TONE   = 0.0    # -1.0 (bahut dark) ... 0 (no change) ... +1.0 (bahut light)
MAX_SIDE    = 2000   # bahut badi image ko isse bada nahi karte (memory ke liye)


# ==============================
# HELPERS
# ==============================
def load_image(path, max_side=MAX_SIDE):
    data = tf.io.read_file(path)
    img = tf.io.decode_image(data, channels=3, expand_animations=False)
    img = tf.image.convert_image_dtype(img, tf.float32)  # [0,255] -> [0,1]
    h, w = img.shape[0], img.shape[1]
    scale = min(1.0, max_side / max(h, w))
    if scale < 1.0:  # aspect ratio preserve
        img = tf.image.resize(img, (int(h * scale), int(w * scale)), antialias=True)
    return tf.clip_by_value(img, 0.0, 1.0)


def gaussian_blur(x, sigma):
    """x: [H, W, C] float tensor"""
    radius = max(1, int(3 * sigma))
    ax = tf.range(-radius, radius + 1, dtype=tf.float32)
    k = tf.exp(-(ax ** 2) / (2.0 * sigma ** 2))
    k = k / tf.reduce_sum(k)
    c = x.shape[-1]
    kh = tf.tile(tf.reshape(k, [1, -1, 1, 1]), [1, 1, c, 1])
    kv = tf.tile(tf.reshape(k, [-1, 1, 1, 1]), [1, 1, c, 1])

    x = x[None]
    x = tf.pad(x, [[0, 0], [0, 0], [radius, radius], [0, 0]], "REFLECT")
    x = tf.nn.depthwise_conv2d(x, kh, [1, 1, 1, 1], "VALID")
    x = tf.pad(x, [[0, 0], [radius, radius], [0, 0], [0, 0]], "REFLECT")
    x = tf.nn.depthwise_conv2d(x, kv, [1, 1, 1, 1], "VALID")
    return x[0]


def auto_levels(img, low=0.01, high=0.99, strength=0.7):
    """Dull/washed-out image ka dynamic range theek karta hai."""
    lum = tf.reduce_mean(img, axis=-1)
    flat = tf.sort(tf.reshape(lum, [-1]))
    n = flat.shape[0]
    lo = flat[int(low * (n - 1))]
    hi = flat[int(high * (n - 1))]
    stretched = tf.clip_by_value((img - lo) / tf.maximum(hi - lo, 1e-3), 0.0, 1.0)
    return (1 - strength) * img + strength * stretched


def skin_mask(img):
    """Soft skin mask (YCbCr thresholding). Returns [H, W, 1] in 0..1"""
    r, g, b = img[..., 0] * 255, img[..., 1] * 255, img[..., 2] * 255
    y  = 0.299 * r + 0.587 * g + 0.114 * b
    cb = 128 - 0.168736 * r - 0.331264 * g + 0.5 * b
    cr = 128 + 0.5 * r - 0.418688 * g - 0.081312 * b
    m = (cb >= 77) & (cb <= 127) & (cr >= 133) & (cr <= 173) & (y > 40)
    m = tf.cast(m, tf.float32)[..., None]
    sigma = max(3.0, max(img.shape[0], img.shape[1]) / 250.0)
    return tf.clip_by_value(gaussian_blur(m, sigma), 0.0, 1.0)  # soft edges


def change_skin_tone(img, amount):
    """amount > 0 -> lighter, amount < 0 -> darker (sirf skin area par)."""
    if amount == 0:
        return img
    gamma = 1.8 ** (-amount)               # <1 brighten, >1 darken
    adjusted = tf.pow(tf.clip_by_value(img, 1e-4, 1.0), gamma)
    m = skin_mask(img)
    return m * adjusted + (1 - m) * img


def sharpen(img, amount=0.5, sigma=1.5):
    """Unsharp mask — detail/clarity badhata hai."""
    blurred = gaussian_blur(img, sigma)
    return img + amount * (img - blurred)


# ==============================
# PIPELINE
# ==============================
original = load_image(INPUT_PATH)
img = original

img = auto_levels(img)                        # 1. exposure/levels fix
img = change_skin_tone(img, SKIN_TONE)        # 2. skin tone (optional)
img = tf.image.adjust_contrast(img, 1.08)     # 3. contrast
img = tf.image.adjust_saturation(img, 1.10)   # 4. colors thode vivid
img = img * tf.constant([1.03, 1.00, 0.97])   # 5. subtle warm tone (FIXED)
img = sharpen(img, amount=0.5)                # 6. sharpness
img = tf.clip_by_value(img, 0.0, 1.0)         # 7. valid range


# ==============================
# SAVE (original resolution me) + PREVIEW
# ==============================
out_u8 = tf.image.convert_image_dtype(img, tf.uint8, saturate=True)
tf.io.write_file(OUTPUT_PATH, tf.io.encode_png(out_u8))

fig, axs = plt.subplots(1, 2, figsize=(14, 7))
axs[0].imshow(original); axs[0].set_title("Before"); axs[0].axis("off")
axs[1].imshow(img);      axs[1].set_title("After");  axs[1].axis("off")
plt.tight_layout()
plt.show()