"""
Brain MRI Analysis System - Streamlit Web Application

Uses a pre-trained U-Net model to perform brain tumor segmentation
on BraTS2020 dataset MRI images.
"""

import os
import io
import streamlit as st
import numpy as np
import nibabel as nib
import cv2
import imageio
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import (
    Conv2D, Input, MaxPooling2D, Dropout, concatenate, UpSampling2D
)
from tensorflow.keras.models import Model

# ──────────────────────────── Constants ────────────────────────────
# These constants MUST match the values used during training (from the notebook)
IMG_SIZE = 128
GIF_DISPLAY_SIZE = 256
VOLUME_SLICES = 40
VOLUME_START_AT = 50

SEGMENT_CLASSES = {
    0: "NOT tumor",
    1: "NECROTIC/CORE",
    2: "EDEMA",
    3: "ENHANCING",
}

# Paths
TRAIN_DATASET_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "BraTS2020_TrainingData",
    "MICCAI_BraTS2020_TrainingData",
)
WEIGHTS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "model_.24-0.039128.weights.h5",
)

# Segmentation colormap (RGBA for overlay)
SEG_CMAP = {
    0: [0, 0, 0, 0],         # NOT tumor – transparent
    1: [255, 0, 0, 160],      # NECROTIC/CORE – red
    2: [0, 255, 0, 160],      # EDEMA – green
    3: [0, 0, 255, 160],      # ENHANCING – blue
}


# ──────────────────── Custom metrics (needed for model load) ──────
def dice_coef(y_true, y_pred, smooth=1.0):
    class_num = 4
    for i in range(class_num):
        y_true_f = tf.reshape(y_true[:, :, :, i], [-1])
        y_pred_f = tf.reshape(y_pred[:, :, :, i], [-1])
        intersection = tf.reduce_sum(y_true_f * y_pred_f)
        loss = (2.0 * intersection + smooth) / (
            tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth
        )
        if i == 0:
            total_loss = loss
        else:
            total_loss = total_loss + loss
    total_loss = total_loss / class_num
    return total_loss


def dice_coef_necrotic(y_true, y_pred, epsilon=1e-6):
    y_true_c = y_true[:, :, :, 1]
    y_pred_c = y_pred[:, :, :, 1]
    intersection = tf.reduce_sum(tf.abs(y_true_c * y_pred_c))
    return (2.0 * intersection) / (
        tf.reduce_sum(tf.square(y_true_c))
        + tf.reduce_sum(tf.square(y_pred_c))
        + epsilon
    )


def dice_coef_edema(y_true, y_pred, epsilon=1e-6):
    y_true_c = y_true[:, :, :, 2]
    y_pred_c = y_pred[:, :, :, 2]
    intersection = tf.reduce_sum(tf.abs(y_true_c * y_pred_c))
    return (2.0 * intersection) / (
        tf.reduce_sum(tf.square(y_true_c))
        + tf.reduce_sum(tf.square(y_pred_c))
        + epsilon
    )


def dice_coef_enhancing(y_true, y_pred, epsilon=1e-6):
    y_true_c = y_true[:, :, :, 3]
    y_pred_c = y_pred[:, :, :, 3]
    intersection = tf.reduce_sum(tf.abs(y_true_c * y_pred_c))
    return (2.0 * intersection) / (
        tf.reduce_sum(tf.square(y_true_c))
        + tf.reduce_sum(tf.square(y_pred_c))
        + epsilon
    )


def precision(y_true, y_pred):
    true_positives = tf.reduce_sum(
        tf.round(tf.clip_by_value(y_true * y_pred, 0, 1))
    )
    predicted_positives = tf.reduce_sum(
        tf.round(tf.clip_by_value(y_pred, 0, 1))
    )
    return true_positives / (predicted_positives + tf.keras.backend.epsilon())


def sensitivity(y_true, y_pred):
    true_positives = tf.reduce_sum(
        tf.round(tf.clip_by_value(y_true * y_pred, 0, 1))
    )
    possible_positives = tf.reduce_sum(
        tf.round(tf.clip_by_value(y_true, 0, 1))
    )
    return true_positives / (possible_positives + tf.keras.backend.epsilon())


def specificity(y_true, y_pred):
    true_negatives = tf.reduce_sum(
        tf.round(tf.clip_by_value((1 - y_true) * (1 - y_pred), 0, 1))
    )
    possible_negatives = tf.reduce_sum(
        tf.round(tf.clip_by_value(1 - y_true, 0, 1))
    )
    return true_negatives / (possible_negatives + tf.keras.backend.epsilon())


# ──────────────────────── Model architecture ──────────────────────
def build_unet(inputs, ker_init, dropout):
    conv1 = Conv2D(32, 3, activation="relu", padding="same", kernel_initializer=ker_init)(inputs)
    conv1 = Conv2D(32, 3, activation="relu", padding="same", kernel_initializer=ker_init)(conv1)

    pool = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv = Conv2D(64, 3, activation="relu", padding="same", kernel_initializer=ker_init)(pool)
    conv = Conv2D(64, 3, activation="relu", padding="same", kernel_initializer=ker_init)(conv)

    pool1 = MaxPooling2D(pool_size=(2, 2))(conv)
    conv2 = Conv2D(128, 3, activation="relu", padding="same", kernel_initializer=ker_init)(pool1)
    conv2 = Conv2D(128, 3, activation="relu", padding="same", kernel_initializer=ker_init)(conv2)

    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation="relu", padding="same", kernel_initializer=ker_init)(pool2)
    conv3 = Conv2D(256, 3, activation="relu", padding="same", kernel_initializer=ker_init)(conv3)

    pool4 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv5 = Conv2D(512, 3, activation="relu", padding="same", kernel_initializer=ker_init)(pool4)
    conv5 = Conv2D(512, 3, activation="relu", padding="same", kernel_initializer=ker_init)(conv5)
    drop5 = Dropout(dropout)(conv5)

    up7 = Conv2D(256, 2, activation="relu", padding="same", kernel_initializer=ker_init)(
        UpSampling2D(size=(2, 2))(drop5)
    )
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(256, 3, activation="relu", padding="same", kernel_initializer=ker_init)(merge7)
    conv7 = Conv2D(256, 3, activation="relu", padding="same", kernel_initializer=ker_init)(conv7)

    up8 = Conv2D(128, 2, activation="relu", padding="same", kernel_initializer=ker_init)(
        UpSampling2D(size=(2, 2))(conv7)
    )
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(128, 3, activation="relu", padding="same", kernel_initializer=ker_init)(merge8)
    conv8 = Conv2D(128, 3, activation="relu", padding="same", kernel_initializer=ker_init)(conv8)

    up9 = Conv2D(64, 2, activation="relu", padding="same", kernel_initializer=ker_init)(
        UpSampling2D(size=(2, 2))(conv8)
    )
    merge9 = concatenate([conv, up9], axis=3)
    conv9 = Conv2D(64, 3, activation="relu", padding="same", kernel_initializer=ker_init)(merge9)
    conv9 = Conv2D(64, 3, activation="relu", padding="same", kernel_initializer=ker_init)(conv9)

    up = Conv2D(32, 2, activation="relu", padding="same", kernel_initializer=ker_init)(
        UpSampling2D(size=(2, 2))(conv9)
    )
    merge = concatenate([conv1, up], axis=3)
    conv = Conv2D(32, 3, activation="relu", padding="same", kernel_initializer=ker_init)(merge)
    conv = Conv2D(32, 3, activation="relu", padding="same", kernel_initializer=ker_init)(conv)

    conv10 = Conv2D(4, (1, 1), activation="softmax")(conv)

    return Model(inputs=inputs, outputs=conv10)


# ──────────────────────── Cached model loader ─────────────────────
@st.cache_resource(show_spinner="Loading U-Net model …")
def load_model():
    """Build the U-Net architecture and load pre-trained weights."""
    input_layer = Input((IMG_SIZE, IMG_SIZE, 2))
    model = build_unet(input_layer, "he_normal", 0.2)
    model.compile(
        loss="categorical_crossentropy",
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        metrics=[
            "accuracy",
            tf.keras.metrics.MeanIoU(num_classes=4),
            dice_coef,
            precision,
            sensitivity,
            specificity,
            dice_coef_necrotic,
            dice_coef_edema,
            dice_coef_enhancing,
        ],
    )
    model.load_weights(WEIGHTS_PATH)
    return model


# ──────────────────────── Dataset helpers ─────────────────────────
@st.cache_data(show_spinner="Scanning dataset …")
def list_cases():
    """Return sorted list of available BraTS case IDs."""
    cases = sorted(
        [
            d
            for d in os.listdir(TRAIN_DATASET_PATH)
            if os.path.isdir(os.path.join(TRAIN_DATASET_PATH, d))
            and d.startswith("BraTS20_Training_")
        ]
    )
    return cases


@st.cache_data(show_spinner="Loading NIfTI volumes …")
def load_volumes(case_id: str):
    """Load flair, t1ce, and seg NIfTI volumes for a given case."""
    case_path = os.path.join(TRAIN_DATASET_PATH, case_id)

    flair = nib.load(os.path.join(case_path, f"{case_id}_flair.nii")).get_fdata()
    t1ce = nib.load(os.path.join(case_path, f"{case_id}_t1ce.nii")).get_fdata()
    seg = nib.load(os.path.join(case_path, f"{case_id}_seg.nii")).get_fdata()

    return flair, t1ce, seg


# ──────────────────── Preprocessing / inference ───────────────────
def preprocess_for_prediction(flair_vol, t1ce_vol):
    """Prepare a (VOLUME_SLICES, IMG_SIZE, IMG_SIZE, 2) input tensor."""
    X = np.empty((VOLUME_SLICES, IMG_SIZE, IMG_SIZE, 2))
    for j in range(VOLUME_SLICES):
        X[j, :, :, 0] = cv2.resize(
            flair_vol[:, :, j + VOLUME_START_AT], (IMG_SIZE, IMG_SIZE)
        )
        X[j, :, :, 1] = cv2.resize(
            t1ce_vol[:, :, j + VOLUME_START_AT], (IMG_SIZE, IMG_SIZE)
        )
    # Normalise exactly as the training notebook does
    max_val = np.max(X)
    if max_val > 0:
        X = X / max_val
    return X


def predict(model, flair_vol, t1ce_vol):
    """Run the model and return predicted class labels."""
    X = preprocess_for_prediction(flair_vol, t1ce_vol)
    preds = model.predict(X, verbose=0)  # (VOLUME_SLICES, IMG_SIZE, IMG_SIZE, 4)
    pred_labels = np.argmax(preds, axis=-1)  # (VOLUME_SLICES, IMG_SIZE, IMG_SIZE)
    return pred_labels
def predict_and_create_gif(model, flair_vol, t1ce_vol):
    """
    Runs model prediction and directly returns GIF bytes.
    """
    X = preprocess_for_prediction(flair_vol, t1ce_vol)

    # Model prediction
    preds = model.predict(X, verbose=0)
    pred_labels = np.argmax(preds, axis=-1)

    frames = []

    for j in range(VOLUME_SLICES):
        # Flair slice
        raw = flair_vol[:, :, j + VOLUME_START_AT]
        resized = cv2.resize(raw, (GIF_DISPLAY_SIZE, GIF_DISPLAY_SIZE))

        if resized.max() > 0:
            flair_img = (resized / resized.max() * 255).astype(np.uint8)
        else:
            flair_img = resized.astype(np.uint8)

        flair_rgb = np.stack([flair_img] * 3, axis=-1)

        # Prediction overlay
        pred_slice = cv2.resize(
            pred_labels[j].astype(np.uint8),
            (GIF_DISPLAY_SIZE, GIF_DISPLAY_SIZE),
            interpolation=cv2.INTER_NEAREST,
        )

        overlay_rgba = seg_to_rgba(pred_slice)
        alpha = overlay_rgba[:, :, 3:4] / 255.0

        blended = flair_rgb.copy()
        for c in range(3):
            blended[:, :, c] = (
                (1 - alpha[:, :, 0]) * flair_rgb[:, :, c]
                + alpha[:, :, 0] * overlay_rgba[:, :, c]
            ).astype(np.uint8)

        # Side-by-side
        frame = np.concatenate([flair_rgb, blended], axis=1)
        frames.append(frame)

    # Convert to GIF
    buf = io.BytesIO()
    imageio.mimsave(buf, frames, format="GIF", duration=0.3, loop=0)
    buf.seek(0)

    return buf.getvalue()

# ──────────────────────── Visualisation ───────────────────────────
def seg_to_rgba(seg_2d):
    """Convert a 2-D label map to an RGBA image using SEG_CMAP."""
    h, w = seg_2d.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    for label, color in SEG_CMAP.items():
        mask = seg_2d == label
        rgba[mask] = color
    return rgba


def render_results(flair_vol, seg_vol, pred_labels, slice_idx):
    """
    Produce a matplotlib figure with 4 panels:
      1. Flair MRI
      2. Ground-truth segmentation
      3. Predicted segmentation
      4. Overlay (Flair + prediction)
    """
    # Prepare the flair slice (resized to IMG_SIZE for consistent display)
    flair_slice = cv2.resize(
        flair_vol[:, :, slice_idx + VOLUME_START_AT], (IMG_SIZE, IMG_SIZE)
    )

    # Ground-truth: remap label 4→3, then resize with nearest-neighbor
    gt_raw = seg_vol[:, :, slice_idx + VOLUME_START_AT].copy()
    gt_raw[gt_raw == 4] = 3
    gt_slice = cv2.resize(
        gt_raw.astype(np.uint8),
        (IMG_SIZE, IMG_SIZE),
        interpolation=cv2.INTER_NEAREST,
    )

    # Predicted slice
    pred_slice = pred_labels[slice_idx]

    # Build overlay
    flair_rgb = np.stack([flair_slice] * 3, axis=-1)
    flair_rgb = (
        (flair_rgb / flair_rgb.max() * 255).astype(np.uint8)
        if flair_rgb.max() > 0
        else flair_rgb.astype(np.uint8)
    )
    overlay_rgba = seg_to_rgba(pred_slice)

    # Blend
    overlay_img = flair_rgb.copy()
    alpha = overlay_rgba[:, :, 3:4] / 255.0
    for c in range(3):
        overlay_img[:, :, c] = (
            (1 - alpha[:, :, 0]) * flair_rgb[:, :, c]
            + alpha[:, :, 0] * overlay_rgba[:, :, c]
        ).astype(np.uint8)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    axes[0].imshow(flair_slice, cmap="gray")
    axes[0].set_title("Flair MRI")
    axes[0].axis("off")

    axes[1].imshow(gt_slice, vmin=0, vmax=3)
    axes[1].set_title("Ground Truth")
    axes[1].axis("off")

    axes[2].imshow(pred_slice, vmin=0, vmax=3)
    axes[2].set_title("Prediction")
    axes[2].axis("off")

    axes[3].imshow(overlay_img)
    axes[3].set_title("Overlay (Flair + Pred)")
    axes[3].axis("off")

    fig.tight_layout()
    return fig


# ──────────────── Tab 2 helpers: GIF + Grid ───────────────────────
@st.cache_data(show_spinner="Generating Flair GIF …")
def create_flair_gif(case_id: str):
    """
    Build an animated GIF from the VOLUME_SLICES Flair slices,
    each resized to GIF_DISPLAY_SIZE×GIF_DISPLAY_SIZE.  Returns raw GIF bytes.
    """
    flair, _, _ = load_volumes(case_id)

    frames = []
    for j in range(VOLUME_SLICES):
        raw = flair[:, :, j + VOLUME_START_AT]
        resized = cv2.resize(raw, (GIF_DISPLAY_SIZE, GIF_DISPLAY_SIZE))
        # Normalise to 0-255 uint8 for imageio
        if resized.max() > 0:
            normed = (resized / resized.max() * 255).astype(np.uint8)
        else:
            normed = resized.astype(np.uint8)
        frames.append(normed)

    buf = io.BytesIO()
    imageio.mimsave(buf, frames, format="GIF", duration=0.2, loop=0)
    buf.seek(0)
    return buf.getvalue()


@st.cache_data(show_spinner="Generating prediction GIF …")
def create_prediction_gif(case_id: str, _pred_labels):
    """
    Build an animated GIF with each frame showing the Flair slice
    side-by-side with its predicted segmentation overlay at
    GIF_DISPLAY_SIZE resolution.

    Parameters
    ----------
    case_id : str
        BraTS case identifier (used as cache key and to load volumes).
    _pred_labels : ndarray (VOLUME_SLICES, IMG_SIZE, IMG_SIZE)
        Predicted class labels.  The leading underscore tells Streamlit
        to skip hashing this argument (predictions are deterministic
        for a given case, so `case_id` is a sufficient cache key).
    """
    flair, _, _ = load_volumes(case_id)
    sz = GIF_DISPLAY_SIZE

    frames = []
    for j in range(VOLUME_SLICES):
        # ── Flair slice → uint8 RGB at display size ──
        raw = flair[:, :, j + VOLUME_START_AT]
        resized = cv2.resize(raw, (sz, sz))
        if resized.max() > 0:
            normed = (resized / resized.max() * 255).astype(np.uint8)
        else:
            normed = resized.astype(np.uint8)
        flair_rgb = np.stack([normed] * 3, axis=-1)  # (sz, sz, 3)

        # ── Upscale prediction labels to display size ──
        pred_slice = cv2.resize(
            _pred_labels[j].astype(np.uint8),
            (sz, sz),
            interpolation=cv2.INTER_NEAREST,
        )
        overlay_rgba = seg_to_rgba(pred_slice)        # (sz, sz, 4)
        alpha = overlay_rgba[:, :, 3:4] / 255.0

        blended = flair_rgb.copy()
        for c in range(3):
            blended[:, :, c] = (
                (1 - alpha[:, :, 0]) * flair_rgb[:, :, c]
                + alpha[:, :, 0] * overlay_rgba[:, :, c]
            ).astype(np.uint8)

        # ── Side-by-side: Flair | Overlay ──
        combined = np.concatenate([flair_rgb, blended], axis=1)  # (sz, 2*sz, 3)
        frames.append(combined)

    buf = io.BytesIO()
    imageio.mimsave(buf, frames, format="GIF", duration=0.3, loop=0)
    buf.seek(0)
    return buf.getvalue()


def render_slice_grid(flair_vol):
    """
    Produce a matplotlib figure showing all VOLUME_SLICES Flair slices
    in a 5-row × 8-column grid.  Each cell is labelled with the
    absolute slice index.
    """
    rows, cols = 5, 8
    fig, axes = plt.subplots(rows, cols, figsize=(16, 10))

    for idx in range(VOLUME_SLICES):
        r, c = divmod(idx, cols)
        abs_slice = idx + VOLUME_START_AT
        raw = flair_vol[:, :, abs_slice]
        resized = cv2.resize(raw, (IMG_SIZE, IMG_SIZE))

        axes[r, c].imshow(resized, cmap="gray")
        axes[r, c].set_title(f"Slice {abs_slice}", fontsize=8)
        axes[r, c].axis("off")

    fig.suptitle("Flair MRI – 40-Slice Grid Overview", fontsize=14, y=1.01)
    fig.tight_layout()
    return fig


# ──────────────────────── Streamlit App ───────────────────────────
def main():
    st.set_page_config(
        page_title="Brain MRI Analysis System",
        page_icon="🧠",
        layout="wide",
    )

    st.title("🧠 Brain MRI Analysis System")
    st.markdown(
        "Perform **brain tumor segmentation** on BraTS-2020 training data "
        "using an U-Net model architecture."
    )

    # ── Sidebar ──
    st.sidebar.header("Controls")

    cases = list_cases()
    if not cases:
        st.error(
            f"No BraTS cases found under `{TRAIN_DATASET_PATH}`.  "
            "Make sure the dataset is in the correct location."
        )
        return

    selected_case = st.sidebar.selectbox("Select MRI Case", cases)

    slice_idx = st.sidebar.slider(
        "Slice index (relative to volume)",
        min_value=0,
        max_value=VOLUME_SLICES - 1,
        value=VOLUME_SLICES // 2,
        help=f"Selects slice (index + {VOLUME_START_AT}) from the original 3-D volume.",
    )

    run_prediction = st.sidebar.button("🔍 Run Segmentation", type="primary")

    # ── Legend ──
    st.sidebar.markdown("---")
    st.sidebar.subheader("Segmentation Legend")
    for label, name in SEGMENT_CLASSES.items():
        if label == 0:
            continue
        color = SEG_CMAP[label][:3]
        hex_color = "#{:02x}{:02x}{:02x}".format(*color)
        st.sidebar.markdown(
            f"<span style='color:{hex_color};font-weight:bold;'>■</span> {name}",
            unsafe_allow_html=True,
        )

    # ── Tabs ──
    tab1, tab2 = st.tabs(["🔬 Single Slice View", "🎞️ Slice Visualization"])

    # ──────────── Tab 1: existing segmentation prediction ─────────
    with tab1:
        if run_prediction:
            model = load_model()

            with st.spinner("Loading volumes …"):
                flair, t1ce, seg = load_volumes(selected_case)

            with st.spinner("Running inference …"):
                pred_labels = predict(model, flair, t1ce)

            # Persist results so Tab 2 can use them
            st.session_state["pred_labels"] = pred_labels
            st.session_state["pred_case"] = selected_case

            st.subheader(
                f"Results — {selected_case}  ·  slice {slice_idx + VOLUME_START_AT}"
            )
            fig = render_results(flair, seg, pred_labels, slice_idx)
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("👈 Select a case and click **Run Segmentation** to begin.")

    # ──────────── Tab 2: GIF + 40-slice grid overview ─────────────
    with tab2:
        st.subheader(f"Volume Overview — {selected_case}")

        # Load volumes once (cached)
        with st.spinner("Loading volumes …"):
            flair, _, _ = load_volumes(selected_case)

        # ── Section 1: Flair GIF animation ──
        st.markdown("### 🎬 Flair Slice Animation")
        gif_bytes = create_flair_gif(selected_case)
        st.image(gif_bytes, caption="Animated Flair MRI slices", use_container_width=True)

        st.markdown("---")

        # ── Section 2: Prediction GIF (runs model on all 40 slices) ──
        st.markdown("### 🧠 Prediction Overlay Animation")

        model = load_model()

        with st.spinner("Loading volumes for prediction …"):
            flair_full, t1ce_full, _ = load_volumes(selected_case)

        with st.spinner("Running inference on 40 slices …"):
            pred_labels = predict(model, flair_full, t1ce_full)

        # Persist so Tab 1 can also benefit
        st.session_state["pred_labels"] = pred_labels
        st.session_state["pred_case"] = selected_case

        pred_gif_bytes = predict_and_create_gif(model, flair_full, t1ce_full)
        st.image(
            pred_gif_bytes,
            caption="Flair (left) · Flair + Predicted Segmentation (right)",
            use_container_width=True,
        )

        st.markdown("---")

        # ── Section 3: 40-slice grid (5 × 8) ──
        st.markdown("### 🔲 40-Slice Grid View")
        grid_fig = render_slice_grid(flair)
        st.pyplot(grid_fig)
        plt.close(grid_fig)


if __name__ == "__main__":
    main()