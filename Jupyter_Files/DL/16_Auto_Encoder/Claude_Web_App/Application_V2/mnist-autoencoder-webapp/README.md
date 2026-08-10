# Signal Restoration Lab — MNIST Denoising Autoencoder Web App

A production-structured Flask application that serves your existing, already-trained
Keras denoising autoencoder (`denoise_image.h5`) for interactive digit reconstruction.

**This app does not train anything.** It is inference-only: it loads your model once
at startup and runs `model.predict(...)` per request.

## Project Overview

Upload a grayscale, handwritten-digit-style image. The app:

1. Converts it to grayscale and resizes it to 28×28 (the model's trained input size).
2. Normalizes pixels to `[0, 1]`.
3. *(Optional)* injects synthetic Gaussian noise, matching the exact noise formula
   used during training — since this model is a **denoiser**, feeding it a clean
   image mostly just returns the same image back. The noise toggle in the UI is what
   actually demonstrates the denoising behavior.
4. Runs the image through the trained autoencoder.
5. Displays the original, reconstructed, and difference ("residual") images side by
   side, along with the Mean Squared Error between original and reconstruction.

### Important: this is a *denoising* autoencoder, not a plain autoencoder

The training notebook (`De_Nosing_Image_AUtoEncoder.ipynb`) trains the model as
`noisy_input → clean_output`, not `clean_input → clean_output`. That's why the UI
includes a **noise intensity slider (0–100%)** — without adding noise, uploading an
already-clean digit mostly demonstrates identity reconstruction, not denoising.
The slider value is sent with each request and applied live, server-side, using the
same noise formula as training (`x + factor * N(0,1)`, clipped to `[0,1]`) — only
the factor itself is now user-controlled per request instead of fixed.

## Architecture

```
User
 ↓
Web UI (index.html / app.js)
 ↓
Flask API (POST /api/predict)
 ↓
Validation (validation_utils.py)
 ↓
Image Preprocessing (image_utils.py)
   grayscale → resize 28x28 → /255.0 → [optional noise+clip] → batch+channel dims
 ↓
Pre-trained Autoencoder (autoencoder_service.py, loaded once at startup)
 ↓
Reconstruction (model.predict)
 ↓
Reconstruction Error (MSE, matching the model's own training loss)
 ↓
Web UI (base64 PNGs + metric, rendered without a page reload)
```

## Model Details (inspected from your notebook, not assumed)

| Property | Value |
|---|---|
| Input shape | `(28, 28, 1)` grayscale |
| Output shape | `(28, 28, 1)` |
| Normalization | `pixel / 255.0` → `[0, 1]` |
| Training noise | `x + 0.3 * N(0,1)`, clipped to `[0, 1]` |
| Architecture | Conv2D(32)→MaxPool→Conv2D(8)→MaxPool→Conv2D(8)→UpSample→Conv2D(32)→UpSample→Conv2D(1) |
| Loss | Mean Squared Error (same metric shown in the UI) |
| Final activation | ReLU (app clips output to `[0,1]` for display only — model itself is untouched) |
| Custom objects | None |
| Save format | `.h5` (legacy HDF5) |

## Installation

```bash
python -m venv venv
```

Activate it:

**Windows:**
```bash
venv\Scripts\activate
```

**Linux/macOS:**
```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Model Setup

Your trained model is already placed at:

```
models/denoise_image.h5
```

If you retrain or replace it, drop the new `.h5` file in the same location (or set
the `MODEL_PATH` environment variable to point elsewhere).

## Run

```bash
python app.py
```

Then open **http://localhost:5000** in your browser.

## API

### `POST /api/predict`

**Request:** `multipart/form-data`

| Field | Type | Required | Description |
|---|---|---|---|
| `file` | file | yes | PNG/JPG/JPEG image |
| `noise_level` | string, float `0.0`–`1.0` | no | Noise factor applied live before prediction (defaults to `0.0` / no noise). Clamped server-side to `[0.0, 1.0]` regardless of what's sent. |

**Response (200):**

```json
{
  "success": true,
  "reconstruction_error": 0.002341,
  "original_image": "data:image/png;base64,...",
  "model_input_image": "data:image/png;base64,...",
  "reconstructed_image": "data:image/png;base64,...",
  "difference_image": "data:image/png;base64,...",
  "noise_applied": true,
  "noise_level": 0.45
}
```

**Response (error, 400/500/503):**

```json
{ "success": false, "error": "Human-readable message" }
```

### `GET /api/health`

Returns whether the model loaded successfully at startup — useful for monitoring.

## Testing

```bash
pytest tests/ -v
```

`tests/test_app.py` covers the Flask layer (routing, validation, error responses) and
doesn't require TensorFlow. `tests/test_prediction.py` exercises the real model and
will skip automatically if TensorFlow or the model file isn't available.

## Troubleshooting

**`ModuleNotFoundError: No module named 'tensorflow'`**
Run `pip install -r requirements.txt` inside your activated virtual environment.

**Model fails to load / `ModelLoadError` in logs at startup**
Confirm `models/denoise_image.h5` exists and wasn't corrupted during transfer. The
app will still start and serve the UI, but `/api/predict` returns `503` until this
is fixed — check `GET /api/health`.

**`ValueError` mentioning incompatible layer/shape on load**
This usually means the `.h5` file was saved with a very different TensorFlow/Keras
major version than what's installed locally. Try matching the TensorFlow version
used in the original Colab notebook (check `!pip show tensorflow` output there), or
re-save the model as `.keras` format from Colab (`model.save("model.keras")`) and
point `MODEL_PATH` at that instead — this does **not** require retraining, just
re-serializing the already-trained model.

**Predictions look identical to the input**
Expected at 0% noise on a denoising model — raise the noise slider to see actual
denoising behavior.

**"Prediction service is temporarily unavailable"**
Check server logs — the app never shows raw tracebacks to the browser by design.
The real error is logged server-side.

## Security Notes

- Uploaded files are validated by extension, MIME type, size (`MAX_CONTENT_LENGTH`),
  and by attempting to actually decode them with Pillow before any processing.
- Images are processed **in memory** — nothing is written to disk under normal
  operation (the `uploads/` folder exists for optional future use, e.g. audit
  logging, but isn't required by the current pipeline).
- No internal filesystem paths are ever returned to the client.
- The Flask secret key and model path are configurable via environment variables
  (see `.env.example`) rather than hardcoded.
