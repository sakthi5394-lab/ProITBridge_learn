"""
app.py
------
Entry point. Run with: python app.py

Loads the trained denoising-autoencoder once at startup, then serves the
web UI and the /api/predict inference endpoint. This process never calls
model.fit(...) - it is inference-only.
"""

from app import create_app

app = create_app()

if __name__ == "__main__":
    # debug=True is convenient for local development; set to False (or use
    # a production WSGI server like gunicorn) before deploying.
    app.run(host="0.0.0.0", port=5000, debug=True)
