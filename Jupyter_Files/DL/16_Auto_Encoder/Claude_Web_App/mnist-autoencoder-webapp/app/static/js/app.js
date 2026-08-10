(() => {
  "use strict";

  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("file-input");
  const chooseBtn = document.getElementById("choose-file-btn");
  const clearBtn = document.getElementById("clear-btn");

  const emptyState = document.getElementById("dropzone-empty");
  const previewState = document.getElementById("dropzone-preview");
  const previewImg = document.getElementById("preview-img");

  const noiseToggle = document.getElementById("noise-toggle");
  const reconstructBtn = document.getElementById("reconstruct-btn");
  const reconstructBtnText = document.getElementById("reconstruct-btn-text");
  const btnSpinner = document.getElementById("btn-spinner");
  const errorMessage = document.getElementById("error-message");

  const resultsPanel = document.getElementById("results-panel");
  const resultsEmpty = document.getElementById("results-empty");
  const noisyCard = document.getElementById("noisy-card");

  const originalImg = document.getElementById("original-img");
  const noisyImg = document.getElementById("noisy-img");
  const reconstructedImg = document.getElementById("reconstructed-img");
  const differenceImg = document.getElementById("difference-img");
  const mseValue = document.getElementById("mse-value");
  const metricInterpretation = document.getElementById("metric-interpretation");
  const completeTag = document.getElementById("complete-tag");

  const statusDot = document.getElementById("status-dot");
  const statusText = document.getElementById("status-text");

  let selectedFile = null;

  // ------------------------- Model health check -------------------------
  fetch("/api/health")
    .then((r) => r.json())
    .then((data) => {
      if (data.model_loaded) {
        statusDot.classList.add("online");
        statusText.textContent = "model ready";
      } else {
        statusDot.classList.add("offline");
        statusText.textContent = "model unavailable";
      }
    })
    .catch(() => {
      statusDot.classList.add("offline");
      statusText.textContent = "status unknown";
    });

  // ------------------------- File selection -------------------------
  function showPreview(file) {
    const reader = new FileReader();
    reader.onload = (e) => {
      previewImg.src = e.target.result;
      emptyState.hidden = true;
      previewState.hidden = false;
    };
    reader.readAsDataURL(file);
  }

  function handleFile(file) {
    if (!file) return;
    hideError();
    selectedFile = file;
    showPreview(file);
    reconstructBtn.disabled = false;
  }

  function resetUpload() {
    selectedFile = null;
    fileInput.value = "";
    emptyState.hidden = false;
    previewState.hidden = true;
    reconstructBtn.disabled = true;
    hideError();
  }

  chooseBtn.addEventListener("click", () => fileInput.click());
  dropzone.addEventListener("click", (e) => {
    if (e.target === clearBtn) return;
    if (previewState.hidden) fileInput.click();
  });
  dropzone.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      fileInput.click();
    }
  });

  fileInput.addEventListener("change", () => {
    if (fileInput.files && fileInput.files[0]) {
      handleFile(fileInput.files[0]);
    }
  });

  clearBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    resetUpload();
  });

  ["dragenter", "dragover"].forEach((evt) => {
    dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropzone.classList.add("dragover");
    });
  });
  ["dragleave", "drop"].forEach((evt) => {
    dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropzone.classList.remove("dragover");
    });
  });
  dropzone.addEventListener("drop", (e) => {
    const file = e.dataTransfer.files && e.dataTransfer.files[0];
    if (file) handleFile(file);
  });

  // ------------------------- Error display -------------------------
  function showError(message) {
    errorMessage.textContent = message;
    errorMessage.hidden = false;
  }
  function hideError() {
    errorMessage.hidden = true;
    errorMessage.textContent = "";
  }

  // ------------------------- Loading state -------------------------
  function setLoading(isLoading) {
    reconstructBtn.disabled = isLoading || !selectedFile;
    btnSpinner.hidden = !isLoading;
    reconstructBtnText.textContent = isLoading ? "Analyzing Image..." : "Reconstruct Signal";
  }

  // ------------------------- Reconstruct -------------------------
  reconstructBtn.addEventListener("click", async () => {
    if (!selectedFile) return;
    hideError();
    completeTag.hidden = true;
    setLoading(true);

    const formData = new FormData();
    formData.append("file", selectedFile);
    formData.append("add_noise", noiseToggle.checked ? "true" : "false");

    try {
      const response = await fetch("/api/predict", {
        method: "POST",
        body: formData,
      });
      const data = await response.json();

      if (!response.ok || !data.success) {
        showError(data.error || "Something went wrong. Please try again.");
        setLoading(false);
        return;
      }

      renderResults(data);
    } catch (err) {
      showError("Could not reach the prediction service. Please try again.");
    } finally {
      setLoading(false);
    }
  });

  function renderResults(data) {
    originalImg.src = data.original_image;
    reconstructedImg.src = data.reconstructed_image;
    differenceImg.src = data.difference_image;

    if (data.noise_applied) {
      noisyImg.src = data.model_input_image;
      noisyCard.hidden = false;
    } else {
      noisyCard.hidden = true;
    }

    mseValue.textContent = data.reconstruction_error.toFixed(6);
    metricInterpretation.textContent = interpretError(data.reconstruction_error);

    resultsEmpty.hidden = true;
    resultsPanel.hidden = false;
    completeTag.hidden = false;
  }

  function interpretError(mse) {
    if (mse < 0.01) {
      return "The Autoencoder reconstructed the image with low reconstruction error.";
    }
    if (mse < 0.03) {
      return "The Autoencoder reconstructed the image with moderate reconstruction error.";
    }
    return "The Autoencoder reconstructed the image with high reconstruction error " +
      "- the input may differ substantially from the MNIST digits it was trained on.";
  }
})();
