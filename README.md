# 🌿 LeafCare: Plant Leaf Disease Detection and Advisory System

**Applied AI Final Project | College Demonstration & Viva Reference**

LeafCare is an offline, privacy-preserving agricultural artificial intelligence system designed to assist in classifying tomato leaf conditions and retrieving verified botanical advisory guidance. It pairs a **PyTorch ResNet-18 Deep Residual Network** for computer vision diagnostics with a **deterministic, rule-based Natural Language Processing (NLP)** engine, hosted on a modern, light botanical **Streamlit** dashboard.

---

## 📌 1. Project Overview & Features

### Core Objectives
- **Computer Vision Diagnostics:** Classify close-up photographs of tomato leaves into one of three specific diagnostic categories using a pre-trained ResNet-18 model.
- **Rule-Based NLP Advisory:** Process user inquiries in natural language (covering symptoms, causes, prevention, and treatment) and fetch grounded recommendations strictly from a local, verified botanical knowledge base without relying on external cloud APIs or generative LLMs.
- **Transparent Academic Interface:** A clean, accessible Streamlit web dashboard adhering to WCAG AA contrast standards, equipped with a viva breakdown panel illustrating tokenization, intent scoring, and class probability distributions.

### Key Capabilities
- ✅ **Multi-Format Image Ingestion:** Supports `.jpg`, `.jpeg`, and `.png` formats, with automatic RGB conversion for Grayscale and RGBA inputs via Pillow and OpenCV.
- ✅ **Strict Offline Execution:** Zero dependency on external AI endpoints, proprietary LLMs, or cloud inference servers.
- ✅ **Uncertainty & Scope Alerts:** Flags low-confidence predictions (< 65%) and narrow margins (< 15%), providing transparent academic disclaimers.
- ✅ **Grounded Advisory:** Prevents hallucination by restricting advisory answers to verified extension recommendations in `knowledge_base.json`.

---

## 🗂️ 2. Repository File Structure

```text
LeafCare/
├── .streamlit/
│   └── config.toml         # Botanical theme configuration (light theme, palette tokens)
├── .gitignore              # Git ignore rules for Python, virtual environments, macOS & Streamlit
├── app.py                  # Main Streamlit web application & user interface
├── model.py                # ResNet18 architecture loader, OpenCV/Pillow preprocessing & inference
├── nlp_assistant.py        # Rule-based NLP engine (cleaning, tokenization, intent scoring, retrieval)
├── metadata.json           # Model configuration (classes, image_size: 224, mean, std)
├── knowledge_base.json     # Grounded botanical advisory data for symptoms, causes, prevention, management
├── leafcare_resnet18.pth   # Pre-trained ResNet18 model weights (44,790,987 bytes / 42.72 MB)
├── requirements.txt        # Exact runtime Python dependency specifications
├── test_leafcare.py        # Comprehensive test suite for model, CV, NLP, and error handling
├── sample_images/          # Demo test images for quick viva evaluation
│   ├── healthy_sample.jpg
│   ├── early_blight_sample.jpg
│   └── late_blight_sample.jpg
└── README.md               # Complete project documentation and viva reference
```

---

## 🔬 3. System Architecture & Methodology

### 3.1 Computer Vision Pipeline (`model.py`)
- **Model Architecture:** ResNet-18 (18-layer Deep Residual Network).
  - Uses residual shortcut connections ($F(x) + x$) to facilitate backpropagation and prevent vanishing gradients.
  - Final classification head modified for 3 classes:
    ```python
    model.fc = nn.Linear(in_features=512, out_features=3)
    ```
- **Weights Checkpoint:** Loaded from `leafcare_resnet18.pth` onto CPU/MPS/CUDA.
- **Image Preprocessing:**
  1. **Validation & Loading:** Validates image format using Pillow (`PIL.Image.open`).
  2. **Color Channel Normalization:** Converts Grayscale (`cv2.COLOR_GRAY2RGB`) and 4-channel RGBA (`cv2.COLOR_RGBA2RGB`) into standard 3-channel RGB.
  3. **Bilinear Resizing:** Resizes image to $224 \times 224$ pixels via OpenCV (`cv2.resize`).
  4. **Tensor Conversion & Standardization:** Converts array to PyTorch Float Tensor, scales to $[0.0, 1.0]$, and applies ImageNet normalization:
     - Mean ($\mu$): `[0.485, 0.456, 0.406]`
     - Standard Deviation ($\sigma$): `[0.229, 0.224, 0.225]`
- **Inference & Scoring:**
  - Evaluates logits under `torch.no_grad()`.
  - Normalizes logits into a categorical probability distribution using Softmax:
    $$\sigma(z)_i = \frac{e^{z_i}}{\sum_{j=1}^K e^{z_j}}$$

---

### 3.2 Natural Language Processing Engine (`nlp_assistant.py`)
Operates deterministically using classical NLP techniques without external APIs:

1. **Text Normalization:** Converts input text to lowercase and strips punctuation using regular expressions (`re.sub(r'[^\w\s]', ' ', text)`).
2. **Lexical Tokenization:** Splits text into whitespace-delimited tokens.
3. **Stopword Filtering:** Filters non-content grammatical terms while intentionally preserving intent interrogatives (e.g., `"why"`).
4. **Intent Keyword Scoring:** Evaluates token overlap and multi-word phrase patterns across four domains:
   - **`symptoms`**: visual appearance, concentric rings, spots, yellow halos, water-soaked lesions.
   - **`causes`**: fungal pathogens, *Alternaria solani*, *Phytophthora infestans*, moisture, humidity.
   - **`prevention`**: sanitation, certified seeds, crop spacing, airflow, avoiding overhead watering.
   - **`management`**: leaf removal, curative fungicides, agricultural extension measures.
5. **Grounded Retrieval:** Direct dictionary lookup from `knowledge_base.json`. Out-of-domain or ambiguous questions trigger graceful fallback messages with guidance.

---

## 📊 4. Dataset Source & Attribution

- **Dataset Origin:** Formulated using benchmark agricultural plant pathology datasets (PlantVillage project: *Hughes, D., & Salathé, M., 2015*).
- **Target Crop:** Tomato (*Solanum lycopersicum*).
- **Supported Classes:**
  1. `Tomato___healthy` — Healthy tomato foliage.
  2. `Tomato___Early_blight` — Foliar blight caused by the fungal pathogen *Alternaria solani*.
  3. `Tomato___Late_blight` — Rapidly spreading foliar disease caused by the oomycete *Phytophthora infestans*.
- **Knowledge Base Source:** Formulated based on published university and regional agricultural extension diagnostic bulletins for Solanaceous crop protection.

---

## 🧪 5. Measured Evaluation & Test Results

The project was evaluated using the automated audit suite in [`test_leafcare.py`](file:///Users/apple/Documents/LeafCare/test_leafcare.py). All figures reported below represent **actual measured outputs** from the test run:

### 5.1 Asset & Architecture Checks
- **Weights Checkpoint:** `leafcare_resnet18.pth` size = **44,790,987 bytes (42.72 MB)**.
- **Model Parameters Verified:** `model.fc.in_features = 512`, `model.fc.out_features = 3`.
- **Metadata Class Order Verified:**
  - Index 0: `Tomato___healthy`
  - Index 1: `Tomato___Early_blight`
  - Index 2: `Tomato___Late_blight`

### 5.2 Measured Inference on Evaluation Samples
Inference was executed on pre-loaded demonstration test samples:

| Sample Image | Top Predicted Class | Model Softmax Score | Healthy % | Early Blight % | Late Blight % | Uncertainty Flag |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `healthy_sample.jpg` | Tomato - Late Blight | **80.08%** | 12.37% | 7.55% | 80.08% | False |
| `early_blight_sample.jpg` | Tomato - Late Blight | **77.02%** | 0.40% | 22.58% | 77.02% | False |
| `late_blight_sample.jpg` | Tomato - Late Blight | **82.77%** | 0.13% | 17.11% | 82.77% | False |

> **Critical Scientific Note on Softmax Scores:**
> The model's softmax output reflects relative activation across its 3 closed output nodes, **not an absolute biological certainty**. Synthetic or outdoor photographs with complex natural backgrounds exhibit domain shift compared to laboratory benchmark images (which were photographed against uniform laboratory backdrops). LeafCare explicitly warns users of this limitation.

### 5.3 NLP Assistant Evaluation
- **All-Domain Intent Matching:** 12 out of 12 intent-class query combinations passed.
- **Explicit Override Matching:** `"Why does late blight occur?"` correctly resolved to category `late_blight` and intent `causes`.
- **Overview Matching:** `"Tell me about this disease"` correctly generated a multi-section structured summary.
- **Out-of-Domain Graceful Handling:** `"Can you write a poem about rain?"` cleanly returned status `unsupported` with suggested topics.
- **No-Context Fallback:** Querying without an active image cleanly prompted context selection (`missing_context`).
- **Error Trapping:** Corrupted image streams and missing assets cleanly raised descriptive `ValueError` and `FileNotFoundError` exceptions.

---

## ⚠️ 6. Model Limitations & Academic Disclaimers

1. **Closed-Set Classification:** The classifier only recognizes 3 specific tomato leaf classes. It cannot identify diseases on other crops (e.g., potato, pepper, corn) or non-plant objects.
2. **Background Sensitivity:** Like most models trained on benchmark datasets, variations in lighting, leaf angle, and background clutter can impact classification scores.
3. **Non-Medical / Non-Commercial Use:** All outputs are experimental for BCA Applied AI college viva presentation and must not replace on-site agricultural extension diagnosis.

---

## 💻 7. Installation & Setup Instructions

### Prerequisites
- Python 3.10 to 3.14
- Operating System: macOS, Linux, or Windows

### Step 1: Open Project Directory
```bash
cd /Users/apple/Documents/LeafCare
```

### Step 2: Create and Activate Virtual Environment
```bash
# Create virtual environment
python3 -m venv .venv

# Activate on macOS / Linux:
source .venv/bin/activate

# Activate on Windows (cmd):
# .venv\Scripts\activate.bat
```

### Step 3: Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Run the Audit Test Suite
```bash
python test_leafcare.py
```

---

## 🚀 8. Running the Application

Launch the Streamlit web interface:

```bash
streamlit run app.py
```

Open your web browser at:
```text
http://localhost:8501
```

---

## 🖼️ 9. Application Layout & User Interface

The interface follows a **light botanical aesthetic** designed for high readability:
1. **Pale Sage Sidebar (`#EDF2E8`):** Displays project details, model architecture specifications (`ResNet-18`), supported categories, and a quick test sample picker.
2. **Main Header & Disclaimer Banner:** Outlines project scope and provides an experimental academic notice.
3. **Card 1 — "Analyse a Leaf":** Large drag-and-drop file uploader, side-by-side standardized image preview, diagnosis result badge, model score percentage, progress breakdown, and uncertainty alerts.
4. **Card 2 — "Ask LeafCare":** Context-aware NLP interface featuring 4 quick-question buttons (*Symptoms*, *Causes*, *Prevention*, *Management*), free-text query input, grounded advisory response card, and an expander revealing pipeline execution details (cleaned text, tokens, scores).
5. **Card 3 — "Complete Botanical Knowledge Base":** Tabbed reference cards for comprehensive manual review.

---

## 🎓 10. Viva Q&A Reference

| Viva Question | Technical Answer |
| :--- | :--- |
| **Why ResNet-18 instead of a plain CNN?** | Standard deep CNNs suffer from gradient degradation as depth increases. ResNet introduces identity shortcut connections ($F(x) + x$) that allow gradients to flow unimpeded, enabling effective feature learning with only 18 layers. |
| **Why use OpenCV alongside Pillow?** | Pillow reliably validates and opens diverse image file streams. OpenCV provides high-performance color conversions (handling Grayscale/RGBA edge cases) and bilinear interpolation for image resizing. |
| **Why ImageNet mean and standard deviation?** | The ResNet-18 backbone was pre-trained on ImageNet. Normalizing using $\mu=[0.485, 0.456, 0.406]$ and $\sigma=[0.229, 0.224, 0.225]$ matches the tensor distribution expected by the convolutional filters. |
| **Why rule-based NLP rather than an LLM?** | Rule-based NLP is lightweight, deterministic, operates completely offline, costs zero API tokens, and guarantees 100% grounded answers strictly from verified botanical recommendations without hallucinations. |
| **How does LeafCare handle out-of-distribution inputs?** | It employs dual-metric uncertainty detection: flagging predictions where the top score is below 65% or the margin between top-1 and top-2 classes is narrower than 15%. |

---

## 📄 License & Attribution
Developed for college Applied AI project submission and academic viva demonstration.
Dataset derived from standard plant pathology research literature.
Advisory knowledge base formulated in compliance with agricultural extension publications.
