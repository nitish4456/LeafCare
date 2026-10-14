"""
LeafCare - Plant Leaf Disease Classification & Advisory System
Applied AI Project | BCA AI & ML Demonstration

A clean, modern, minimalist botanical interface for:
1. ResNet18-based Tomato Leaf Disease Classification
2. Rule-based offline Natural Language Processing (NLP) Advisory Assistant
3. Grounded knowledge retrieval from verified agricultural literature
"""

import os
import streamlit as st
from PIL import Image

# Import computer vision model and NLP assistant modules
from model import load_model, preprocess_image, predict, CLASS_DISPLAY_NAMES
from nlp_assistant import NLPAssistant, CLASS_TO_KB_KEY


# -----------------------------------------------------------------------------
# 1. Streamlit Configuration & Botanical Design System
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="LeafCare — Plant Health Companion",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Centralized, high-contrast botanical stylesheet (WCAG AA & AAA compliant)
st.markdown("""
<style>
    /* =========================================================================
       Color Palette Tokens & Global Overrides
       ========================================================================= */
    :root {
        --bg-main: #F7F8F3;          /* Warm off-white */
        --bg-sidebar: #EDF2E8;       /* Pale sage */
        --bg-card: #FFFFFF;          /* Pure white */
        --primary: #426B50;          /* Botanical green */
        --primary-dark: #2F4D3A;     /* Deep botanical green */
        --secondary: #7FA184;        /* Muted sage */
        --text-main: #25352B;        /* Deep forest charcoal (14:1+ contrast) */
        --text-muted: #59675D;       /* Readable muted green-grey (5.5:1+ contrast) */
        --border-subtle: #DCE5D9;    /* Soft grey-green */
        --bg-success: #E7F3E8;       /* Soft botanical green */
        --border-success: #B8D8BA;
        --text-success: #1E432A;
        --bg-warning: #FFF3D9;       /* Soft warm amber */
        --border-warning: #F1D69C;
        --text-warning: #5C4308;
        --bg-error: #FBE9E7;         /* Soft terracotta */
        --border-error: #F4B8AF;
        --text-error: #6E2219;
    }

    /* Main Viewport Background */
    [data-testid="stAppViewContainer"], .main, .block-container {
        background-color: var(--bg-main) !important;
        color: var(--text-main) !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif !important;
    }

    /* Top Streamlit Header */
    header[data-testid="stHeader"] {
        background-color: var(--bg-main) !important;
    }

    /* =========================================================================
       Sidebar Styling (Pale Sage)
       ========================================================================= */
    [data-testid="stSidebar"] {
        background-color: var(--bg-sidebar) !important;
        border-right: 1px solid var(--border-subtle) !important;
    }
    [data-testid="stSidebar"] * {
        color: var(--text-main) !important;
    }
    [data-testid="stSidebar"] hr {
        border-color: var(--border-subtle) !important;
    }
    [data-testid="stSidebar"] .stCaption {
        color: var(--text-muted) !important;
    }

    /* =========================================================================
       Typography & Readability
       ========================================================================= */
    h1, h2, h3, h4, h5, h6 {
        color: var(--text-main) !important;
        font-weight: 600 !important;
        letter-spacing: -0.015em;
    }
    p, span, div, li, label {
        color: var(--text-main);
    }
    strong, b {
        color: var(--text-main) !important;
        font-weight: 600 !important;
    }
    code {
        background-color: #EDF2E8 !important;
        color: #2F4D3A !important;
        border: 1px solid var(--border-subtle) !important;
        border-radius: 4px !important;
        padding: 0.15rem 0.35rem !important;
        font-size: 0.88em !important;
    }

    /* =========================================================================
       Cards: Native st.container(border=True) Overrides
       ========================================================================= */
    [data-testid="stVerticalBlockBorderWrapper"] {
        background-color: var(--bg-card) !important;
        border: 1px solid var(--border-subtle) !important;
        border-radius: 12px !important;
        padding: 1.5rem !important;
        margin-bottom: 1.25rem !important;
        box-shadow: 0 1px 3px rgba(37, 53, 43, 0.04) !important;
    }
    [data-testid="stVerticalBlockBorderWrapper"] * {
        color: var(--text-main);
    }

    /* =========================================================================
       Academic Disclaimer / Warning Banner
       ========================================================================= */
    .botanical-notice-card {
        background-color: var(--bg-warning);
        border: 1px solid var(--border-warning);
        border-left: 4px solid #D99B00;
        border-radius: 8px;
        padding: 0.9rem 1.2rem;
        margin-bottom: 1.5rem;
        font-size: 0.92rem;
        line-height: 1.55;
        color: var(--text-warning);
    }
    .botanical-notice-card strong {
        color: #4A3500 !important;
    }

    /* Native Alert Overrides */
    div[data-testid="stAlert"] {
        border-radius: 8px !important;
        border-width: 1px !important;
        border-style: solid !important;
    }
    div[data-testid="stAlert"] * {
        color: inherit !important;
    }

    /* =========================================================================
       Diagnosis Result Box & Badges
       ========================================================================= */
    .result-display-panel {
        background-color: #FFFFFF;
        border: 1px solid var(--border-subtle);
        border-radius: 10px;
        padding: 1.25rem;
        margin-bottom: 1rem;
    }
    .result-badge {
        display: inline-block;
        padding: 0.4rem 0.9rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 1.05rem;
        margin: 0.3rem 0 0.85rem 0;
    }
    .badge-healthy {
        background-color: var(--bg-success);
        color: var(--text-success);
        border: 1px solid var(--border-success);
    }
    .badge-early {
        background-color: var(--bg-warning);
        color: var(--text-warning);
        border: 1px solid var(--border-warning);
    }
    .badge-late {
        background-color: var(--bg-error);
        color: var(--text-error);
        border: 1px solid var(--border-error);
    }

    /* =========================================================================
       Advisory Response Box (High Contrast & Clear Readability)
       ========================================================================= */
    .advisory-container {
        background-color: #FFFFFF;
        border: 1px solid var(--border-subtle);
        border-left: 4px solid var(--primary);
        border-radius: 8px;
        padding: 1.25rem 1.5rem;
        margin: 1.25rem 0;
        color: var(--text-main);
        box-shadow: 0 1px 3px rgba(37, 53, 43, 0.03);
    }
    .advisory-tag {
        display: inline-block;
        background-color: var(--bg-sidebar);
        color: var(--primary);
        border: 1px solid var(--border-subtle);
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        padding: 0.25rem 0.65rem;
        border-radius: 4px;
        margin-bottom: 0.75rem;
    }
    .advisory-body {
        color: var(--text-main) !important;
        font-size: 0.98rem;
        line-height: 1.65;
    }
    .advisory-body strong {
        color: var(--primary) !important;
    }
    .advisory-body p {
        color: var(--text-main) !important;
        margin-bottom: 0.6rem;
    }

    /* =========================================================================
       Form Inputs & Controls (High Contrast)
       ========================================================================= */
    div[data-testid="stTextInput"] input {
        background-color: #FFFFFF !important;
        color: var(--text-main) !important;
        border: 1.5px solid var(--border-subtle) !important;
        border-radius: 8px !important;
        padding: 0.65rem 0.85rem !important;
        font-size: 0.95rem !important;
    }
    div[data-testid="stTextInput"] input:focus {
        border-color: var(--primary) !important;
        box-shadow: 0 0 0 2px rgba(66, 107, 80, 0.15) !important;
        outline: none !important;
    }
    div[data-testid="stTextInput"] input::placeholder {
        color: #7A897E !important;
    }

    /* Dropdowns / Selectboxes */
    div[data-baseweb="select"] > div {
        background-color: #FFFFFF !important;
        border: 1.5px solid var(--border-subtle) !important;
        border-radius: 8px !important;
        color: var(--text-main) !important;
    }
    div[data-baseweb="select"] span {
        color: var(--text-main) !important;
    }

    /* File Uploader Dropzone */
    [data-testid="stFileUploader"] section {
        background-color: #FFFFFF !important;
        border: 1.5px dashed var(--secondary) !important;
        border-radius: 10px !important;
        padding: 1.5rem !important;
    }
    [data-testid="stFileUploader"] section * {
        color: var(--text-main) !important;
    }
    [data-testid="stFileUploader"] button {
        background-color: var(--bg-sidebar) !important;
        color: var(--primary) !important;
        border: 1px solid var(--border-subtle) !important;
        border-radius: 6px !important;
        font-weight: 500 !important;
    }

    /* Buttons */
    div.stButton > button {
        background-color: #FFFFFF !important;
        color: var(--primary) !important;
        border: 1.5px solid var(--border-subtle) !important;
        border-radius: 8px !important;
        padding: 0.5rem 1rem !important;
        font-size: 0.9rem !important;
        font-weight: 500 !important;
        transition: all 0.15s ease-in-out !important;
        width: 100% !important;
    }
    div.stButton > button:hover {
        background-color: var(--bg-sidebar) !important;
        border-color: var(--primary) !important;
        color: var(--primary-dark) !important;
    }

    /* Progress Indicators */
    div[data-testid="stProgress"] > div > div > div > div {
        background-color: var(--primary) !important;
    }
    div[data-testid="stProgress"] > div > div {
        background-color: #E8ECE5 !important;
    }

    /* Expanders */
    [data-testid="stExpander"] {
        background-color: #FFFFFF !important;
        border: 1px solid var(--border-subtle) !important;
        border-radius: 10px !important;
    }
    [data-testid="stExpander"] summary {
        color: var(--text-main) !important;
        font-weight: 600 !important;
    }

    /* Tab Headers */
    button[data-baseweb="tab"] {
        color: var(--text-muted) !important;
        font-weight: 500 !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: var(--primary) !important;
        font-weight: 600 !important;
        border-bottom-color: var(--primary) !important;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 2. Cached Resource Loaders (Loaded once for performance & stability)
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading LeafCare ResNet18 model and configuration...")
def get_model_and_metadata():
    """
    Loads PyTorch weights and metadata once, caching in memory across interactions.
    """
    weights_path = "leafcare_resnet18.pth"
    metadata_path = "metadata.json"
    
    if not os.path.exists(weights_path):
        st.error(f"Missing Model Weights: `{weights_path}` was not found in project root.")
        st.stop()
    if not os.path.exists(metadata_path):
        st.error(f"Missing Metadata: `{metadata_path}` was not found in project root.")
        st.stop()
        
    return load_model(weights_path, metadata_path, device="cpu")


@st.cache_resource(show_spinner="Initializing LeafCare Rule-Based NLP Assistant...")
def get_nlp_assistant():
    """
    Loads knowledge base and initializes NLP assistant once.
    """
    kb_path = "knowledge_base.json"
    if not os.path.exists(kb_path):
        st.error(f"Missing Knowledge Base: `{kb_path}` was not found in project root.")
        st.stop()
    return NLPAssistant(kb_path)


# Initialize components with error trapping
try:
    model, metadata = get_model_and_metadata()
    nlp_assistant = get_nlp_assistant()
except Exception as init_err:
    st.error(f"Initialization Error: {init_err}")
    st.stop()


# -----------------------------------------------------------------------------
# 3. Sidebar: Pale Sage Botanical Theme
# -----------------------------------------------------------------------------
with st.sidebar:
    # Sidebar Header
    st.markdown("## 🌿 LeafCare")
    st.caption("Plant Health Companion • College Applied AI Project")
    st.markdown("<hr style='margin: 0.8rem 0; border: none; border-top: 1px solid #DCE5D9;'>", unsafe_allow_html=True)
    
    # Model Information Section
    st.markdown("### 🔬 Model Information")
    st.markdown(f"""
    - **Architecture:** {metadata.get('architecture', 'ResNet18')} (PyTorch)
    - **Resolution:** {metadata.get('image_size', 224)} × {metadata.get('image_size', 224)} px
    - **Weights Checkpoint:** `leafcare_resnet18.pth`
    - **Preprocessing:** Pillow & OpenCV (RGB, Resize, Normalization)
    """)
    
    st.markdown("<hr style='margin: 0.8rem 0; border: none; border-top: 1px solid #DCE5D9;'>", unsafe_allow_html=True)
    
    # Supported Categories
    st.markdown("### 🎯 Supported Classes (3)")
    st.markdown("""
    - 🟢 **Tomato — Healthy**
    - 🟠 **Tomato — Early Blight** (*Alternaria solani*)
    - 🔴 **Tomato — Late Blight** (*Phytophthora infestans*)
    """)
    
    st.markdown("<hr style='margin: 0.8rem 0; border: none; border-top: 1px solid #DCE5D9;'>", unsafe_allow_html=True)
    
    # Quick Test Sample Picker
    st.markdown("### 🧪 Demo Sample Leaves")
    sample_options = ["None (Upload custom leaf)"]
    sample_dir = "sample_images"
    sample_files = {
        "Healthy Tomato Leaf": "healthy_sample.jpg",
        "Early Blight Leaf": "early_blight_sample.jpg",
        "Late Blight Leaf": "late_blight_sample.jpg"
    }
    
    if os.path.exists(sample_dir):
        sample_options.extend(list(sample_files.keys()))
        
    selected_sample = st.selectbox(
        "Load demo leaf for evaluation:",
        options=sample_options,
        index=0,
        help="Select a pre-loaded sample leaf or upload your own photo in the main section."
    )


# -----------------------------------------------------------------------------
# 4. Main Page Header & Academic Scope Notice
# -----------------------------------------------------------------------------
st.markdown("<h1 style='font-size: 2.15rem; margin-bottom: 0.15rem; color: #25352B;'>LeafCare</h1>", unsafe_allow_html=True)
st.markdown("<p style='font-size: 1.05rem; color: #59675D; margin-bottom: 1.2rem;'>Tomato Leaf Classification & Advisory</p>", unsafe_allow_html=True)

st.markdown("""
<p style='color: #25352B; font-size: 0.95rem; line-height: 1.6; max-width: 900px; margin-bottom: 1.25rem;'>
    LeafCare is an Applied AI diagnostic tool that evaluates tomato leaf health using an 18-layer Deep Residual Network 
    (ResNet-18) and provides verified agricultural guidance through an offline, rule-based natural language advisory engine.
</p>
""", unsafe_allow_html=True)

# Mandatory Academic Scope & Limitation Warning
st.markdown("""
<div class="botanical-notice-card">
    <strong>⚠️ Experimental Academic Scope & Evaluation Notice:</strong><br>
    This diagnostic model is specifically trained <strong>only on three tomato leaf classes</strong> (Healthy, Early Blight, and Late Blight). 
    It cannot diagnose other plant species, unsupported diseases, or non-leaf objects. 
    Predictions and advisory outputs are strictly experimental for college project demonstration and must not replace certified agricultural extension guidance.
</div>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 5. Section: Analyse a Leaf (Computer Vision Pipeline)
# -----------------------------------------------------------------------------
with st.container(border=True):
    st.markdown("<h3 style='margin-top: 0; font-size: 1.3rem; color: #25352B;'>🔍 Analyse a Leaf</h3>", unsafe_allow_html=True)
    st.markdown("<p style='color: #59675D; font-size: 0.92rem; margin-bottom: 1.2rem;'>Upload a leaf photo or select a demo sample to evaluate disease classification and model confidence.</p>", unsafe_allow_html=True)

    col_upload, col_guidance = st.columns([3, 2], gap="medium")

    with col_upload:
        uploaded_file = st.file_uploader(
            "Upload tomato leaf photograph (JPG, JPEG, PNG):",
            type=["jpg", "jpeg", "png"],
            help="Upload a clear close-up picture of a tomato leaf."
        )

    with col_guidance:
        st.markdown("""
        <div style="background-color: #FAFCF9; border: 1px solid #DCE5D9; border-radius: 8px; padding: 1rem 1.2rem; font-size: 0.88rem; color: #59675D; line-height: 1.55;">
            <strong style="color: #25352B;">Tips for reliable analysis:</strong><br>
            • Capture the leaf in clear, natural lighting.<br>
            • Center lesion spots or discoloration in the frame.<br>
            • Supported formats: JPG, JPEG, and PNG.
        </div>
        """, unsafe_allow_html=True)

    # Determine active image source (uploaded file takes priority over demo sample)
    active_image = None
    active_image_source_name = ""

    if uploaded_file is not None:
        active_image = uploaded_file
        active_image_source_name = f"Uploaded File: {uploaded_file.name}"
    elif selected_sample != "None (Upload custom leaf)" and selected_sample in sample_files:
        sample_path = os.path.join(sample_dir, sample_files[selected_sample])
        if os.path.exists(sample_path):
            active_image = sample_path
            active_image_source_name = f"Demo Sample: {selected_sample}"

    prediction_result = None

    # If an image is provided, run preprocessing and model inference
    if active_image is not None:
        try:
            # Preprocessing matching training configuration (Pillow + OpenCV)
            input_tensor, display_img = preprocess_image(active_image, metadata)
            
            # ResNet18 forward pass
            prediction_result = predict(model, input_tensor, metadata, device="cpu")
            
            # Layout: Left column = Display image, Right column = Results card
            st.markdown("<hr style='margin: 1.5rem 0; border: none; border-top: 1px solid #DCE5D9;'>", unsafe_allow_html=True)
            col_img, col_results = st.columns([1, 1], gap="large")
            
            with col_img:
                st.image(
                    display_img,
                    caption=f"Standardized Input ({active_image_source_name}) — 224 × 224 px",
                    use_container_width=True
                )
                
            with col_results:
                predicted_class = prediction_result["display_name"]
                raw_class = prediction_result["raw_class"]
                confidence = prediction_result["confidence"]
                conf_percent = prediction_result["confidence_percentage"]
                
                # Badge color selection
                badge_class = "badge-healthy"
                if "Early" in predicted_class:
                    badge_class = "badge-early"
                elif "Late" in predicted_class:
                    badge_class = "badge-late"
                    
                st.markdown(f"""
                <div class="result-display-panel">
                    <span style="font-size: 0.82rem; text-transform: uppercase; letter-spacing: 0.05em; color: #59675D; font-weight: 600;">Predicted Category</span><br>
                    <div style="margin: 0.35rem 0 0.85rem 0;">
                        <span class="result-badge {badge_class}">{predicted_class}</span>
                    </div>
                    <div style="font-size: 0.95rem; color: #25352B;">
                        <strong>Model Score:</strong> <span style="font-size: 1.25rem; font-weight: 700; color: #426B50;">{conf_percent}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Uncertainty warning if score is below threshold or separation is narrow
                if prediction_result["is_uncertain"]:
                    st.warning(
                        f"⚠️ **Uncertainty Alert:** " +
                        " ".join(prediction_result["uncertainty_reasons"]) +
                        " The model only recognizes three tomato leaf classes; atypical leaves or unlisted conditions will yield inconclusive classifications."
                    )
                else:
                    st.info(
                        "ℹ️ **Evaluation Note:** Model score reflects categorical softmax output on trained features. Always cross-examine visual symptoms with botanical references."
                    )
                    
                # Class Probability Distribution Breakdown
                st.markdown("<h4 style='font-size: 0.95rem; margin-top: 1rem; margin-bottom: 0.5rem; color: #25352B;'>Categorical Model Score Breakdown</h4>", unsafe_allow_html=True)
                for c_name, c_data in prediction_result["probabilities"].items():
                    p_val = c_data["probability"]
                    p_str = c_data["percentage"]
                    lbl = c_data["display_name"]
                    st.markdown(f"<div style='font-size: 0.88rem; color: #25352B; margin-bottom: 2px;'><strong>{lbl}</strong>: {p_str}</div>", unsafe_allow_html=True)
                    st.progress(min(max(p_val, 0.0), 1.0))

        except Exception as img_err:
            st.error(f"Image Processing Error: {img_err}")
            st.info("Please verify that the file is an uncorrupted image in JPG, JPEG, or PNG format.")

    else:
        st.markdown("""
        <div style="background-color: #FAFCF9; border: 1px solid #DCE5D9; border-radius: 8px; padding: 1.25rem; text-align: center; color: #59675D; font-size: 0.92rem; margin-top: 0.5rem;">
            Select a sample leaf from the sidebar or upload a tomato leaf image above to view diagnostic classification.
        </div>
        """, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 6. Section: Ask LeafCare (NLP Advisory Assistant)
# -----------------------------------------------------------------------------
with st.container(border=True):
    st.markdown("<h3 style='margin-top: 0; font-size: 1.3rem; color: #25352B;'>💬 Ask LeafCare</h3>", unsafe_allow_html=True)
    st.markdown("<p style='color: #59675D; font-size: 0.92rem; margin-bottom: 1.2rem;'>Ask questions regarding symptoms, causes, prevention, and management. Answers are strictly retrieved from verified local botanical knowledge.</p>", unsafe_allow_html=True)

    # Active Diagnosis Context Resolution
    context_class = prediction_result["raw_class"] if prediction_result else None
    context_display = prediction_result["display_name"] if prediction_result else "None (Upload or diagnose a leaf first)"

    col_ctx_info, col_ctx_override = st.columns([3, 2], gap="medium")

    with col_ctx_info:
        st.markdown(f"""
        <div style="font-size: 0.95rem; color: #25352B; padding-top: 0.4rem;">
            <strong>Active Diagnosis Context:</strong> <span style="color: #426B50; font-weight: 600;">{context_display}</span>
        </div>
        """, unsafe_allow_html=True)

    with col_ctx_override:
        manual_category_override = st.selectbox(
            "Or query a specific category directly:",
            options=["Use Active Prediction", "Tomato — Healthy", "Tomato — Early Blight", "Tomato — Late Blight"],
            index=0,
            help="Allows asking questions about a specific category without changing the uploaded image."
        )

    override_mapping = {
        "Tomato — Healthy": "Tomato___healthy",
        "Tomato — Early Blight": "Tomato___Early_blight",
        "Tomato — Late Blight": "Tomato___Late_blight"
    }

    if manual_category_override != "Use Active Prediction":
        active_nlp_class = override_mapping[manual_category_override]
    else:
        active_nlp_class = context_class

    # Quick question suggestions (4 distinct buttons)
    st.markdown("<p style='font-size: 0.9rem; font-weight: 600; color: #25352B; margin: 1rem 0 0.5rem 0;'>Frequently Asked Questions:</p>", unsafe_allow_html=True)
    btn_c1, btn_c2, btn_c3, btn_c4 = st.columns(4)

    selected_quick_q = None
    with btn_c1:
        if st.button("🔍 Symptoms", use_container_width=True):
            selected_quick_q = "What are the symptoms?"
    with btn_c2:
        if st.button("🦠 Causes", use_container_width=True):
            selected_quick_q = "What causes this disease?"
    with btn_c3:
        if st.button("🛡️ Prevention", use_container_width=True):
            selected_quick_q = "How can I prevent this disease?"
    with btn_c4:
        if st.button("💊 Management", use_container_width=True):
            selected_quick_q = "What is the recommended management?"

    # User text question input
    user_question = st.text_input(
        "Enter your question for the advisory system:",
        value=selected_quick_q if selected_quick_q else "",
        placeholder="e.g., What are the common symptoms of this disease?",
        help="Type in natural language questions about symptoms, causes, prevention, or treatment."
    )

    if user_question:
        if not active_nlp_class:
            st.warning(
                "⚠️ Please upload a leaf image or select a disease category from the dropdown above before querying the advisory system."
            )
        else:
            # Execute rule-based NLP pipeline
            nlp_res = nlp_assistant.query(user_question, active_nlp_class)
            status = nlp_res["status"]
            intent_label = nlp_res.get("intent") or "Advisory"
            
            # Display answer inside styled advisory card
            with st.container(border=True):
                st.markdown(
                    f'<span class="advisory-tag">Topic: {intent_label.replace("_", " ").title()}</span>',
                    unsafe_allow_html=True
                )
                st.markdown(nlp_res["response"])
            
            # Viva demonstration expander: NLP Pipeline breakdown
            with st.expander("🔬 View NLP Pipeline Execution Details (BCA Viva Demonstration)"):
                st.markdown(f"""
                - **Raw Input:** `{user_question}`
                - **Cleaned Text:** `{nlp_assistant.clean_text(user_question)}`
                - **Filtered Tokens:** `{nlp_res['tokens']}`
                - **Resolved Category:** `{nlp_res['category']}`
                - **Identified Intent:** `{nlp_res['intent']}`
                """)
                st.markdown("**Intent Keyword Scoring Breakdown:**")
                st.json(nlp_res["intent_scores"])
                st.caption("Adheres strictly to deterministic rule-based NLP: Lowercase normalization → Punctuation removal → Tokenization → Stopword filtering → Keyword matching → Grounded JSON retrieval.")


# -----------------------------------------------------------------------------
# 7. Section: Complete Botanical Knowledge Base Reference (Tabs)
# -----------------------------------------------------------------------------
if active_nlp_class:
    kb_key = CLASS_TO_KB_KEY.get(active_nlp_class, "healthy")
    all_guidance = nlp_assistant.get_all_guidance(kb_key)
    c_title = CLASS_DISPLAY_NAMES.get(kb_key, kb_key.title())
    
    with st.expander(f"📖 Browse Complete Knowledge Base Reference for {c_title}"):
        tab1, tab2, tab3, tab4 = st.tabs(["🔍 Symptoms", "🦠 Causes", "🛡️ Prevention", "💊 Management"])
        with tab1:
            st.markdown(f"<p style='color: #25352B; line-height: 1.65;'>{all_guidance.get('symptoms', 'No information available.')}</p>", unsafe_allow_html=True)
        with tab2:
            st.markdown(f"<p style='color: #25352B; line-height: 1.65;'>{all_guidance.get('causes', 'No information available.')}</p>", unsafe_allow_html=True)
        with tab3:
            st.markdown(f"<p style='color: #25352B; line-height: 1.65;'>{all_guidance.get('prevention', 'No information available.')}</p>", unsafe_allow_html=True)
        with tab4:
            st.markdown(f"<p style='color: #25352B; line-height: 1.65;'>{all_guidance.get('management', 'No information available.')}</p>", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 8. Page Footer
# -----------------------------------------------------------------------------
st.markdown("<hr style='margin: 2.5rem 0 1rem 0; border: none; border-top: 1px solid #DCE5D9;'>", unsafe_allow_html=True)
st.markdown(
    "<div style='text-align: center; color: #59675D; font-size: 0.85rem; line-height: 1.5;'>"
    "LeafCare — Plant Leaf Disease Classification & Advisory System<br>"
    "Developed with PyTorch, Torchvision, OpenCV, Pillow, and Streamlit. Strictly offline & grounded."
    "</div>",
    unsafe_allow_html=True
)
