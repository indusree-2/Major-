import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
import cv2
import numpy as np
import json
import tifffile
from PIL import Image
import os

# ==============================================================
# 🦠 MODEL ARCHITECTURE (Must match Kaggle exactly)
# ==============================================================
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        return self.sigmoid(self.fc(self.avg_pool(x)) + self.fc(self.max_pool(x)))

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        return self.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))

class CBAMBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention()
    def forward(self, x):
        return x * self.ca(x) * self.sa(x)

class VirusClassifier(nn.Module):
    def __init__(self, num_classes, gru_hidden, attn_red, dropout):
        super().__init__()
        backbone = models.mobilenet_v2(pretrained=False)
        self.features = backbone.features
        self.cbam = CBAMBlock(1280, attn_red)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.bi_gru = nn.GRU(1280, gru_hidden, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(gru_hidden * 2, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(512, num_classes),
        )
    def forward(self, x):
        x = self.features(x)
        x = self.cbam(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1).unsqueeze(1)
        gru_out, _ = self.bi_gru(x)
        return self.head(gru_out.squeeze(1))

# ==============================================================
# 🔬 DIAGNOSTIC ENGINES
# ==============================================================
@st.cache_resource
def load_system():
    with open('model_metadata.json', 'r') as f:
        meta = json.load(f)
    
    model = VirusClassifier(
        num_classes=meta['num_classes'],
        gru_hidden=meta['gru_hidden_size'],
        attn_red=meta['attention_reduction'],
        dropout=meta['dropout_rate']
    )
    
    # Load weights with security fix for PyTorch 2.6+
    checkpoint = torch.load('best_virus_model.pth', map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, meta

def process_image(uploaded_file, target_size):
    # Support for .tif and standard formats
    if uploaded_file.name.lower().endswith(('.tif', '.tiff')):
        img = tifffile.imread(uploaded_file)
    else:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)

    if img is None: return None, None
    
    # Standardize to grayscale uint8
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.dtype != np.uint8:
        img = ((img - img.min()) / (img.max() - img.min() + 1e-8) * 255).astype(np.uint8)
    
    raw_gray = img.copy()
    
    # Preprocessing for Engine 1 (DL)
    img_resized = cv2.resize(img, (target_size, target_size))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(img_resized)
    img_rgb = cv2.cvtColor(img_clahe, cv2.COLOR_GRAY2RGB)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    tensor = transform(Image.fromarray(img_rgb)).unsqueeze(0)
    
    return tensor, raw_gray

def engine_2_count(img_gray, min_area, max_area):
    img_blur = cv2.GaussianBlur(cv2.resize(img_gray, (256, 256)), (5, 5), 0)
    thresh_type = cv2.THRESH_BINARY_INV if img_blur.mean() > 127 else cv2.THRESH_BINARY
    binary = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, thresh_type, 21, 5)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)
    count = 0
    annotated = cv2.cvtColor(cv2.resize(img_gray, (256, 256)), cv2.COLOR_GRAY2BGR)
    
    for i in range(1, n_labels):
        if min_area <= stats[i, cv2.CC_STAT_AREA] <= max_area:
            count += 1
            x, y, w, h = stats[i, :4]
            cv2.rectangle(annotated, (x, y), (x+w, y+h), (0, 255, 0), 1)
    return count, annotated

# ==============================================================
# 🖥️ STREAMLIT UI
# ==============================================================
st.set_page_config(page_title="TEM Virus Hybrid Suite", layout="wide", page_icon="🦠")

st.title("🦠 TEM Virus Hybrid Diagnostic Suite")
st.markdown("---")

try:
    model, meta = load_system()
    config = meta['severity_config']
except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.stop()

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📤 Upload TEM Micrograph")
    uploaded_file = st.file_uploader("Choose a .tif or .jpg image...", type=["tif", "tiff", "jpg", "jpeg", "png"])
    
    if uploaded_file:
        tensor, raw_gray = process_image(uploaded_file, meta['image_size'])
        st.image(uploaded_file, caption="Original Upload", use_container_width=True)

with col2:
    if uploaded_file:
        with st.spinner("Executing Dual-Engine Analysis..."):
            # Engine 1: Classification
            with torch.no_grad():
                logits = model(tensor)
                probs = torch.softmax(logits, dim=1)
                conf, pred = torch.max(probs, dim=1)
                virus_key = meta['class_names'][pred.item()]
            
            # Engine 2: Particle Counting
            count, annotated = engine_2_count(raw_gray, config['blob_min_area'], config['blob_max_area'])
            
            # Hybrid Calculation
            bsl = meta['bsl_mapping'].get(virus_key.lower(), 2)
            bsl_w = meta['bsl_weights'][str(bsl)]
            dens_score = min(count / config['density_cap'], 1.0)
            sev_score = (bsl_w * config['w_bsl']) + (dens_score * config['w_density'])
            
            # Staging
            if sev_score < config['mild_threshold']: 
                stage, color = "Stage 1: MILD", "#27ae60"
            elif sev_score < config['high_threshold']: 
                stage, color = "Stage 2: MODERATE", "#e67e22"
            else: 
                stage, color = "Stage 3: HIGH RISK", "#c0392b"

        # RESULTS DASHBOARD
        st.subheader("🩺 Diagnostic Results")
        
        # Identity Card
        st.metric(label="Predicted Pathogen", value=meta['display_names'].get(virus_key.lower(), virus_key))
        
        # Metric Row
        m1, m2, m3 = st.columns(3)
        m1.metric("BSL Level", f"Level {bsl}")
        m2.metric("Particle Count (N)", count)
        m3.metric("Engine 1 Conf.", f"{conf.item()*100:.1f}%")

        # Severity Score
        st.markdown(f"""
        <div style="background-color:{color}; padding:20px; border-radius:10px; text-align:center;">
            <h2 style="color:white; margin:0;">{stage}</h2>
            <p style="color:white; margin:0; font-size:1.2rem;">Hybrid Severity Score: {sev_score:.4f}</p>
        </div>
        """, unsafe_allow_html=True)

        with st.expander("Formula Breakdown"):
            st.write(f"**Species Factor:** BSL-{bsl} weight ({bsl_w}) × {config['w_bsl']} = {bsl_w*config['w_bsl']:.3f}")
            st.write(f"**Load Factor:** Density Score ({dens_score:.3f}) × {config['w_density']} = {dens_score*config['w_density']:.3f}")
            st.write(f"**Total:** {sev_score:.4f}")

        st.subheader("🔍 Engine 2 Visualization")
        st.image(annotated, caption=f"Blob Detection (Found N={count})", use_container_width=True)

else:
    st.info("Please upload a TEM micrograph to begin analysis.")

st.markdown("---")
st.caption("TEM Hybrid Suite | Engine 1: MobileNetV2-CBAM-BiGRU | Engine 2: CV Blob Counter")