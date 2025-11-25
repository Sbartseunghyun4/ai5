# streamlit_app.py

import os, re
from io import BytesIO
import numpy as np
import streamlit as st
from PIL import Image, ImageOps
from fastai.vision.all import *
import gdown

# ======================
# 페이지/스타일
# ======================
st.set_page_config(page_title="Fastai 이미지 분류기", page_icon="🤖", layout="wide")
st.markdown("""
<style>
h1 { color:#1E88E5; text-align:center; font-weight:800; letter-spacing:-0.5px; }
.prediction-box { background:#E3F2FD; border:2px solid #1E88E5; border-radius:12px; padding:22px; text-align:center; margin:16px 0; box-shadow:0 4px 10px rgba(0,0,0,.06);}
.prediction-box h2 { color:#0D47A1; margin:0; font-size:2.0rem; }
.prob-card { background:#fff; border-radius:10px; padding:12px 14px; margin:10px 0; box-shadow:0 2px 6px rgba(0,0,0,.06); }
.prob-bar-bg { background:#ECEFF1; border-radius:6px; width:100%; height:22px; overflow:hidden; }
.prob-bar-fg { background:#4CAF50; height:100%; border-radius:6px; transition:width .5s; }
.prob-bar-fg.highlight { background:#FF6F00; }
.info-grid { display:grid; grid-template-columns:repeat(12,1fr); gap:14px; }
.card { border:1px solid #e3e6ea; border-radius:12px; padding:14px; background:#fff; box-shadow:0 2px 6px rgba(0,0,0,.05); }
.card h4 { margin:0 0 10px; font-size:1.05rem; color:#0D47A1; }
.thumb { width:100%; height:auto; border-radius:10px; display:block; }
.thumb-wrap { position:relative; display:block; }
.play { position:absolute; top:50%; left:50%; transform:translate(-50%,-50%); width:60px; height:60px; border-radius:50%; background:rgba(0,0,0,.55); }
.play:after{ content:''; border-style:solid; border-width:12px 0 12px 20px; border-color:transparent transparent transparent #fff; position:absolute; top:50%; left:50%; transform:translate(-40%,-50%); }
.helper { color:#607D8B; font-size:.9rem; }
.stFileUploader, .stCameraInput { border:2px dashed #1E88E5; border-radius:12px; padding:16px; background:#f5fafe; }
</style>
""", unsafe_allow_html=True)

st.title("이미지 분류기 (Fastai) — 확률 막대 + 라벨별 고정 콘텐츠")

# ======================
# 세션 상태
# ======================
if "img_bytes" not in st.session_state:
    st.session_state.img_bytes = None
if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None

# ======================
# 모델 로드
# ======================
FILE_ID = st.secrets.get("GDRIVE_FILE_ID", "1j9Ilf-SM5276pSVWC9vyxoUI5JSDXn5Z")
MODEL_PATH = st.secrets.get("MODEL_PATH", "model.pkl")

@st.cache_resource
def load_model_from_drive(file_id: str, output_path: str):
    if not os.path.exists(output_path):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output_path, quiet=False)
    return load_learner(output_path, cpu=True)

with st.spinner("🤖 모델 로드 중..."):
    learner = load_model_from_drive(FILE_ID, MODEL_PATH)
st.success("✅ 모델 로드 완료")

labels = [str(x) for x in learner.dls.vocab]
st.write(f"**분류 가능한 항목:** `{', '.join(labels)}`")
st.markdown("---")

# ======================
# 라벨 이름 매핑: 여기를 채우세요!
# 각 라벨당 최대 3개씩 표시됩니다.
# ======================
CONTENT_BY_LABEL: dict[str, dict[str, list[str]]] = {

     labels[0]: {
       "texts": ["상대가 기뻐하고 있습니다.(같이 웃었으면 더 기뻐하게 됨)"],
       "images": ["https://i.namu.wiki/i/9L2HhZhdAUlkD8Lvs_09PPlRNQ5fWnxBHFZ18eEbbfI09erBEoz0v3_sFCSZRfX_hszGYY1a5FNu5Pobv_5azQ.webp"],
       "videos": [""]
     },
     labels[1]: {
       "texts": ["상대가 슬퍼하고 있습니다.(상대를 달래줘야 됨)"],
       "images": ["data:image/png;base64,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"],
       "videos": [""]
     },
     labels[2]: {
       "texts": ["상대가 짜증나하고 있습니다.(상대 기분을 맞춰야지 풀림)"],
       "images": ["data:image/jpeg;base64,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"],
       "videos": [""]
     },
}

# ======================
# 유틸
# ======================
def load_pil_from_bytes(b: bytes) -> Image.Image:
    pil = Image.open(BytesIO(b))
    pil = ImageOps.exif_transpose(pil)
    if pil.mode != "RGB": pil = pil.convert("RGB")
    return pil

def yt_id_from_url(url: str) -> str | None:
    if not url: return None
    pats = [r"(?:v=|/)([0-9A-Za-z_-]{11})(?:\?|&|/|$)", r"youtu\.be/([0-9A-Za-z_-]{11})"]
    for p in pats:
        m = re.search(p, url)
        if m: return m.group(1)
    return None

def yt_thumb(url: str) -> str | None:
    vid = yt_id_from_url(url)
    return f"https://img.youtube.com/vi/{vid}/hqdefault.jpg" if vid else None

def pick_top3(lst):
    return [x for x in lst if isinstance(x, str) and x.strip()][:3]

def get_content_for_label(label: str):
    """라벨명으로 콘텐츠 반환 (texts, images, videos). 없으면 빈 리스트."""
    cfg = CONTENT_BY_LABEL.get(label, {})
    return (
        pick_top3(cfg.get("texts", [])),
        pick_top3(cfg.get("images", [])),
        pick_top3(cfg.get("videos", [])),
    )

# ======================
# 입력(카메라/업로드)
# ======================
tab_cam, tab_file = st.tabs(["📷 카메라로 촬영", "📁 파일 업로드"])
new_bytes = None

with tab_cam:
    cam = st.camera_input("카메라 스냅샷", label_visibility="collapsed")
    if cam is not None:
        new_bytes = cam.getvalue()

with tab_file:
    f = st.file_uploader("이미지를 업로드하세요 (jpg, png, jpeg, webp, tiff)",
                         type=["jpg","png","jpeg","webp","tiff"])
    if f is not None:
        new_bytes = f.getvalue()

if new_bytes:
    st.session_state.img_bytes = new_bytes

# ======================
# 예측 & 레이아웃
# ======================
if st.session_state.img_bytes:
    top_l, top_r = st.columns([1, 1], vertical_alignment="center")

    pil_img = load_pil_from_bytes(st.session_state.img_bytes)
    with top_l:
        st.image(pil_img, caption="입력 이미지", use_container_width=True)

    with st.spinner("🧠 분석 중..."):
        pred, pred_idx, probs = learner.predict(PILImage.create(np.array(pil_img)))
        st.session_state.last_prediction = str(pred)

    with top_r:
        st.markdown(
            f"""
            <div class="prediction-box">
                <span style="font-size:1.0rem;color:#555;">예측 결과:</span>
                <h2>{st.session_state.last_prediction}</h2>
                <div class="helper">오른쪽 패널에서 예측 라벨의 콘텐츠가 표시됩니다.</div>
            </div>
            """, unsafe_allow_html=True
        )

    left, right = st.columns([1,1], vertical_alignment="top")

    # 왼쪽: 확률 막대
    with left:
        st.subheader("상세 예측 확률")
        prob_list = sorted(
            [(labels[i], float(probs[i])) for i in range(len(labels))],
            key=lambda x: x[1], reverse=True
        )
        for lbl, p in prob_list:
            pct = p * 100
            hi = "highlight" if lbl == st.session_state.last_prediction else ""
            st.markdown(
                f"""
                <div class="prob-card">
                  <div style="display:flex;justify-content:space-between;margin-bottom:6px;">
                    <strong>{lbl}</strong><span>{pct:.2f}%</span>
                  </div>
                  <div class="prob-bar-bg">
                    <div class="prob-bar-fg {hi}" style="width:{pct:.4f}%;"></div>
                  </div>
                </div>
                """, unsafe_allow_html=True
            )

    # 오른쪽: 정보 패널 (예측 라벨 기본, 다른 라벨로 바꿔보기 가능)
    with right:
        st.subheader("라벨별 고정 콘텐츠")
        default_idx = labels.index(st.session_state.last_prediction) if st.session_state.last_prediction in labels else 0
        info_label = st.selectbox("표시할 라벨 선택", options=labels, index=default_idx)

        texts, images, videos = get_content_for_label(info_label)

        if not any([texts, images, videos]):
            st.info(f"라벨 `{info_label}`에 대한 콘텐츠가 아직 없습니다. 코드의 CONTENT_BY_LABEL에 추가하세요.")
        else:
            # 텍스트
            if texts:
                st.markdown('<div class="info-grid">', unsafe_allow_html=True)
                for t in texts:
                    st.markdown(f"""
                    <div class="card" style="grid-column:span 12;">
                      <h4>텍스트</h4>
                      <div>{t}</div>
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            # 이미지(최대 3, 3열)
            if images:
                st.markdown('<div class="info-grid">', unsafe_allow_html=True)
                for url in images[:3]:
                    st.markdown(f"""
                    <div class="card" style="grid-column:span 4;">
                      <h4>이미지</h4>
                      <img src="{url}" class="thumb" />
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            # 동영상(유튜브 썸네일)
            if videos:
                st.markdown('<div class="info-grid">', unsafe_allow_html=True)
                for v in videos[:3]:
                    thumb = yt_thumb(v)
                    if thumb:
                        st.markdown(f"""
                        <div class="card" style="grid-column:span 6;">
                          <h4>동영상</h4>
                          <a href="{v}" target="_blank" class="thumb-wrap">
                            <img src="{thumb}" class="thumb"/>
                            <div class="play"></div>
                          </a>
                          <div class="helper">{v}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="card" style="grid-column:span 6;">
                          <h4>동영상</h4>
                          <a href="{v}" target="_blank">{v}</a>
                        </div>
                        """, unsafe_allow_html=True)
else:
    st.info("카메라로 촬영하거나 파일을 업로드하면 분석 결과와 라벨별 콘텐츠가 표시됩니다.")
