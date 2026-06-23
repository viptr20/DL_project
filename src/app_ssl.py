"""Streamlit web app for interactive search of similar images using SSL.

The user uploads an image, and the app displays the closest images from COIL-100 based on SSL embeddings.
"""

import io
from pathlib import Path

import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from config.config import Config
from src.models import ResNet18SSL
from src.dataset import Coil100Dataset


MODEL_PATH = "./checkpoints/resnet18_ssl.pt"


def build_transform(image_size: int = 128):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])


@st.cache_resource
def load_model_and_data():
    cfg = Config()
    device = torch.device(cfg.device if torch.cuda.is_available() or str(cfg.device) == "cpu" else "cpu")

    model = ResNet18SSL(cfg.embedding_dim).to(device)

    state = torch.load(MODEL_PATH, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    model.load_state_dict(state, strict=True)
    model.eval()

    dataset = Coil100Dataset(cfg.data_dir, image_size=128, n_views=1)
    transform = build_transform(128)

    feats = []
    paths = []

    with torch.no_grad():
        for sample in dataset:
            x = sample["views"][0].unsqueeze(0).to(device)
            h, _ = model(x)
            h = F.normalize(h, dim=1)
            feats.append(h.cpu())
            paths.append(sample["path"])

    feats = torch.cat(feats, dim=0)

    return model, transform, feats, paths, device


def extract_feature(model, image, transform, device):
    x = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        h, _ = model(x)
        h = F.normalize(h, dim=1)
    return h.cpu()


def main():
    st.set_page_config(page_title="Групиране на визуални обекти", layout="wide")
    st.title("Самообучение за групиране на визуални обекти")
    st.write("Качи изображение и виж най-близките подобни обекти от COIL-100.")

    if not Path(MODEL_PATH).exists():
        st.error(f"Липсва модел файл: {MODEL_PATH}")
        st.stop()

    uploaded = st.file_uploader("Качи изображение", type=["png", "jpg", "jpeg"])

    if uploaded is None:
        st.info("Моля, качете изображение за търсене на подобни.")
        return

    try:
        img = Image.open(io.BytesIO(uploaded.read())).convert("RGB")
    except Exception as e:
        st.error(f"Грешка при отваряне на изображението: {e}")
        st.stop()

    st.image(img, caption="Входно изображение", use_container_width=True)

    try:
        model, transform, feats, paths, device = load_model_and_data()
    except FileNotFoundError as e:
        st.error(f"Липсва файл: {e}")
        st.stop()
    except RuntimeError as e:
        st.error(f"Грешка при зареждане на модела: {e}")
        st.stop()
    except Exception as e:
        st.error(f"Неочаквана грешка: {e}")
        st.stop()

    q = extract_feature(model, img, transform, device)
    sims = torch.matmul(feats, q.T).squeeze(1)
    topk = torch.topk(sims, k=min(5, len(paths))).indices.tolist()

    st.subheader("Най-близки изображения")
    cols = st.columns(len(topk))

    for col, idx in zip(cols, topk):
        path = paths[idx]
        try:
            im = Image.open(path).convert("RGB")
            score = float(sims[idx].item())
            col.image(im, caption=f"{Path(path).name}\ncos={score:.4f}", use_container_width=True)
        except Exception:
            col.write(Path(path).name)


if __name__ == "__main__":
    main()