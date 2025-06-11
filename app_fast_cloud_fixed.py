
import streamlit as st
import json
import torch
import requests
from PIL import Image
from io import BytesIO
from sentence_transformers import SentenceTransformer, util
import os

st.set_page_config(page_title="Prop Matcher (Cloud)", layout="wide")

GDRIVE_FILE_ID = "1d40Wv1FEPhhc3Tmu93kFmvGDOhARSNLt"
EMBEDDING_PATH = "precomputed_embeddings.pt"

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()
    response = session.get(URL, params={"id": id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {"id": id, "confirm": token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            return value
    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768
    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:
                f.write(chunk)

@st.cache_resource
def download_and_load_embeddings():
    if not os.path.exists(EMBEDDING_PATH):
        with st.spinner("Κατέβασμα embeddings από Google Drive..."):
            download_file_from_google_drive(GDRIVE_FILE_ID, EMBEDDING_PATH)
    bundle = torch.load(EMBEDDING_PATH, map_location=torch.device("cpu"))
    return bundle["items"], bundle["embeddings"]

@st.cache_resource
def load_model():
    return SentenceTransformer("clip-ViT-B-32")

def compute_image_embedding(img, model):
    return model.encode(img, convert_to_tensor=True)

def show_result(item):
    st.image(item["image"], width=180)
    st.markdown(f"**{item['name']}**")
    st.markdown(f"💰 {item['price']}")
    st.markdown(f"[🔗 Προβολή προϊόντος]({item['url']})")

# Load resources
model = load_model()
items, embeddings = download_and_load_embeddings()

# UI
st.title("⚡ Prop Matcher (Cloud Ready)")
st.markdown("Ανεβάστε φωτογραφία ή κάντε αναζήτηση για να βρείτε αντίστοιχα αντικείμενα (Αττική).")

tab1, tab2 = st.tabs(["📷 Αναζήτηση με εικόνα", "🔤 Αναζήτηση με κείμενο"])

with tab1:
    uploaded = st.file_uploader("Ανεβάστε φωτογραφία αντικειμένου", type=["jpg", "jpeg", "png"])
    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="Εικόνα αντικειμένου", width=300)
        with st.spinner("Ανάλυση εικόνας..."):
            query_emb = compute_image_embedding(image, model)
            scores = util.cos_sim(query_emb, embeddings)[0]
            top_k = torch.topk(scores, k=5)

        st.subheader("🔗 Πιο κοντινά αντικείμενα:")
        for score, idx in zip(top_k.values, top_k.indices):
            show_result(items[idx])

with tab2:
    query = st.text_input("Αναζήτηση με λέξεις-κλειδιά (π.χ. ξύλινη, μεταλλική, μαύρη):")
    if query:
        matches = [item for item in items if query.lower() in item["name"].lower()]
        st.subheader(f"🔍 Βρέθηκαν {len(matches)} αποτελέσματα:")
        for item in matches:
            show_result(item)
