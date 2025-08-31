import streamlit as st
import pandas as pd
import os
import uuid
import json
import langdetect
from deep_translator import GoogleTranslator
from streamlit_cookies_manager import EncryptedCookieManager
import base64
import re
import stanza
from sentence_transformers import SentenceTransformer, util
import torch
from rapidfuzz import process, fuzz
import openpyxl

st.set_page_config(page_title="Nasa Space Apps ChatBot", layout="wide")

# -------------------- إعداد Stanza مرة واحدة --------------------
@st.cache_resource
def load_stanza_pipeline():
    stanza.download('ar', verbose=False)
    return stanza.Pipeline('ar', processors='tokenize', verbose=False)

nlp_ar = load_stanza_pipeline()

@st.cache_data
def preprocess_arabic(text):
    doc = nlp_ar(text)
    tokens = [word.text for sent in doc.sentences for word in sent.words]
    normalized = ' '.join(tokens).lower()
    normalized = re.sub("[إأآا]", "ا", normalized)
    normalized = re.sub("ى", "ي", normalized)
    normalized = re.sub("ؤ", "و", normalized)
    normalized = re.sub("ئ", "ي", normalized)
    normalized = re.sub("ة", "ه", normalized)
    normalized = re.sub("[ًٌٍَُِّْ]", "", normalized)
    return normalized

def detect_language(text):
    try:
        return langdetect.detect(text)
    except:
        return 'en'

# -------------------- تحميل موديل الـ Embeddings --------------------
@st.cache_resource
def load_model():
    return SentenceTransformer("intfloat/multilingual-e5-base")

model = load_model()

def get_embedding(text, is_query=False):
    prefix = "query: " if is_query else "passage: "
    return model.encode(prefix + text, convert_to_tensor=True, normalize_embeddings=True)

# -------------------- تحميل البيانات --------------------
excel_file = "cleaned.xlsx"

@st.cache_data
def load_data():
    if os.path.exists(excel_file):
        df = pd.read_excel(excel_file)
    else:
        df = pd.DataFrame(columns=["Question", "Answer"])
    
    processed_questions = []
    for q in df['Question'].dropna():
        if detect_language(q) == 'ar':
            processed_questions.append(preprocess_arabic(q))
        else:
            processed_questions.append(q.lower().strip())
    
    # احسب embeddings لكل الأسئلة
    embeddings = [get_embedding(q) for q in processed_questions] if processed_questions else []
    if embeddings:
        embeddings = torch.stack(embeddings)
    else:
        embeddings = torch.empty((0, model.get_sentence_embedding_dimension()))
    
    question_to_index = {processed_questions[i]: i for i in range(len(processed_questions))}
    
    return df, processed_questions, question_to_index, embeddings

df, processed_questions, question_to_index, question_embeddings = load_data()

# -------------------- البحث السريع باستخدام fuzzy --------------------
def get_closest_match_fast(user_q):
    match = process.extractOne(user_q, processed_questions, scorer=fuzz.ratio, score_cutoff=30)
    if match:
        return question_to_index[match[0]]
    return None

# -------------------- جلب الإجابة --------------------
def get_answer(user_question):
    global df, processed_questions, question_to_index, question_embeddings

    original_question = user_question.strip()
    if not original_question:
        return "Please enter a question."

    # تحديد اللغة ومعالجة النص
    lang = detect_language(original_question)
    if lang == 'ar':
        user_q = preprocess_arabic(original_question)
    else:
        user_q = original_question.lower().strip()

    # 1- Fuzzy search أولاً
    idx = get_closest_match_fast(user_q)
    if idx is not None:
        answer = df.iloc[idx]['Answer']
        if pd.notna(answer) and answer.strip():
            return answer
        else:
            return "I found a similar question but the answer is not available yet."

    # 2- Semantic search
    user_emb = get_embedding(user_q, is_query=True)
    if len(processed_questions) > 0:
        cos_scores = util.cos_sim(user_emb, question_embeddings)[0]
        best_idx = torch.argmax(cos_scores).item()
        best_score = cos_scores[best_idx].item()
        answer = df.iloc[best_idx]['Answer']
        if pd.notna(answer) and answer.strip():
            # لو الدقة منخفضة لكن فيه إجابة موجودة، نعرضها مع تحذير
            if best_score < 0.45:
                return f"(Low confidence: {best_score:.2f}) {answer}"
            return answer

    # 3- نص جديد → إضافة للـ Excel والذاكرة
    new_row = pd.DataFrame([{"Question": original_question, "Answer": ""}])
    df = pd.concat([df, new_row], ignore_index=True)
    processed_questions.append(user_q)
    question_to_index[user_q] = len(processed_questions) - 1
    new_emb = get_embedding(user_q)
    question_embeddings = torch.cat([question_embeddings, new_emb.unsqueeze(0)], dim=0)

    # لو نص طويل/ارشادي → نعلم المستخدم
    if len(original_question.split()) > 20:
        return "This seems like an informative statement. I have noted it and can answer related questions later."

    # النص جديد تمام
    return "I have noted your question and will be able to answer similar questions in the future."


# -------------------- حفظ البيانات --------------------
def save_excel():
    df.to_excel(excel_file, index=False)

# -------------------- إدارة المحادثات --------------------
qa_file = "chat_data.json"

def load_qa():
    if os.path.exists(qa_file):
        with open(qa_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_qa(data):
    with open(qa_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def delete_conversation_by_id(chat_id):
    data = load_qa()
    data = [item for item in data if item['chat_id'] != chat_id]
    save_qa(data)
    st.success("Chat deleted.")

def delete_conversations_by_user_id(user_id):
    data = load_qa()
    data = [item for item in data if item['user_id'] != user_id]
    save_qa(data)
    st.success("All your chats have been deleted.")

# -------------------- الخلفية والواجهات --------------------
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

background_image = "slider5.png"
base64_img = get_base64_of_bin_file(background_image)
st.markdown(f"""
<style>
[data-testid="stAppViewContainer"] {{
    background-image: url("data:image/png;base64,{base64_img}");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
}}
.block-container {{
    background-color: transparent !important;
}}
</style>
""", unsafe_allow_html=True)

# -------------------- Cookies --------------------
cookies = EncryptedCookieManager(prefix="chatbot_", password="Nasa_AS-2025")
if not cookies.ready():
    st.stop()
if not cookies.get("user_id"):
    cookies["user_id"] = str(uuid.uuid4())
    cookies.save()
user_id = cookies.get("user_id")

# ------------------- واجهة المستخدم --------------------
all_chats = load_qa()
user_chats = [c for c in all_chats if c['user_id'] == user_id]
chat_ids = sorted(set(c['chat_id'] for c in user_chats), reverse=True)

st.sidebar.title("Chat History")
if "chat_id" not in st.session_state:
    st.session_state.chat_id = None

if st.sidebar.button("New Chat", key="new_chat"):
    st.session_state.chat_id = str(uuid.uuid4())
    st.rerun()

if not st.session_state.chat_id and chat_ids:
    st.session_state.chat_id = chat_ids[0]  
elif not st.session_state.chat_id:
    st.session_state.chat_id = str(uuid.uuid4())

st.sidebar.markdown("### Select a chat:")
for cid in chat_ids:
    chat_label = f"Chat: {cid[:8]}..."
    col1, col2 = st.sidebar.columns([0.8, 0.2])
    if col1.button(chat_label, key=f"select_{cid}"):
        st.session_state.chat_id = cid
        st.rerun()
    if col2.button("🗑", key=f"delete_{cid}"):
        delete_conversation_by_id(cid)
        if st.session_state.chat_id == cid:
            st.session_state.chat_id = None
        st.rerun()

if st.sidebar.button("🗑 Delete all my chats"):
    delete_conversations_by_user_id(user_id)
    st.session_state.chat_id = str(uuid.uuid4())
    st.rerun()

chat_id = st.session_state.chat_id
current_chat = [c for c in user_chats if c['chat_id'] == chat_id]

# -------------------- شعار و عنوان --------------------
logo_path = "logo.jpg"
logo_base64 = get_base64_of_bin_file(logo_path)
st.markdown(f"""
<div style="display: flex; align-items: center; justify-content: flex-end; margin-bottom: 20px;">
    <img src="data:image/jpeg;base64,{logo_base64}" width="200">
</div>
""", unsafe_allow_html=True)

st.markdown("""
<h1 style='color: white; text-align: center;'>
    Nasa Space Apps Assiut ChatBot
</h1>
""", unsafe_allow_html=True)

st.markdown("""
<style>
.stChatMessage[data-testid="stChatMessage"] div[data-testid="stMarkdownContainer"] p {
    background-color: white;
    color: black;
    padding: 10px 15px;
    border-radius: 15px;
    display: inline-block;
    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

# -------------------- عرض المحادثة --------------------
for msg in current_chat:
    with st.chat_message("user"):
        st.markdown(msg["question"])
    with st.chat_message("assistant"):
        st.markdown(msg["answer"])

# -------------------- إدخال المستخدم --------------------
if user_input := st.chat_input("Ask me anything..."):
    answer = get_answer(user_input)

    with st.chat_message("user"):
        st.markdown(user_input)
    with st.chat_message("assistant"):
        st.markdown(answer)

    all_chats.append({
        "user_id": user_id,
        "chat_id": chat_id,
        "question": user_input,
        "answer": answer
    })
    save_qa(all_chats)
    save_excel()


