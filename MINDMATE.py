import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import time
import speech_recognition as sr
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
import matplotlib.pyplot as plt
import io
import re
from gtts import gTTS
import base64
from PIL import Image

st.set_page_config(page_title="Mindmate", page_icon="🤖", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
* { 
    font-family: 'Inter', sans-serif; 
    box-sizing: border-box;
}
.stApp { 
    background: linear-gradient(135deg, #f5f7fa 0%, #e4e8f0 100%); 
}
.header-container { 
    background: white; 
    border-radius: 16px; 
    padding: 25px; 
    margin-bottom: 25px; 
    box-shadow: 0 4px 20px rgba(0,0,0,0.08); 
    text-align: center;
    border: 1px solid #e1e5eb;
}
.chat-container { 
    background: white; 
    padding: 20px; 
    border-radius: 20px; 
    box-shadow: 0 4px 20px rgba(0,0,0,0.08);
    max-height: 400px; 
    overflow-y: auto; 
    margin-bottom: 20px; 
    border: 1px solid #e1e5eb;
}
.message-container {
    display: flex;
    margin-bottom: 16px;
    align-items: flex-start;
}
.user-message-container {
    justify-content: flex-end;
}
.bot-message-container {
    justify-content: flex-start;
}
.avatar {
    width: 36px;
    height: 36px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    margin: 0 10px;
    flex-shrink: 0;
}
.user-avatar {
    background: linear-gradient(135deg, #4A90E2 0%, #5C9EAD 100%);
    color: white;
    order: 2;
}
.bot-avatar {
    background: linear-gradient(135deg, #009688 0%, #26A69A 100%);
    color: white;
}
.message { 
    border-radius: 18px; 
    padding: 12px 16px; 
    max-width: 70%; 
    word-wrap: break-word; 
    animation: fadeIn 0.3s ease-in; 
    line-height: 1.4;
    position: relative;
}
.user-message { 
    background: linear-gradient(135deg, #4A90E2 0%, #5C9EAD 100%); 
    color: white; 
    box-shadow: 0 2px 10px rgba(74, 144, 226, 0.3);
    border-bottom-right-radius: 4px;
}
.bot-message { 
    background: #f0f4f9; 
    color: #2d3748; 
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
    border-bottom-left-radius: 4px;
}
.message-time {
    font-size: 0.7em;
    opacity: 0.7;
    margin-top: 4px;
    text-align: right;
}
.stButton button { 
    background: linear-gradient(135deg, #4A90E2 0%, #5C9EAD 100%); 
    color: white; 
    border: none; 
    border-radius: 12px; 
    padding: 10px 20px; 
    font-weight: 500; 
    font-size: 14px; 
    transition: all 0.2s ease; 
    box-shadow: 0 2px 8px rgba(74, 144, 226, 0.3);
}
.stButton button:hover { 
    transform: translateY(-1px); 
    box-shadow: 0 4px 12px rgba(74, 144, 226, 0.4); 
    background: linear-gradient(135deg, #5C9EAD 0%, #4A90E2 100%);
}
.stTextInput input { 
    border-radius: 12px; 
    padding: 12px; 
    border: 1px solid #e1e5eb; 
    font-size: 14px; 
}
.stTextInput input:focus { 
    border-color: #4A90E2; 
    box-shadow: 0 0 0 2px rgba(74, 144, 226, 0.2); 
}
.quick-questions { 
    display: flex; 
    flex-wrap: wrap; 
    gap: 8px; 
    margin: 15px 0; 
    justify-content: center;
}
.quick-btn { 
    background: white;
    border: 1px solid #e1e5eb;
    border-radius: 16px; 
    padding: 12px 18px; 
    color: #4A90E2; 
    font-weight: 500; 
    cursor: pointer; 
    transition: all 0.2s ease; 
    box-shadow: 0 2px 6px rgba(0,0,0,0.05);
    font-size: 14px;
    flex: 1;
    min-width: 120px;
    text-align: center;
}
.quick-btn:hover { 
    transform: translateY(-1px); 
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    background: #f7f9fc;
}
.voice-btn { 
    background: linear-gradient(135deg, #4A90E2 0%, #5C9EAD 100%); 
    border: none; 
    border-radius: 50%; 
    width: 50px; 
    height: 50px; 
    display: flex; 
    align-items: center; 
    justify-content: center; 
    cursor: pointer; 
    transition: all 0.2s ease; 
    box-shadow: 0 2px 8px rgba(74, 144, 226, 0.3); 
    margin-left: 8px; 
    color: white;
}
.voice-btn:hover { 
    transform: scale(1.05); 
    box-shadow: 0 4px 12px rgba(74, 144, 226, 0.4); 
}
.emergency-alert { 
    background: linear-gradient(135deg, #ff6b6b 0%, #ee5a52 100%); 
    color: white; 
    padding: 16px; 
    border-radius: 12px; 
    margin: 16px 0; 
    box-shadow: 0 4px 15px rgba(255, 107, 107, 0.3); 
    border: 1px solid rgba(255, 255, 255, 0.2); 
}
.crisis-section { 
    background: white; 
    padding: 16px; 
    border-radius: 16px; 
    margin: 20px 0; 
    box-shadow: 0 4px 20px rgba(0,0,0,0.08); 
    border: 1px solid #e1e5eb;
}
.audio-player { 
    margin-top: 8px; 
}
.typing-indicator {
    display: flex;
    padding: 12px 16px;
    background: #f0f4f9;
    border-radius: 18px;
    border-bottom-left-radius: 4px;
    max-width: 70%;
    align-items: center;
}
.typing-dot {
    width: 8px;
    height: 8px;
    background: #a0aec0;
    border-radius: 50%;
    margin: 0 2px;
    animation: typingAnimation 1.4s infinite ease-in-out;
}
.typing-dot:nth-child(1) { animation-delay: 0s; }
.typing-dot:nth-child(2) { animation-delay: 0.2s; }
.typing-dot:nth-child(3) { animation-delay: 0.4s; }
@keyframes typingAnimation {
    0%, 60%, 100% { transform: translateY(0); }
    30% { transform: translateY(-5px); }
}
@keyframes fadeIn { 
    from { opacity: 0; transform: translateY(8px); } 
    to { opacity: 1; transform: translateY(0); } 
}
.sidebar .sidebar-content {
    background: linear-gradient(135deg, #f5f7fa 0%, #e4e8f0 100%);
}
.progress-container {
    background: white;
    padding: 20px;
    border-radius: 16px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.08);
    margin: 20px 0;
    border: 1px solid #e1e5eb;
}
</style>
""", unsafe_allow_html=True)

if "authenticated" not in st.session_state: st.session_state.authenticated = False
if "user_info" not in st.session_state: st.session_state.user_info = {}
if "messages" not in st.session_state: st.session_state.messages = []
if "last_query" not in st.session_state: st.session_state.last_query = None
if "quick_question" not in st.session_state: st.session_state.quick_question = None
if "session_start" not in st.session_state: st.session_state.session_start = datetime.now()
if "mood_data" not in st.session_state: st.session_state.mood_data = []
if "show_progress" not in st.session_state: st.session_state.show_progress = False
if "processing" not in st.session_state: st.session_state.processing = False
if "voice_text" not in st.session_state: st.session_state.voice_text = ""
if "user_contact" not in st.session_state: st.session_state.user_contact = ""
if "emergency_triggered" not in st.session_state: st.session_state.emergency_triggered = False
if "listening" not in st.session_state: st.session_state.listening = False
if "preferred_lang" not in st.session_state: st.session_state.preferred_lang = "en"

helplines = {
    "india": {
        "national": [
            "AASRA: +91-98204 66726 (24/7)",
            "Vandrevala Foundation: 1860 2662 345",
            "iCall: +91-9152987821",
            "Sumaitri: +91-11-23389090",
            "SNEHA: +91-44-24640050"
        ]
    },
    "default": {
        "national": [
            "International Emergency: 112",
            "Befrienders Worldwide: Visit befrienders.org"
        ]
    }
}

language_codes = {
    "en": "en",
    "hi": "hi",
    "ta": "ta",
    "te": "te"
}

SYSTEM_PROMPT = """You are Mindmate, a compassionate mental health assistant for youth. Provide supportive, non-judgmental guidance on mental health topics. Always maintain a warm, empathetic tone. Offer practical advice and coping strategies. If someone mentions serious issues like self-harm, encourage professional help. Never provide medical diagnoses. Use age-appropriate language. Keep responses concise and helpful."""

@st.cache_resource
def load_chain():
    prompt = ChatPromptTemplate.from_messages([("system", SYSTEM_PROMPT),("human", "{query}")])
    try:
        llm = Ollama(model="llama3")
        llm.temperature = 0.7
    except Exception:
        class FallbackLLM:
            def invoke(self, input): return "I'm here to listen. How are you feeling today?"
        llm = FallbackLLM()
    output_parser = StrOutputParser()
    return prompt | llm | output_parser

chain = load_chain()

def translate_with_ollama(text, target_lang):
    if target_lang == "en":
        return text
    
    lang_names = {"hi": "Hindi", "ta": "Tamil", "te": "Telugu"}
    
    prompt = f"""
    Translate the following English text to {lang_names[target_lang]}.
    Provide only the translation, nothing else.
    
    Text: {text}
    """
    
    try:
        response = chain.invoke({"query": prompt})
        return response.strip()
    except Exception as e:
        return text

def text_to_speech(text, lang_code):
    try:
        tts = gTTS(text=text, lang=lang_code, slow=False)
        audio_file = io.BytesIO()
        tts.write_to_fp(audio_file)
        audio_file.seek(0)
        return audio_file
    except Exception as e:
        return None

def get_audio_player(audio_bytes):
    audio_base64 = base64.b64encode(audio_bytes.read()).decode()
    audio_html = f'''
    <audio controls autoplay style="width: 100%;">
        <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
    </audio>
    '''
    return audio_html

def record_voice():
    st.session_state.listening=True
    r=sr.Recognizer()
    try:
        with sr.Microphone() as source:
            st.info("🎤 Listening... Speak now")
            r.adjust_for_ambient_noise(source,duration=0.5)
            audio=r.listen(source,timeout=5,phrase_time_limit=8)
        try:
            text=r.recognize_google(audio)
            st.session_state.voice_text=text
            st.session_state.quick_question=text
            st.session_state.listening=False
            st.rerun()
        except sr.UnknownValueError: st.error("Could not understand audio. Please try again."); st.session_state.listening=False
        except sr.RequestError: st.error("Speech recognition service unavailable. Please type your message."); st.session_state.listening=False
    except sr.WaitTimeoutError: st.error("No speech detected. Please try again."); st.session_state.listening=False
    except Exception as e: st.error(f"Microphone error: {e}"); st.session_state.listening=False

def analyze_text_emotion(message):
    positive_words=["good","happy","fine","better","okay","great","relaxed","calm","sleep","sleeping","helpful"]
    negative_words=["sad","depressed","bad","angry","upset","lonely","tired","stressed","anxious","panic","worried","hate"]
    anxiety_words=["anxious","panic","panic attack","worried","nervous","overwhelmed","afraid"]
    suicidal_words=["suicide","kill myself","end it all","want to die","harm myself","not worth living"]
    msg=message.lower()
    score=5
    for w in positive_words: 
        if w in msg: score+=1
    for w in negative_words: 
        if w in msg: score-=1
    for w in anxiety_words: 
        if w in msg: score-=1
    if any(w in msg for w in suicidal_words): score=1
    punct_factor=msg.count("!")
    score-=punct_factor
    if score<1: score=1
    if score>10: score=10
    stress_level="Low"; anxiety_level="Low"
    if score<=3: stress_level="High"; anxiety_level="High"
    elif score<=5: stress_level="Moderate"; anxiety_level="Moderate"
    elif score<=7: stress_level="Mild"; anxiety_level="Mild"
    return int(score),stress_level,anxiety_level

def extract_country_from_address(address):
    if not address:
        return None
    address_lower = address.lower()
    country_patterns = {
        'india': r'\bindia\b|\bind\b|\bin\b|\b\+91\b|\b\d{6}\b',
        'united states': r'\busa\b|\bus\b|\bunited states\b|\b\+1\b',
        'united kingdom': r'\buk\b|\bunited kingdom\b|\bgb\b|\bengland\b|\bscotland\b|\bwales\b|\b\+44\b',
        'australia': r'\bau\b|\baustralia\b|\b\+61\b',
        'canada': r'\bcanada\b|\bca\b|\b\+1\b'
    }
    for country, pattern in country_patterns.items():
        if re.search(pattern, address_lower, re.IGNORECASE):
            return country
    return None

def localize_helpline(address):
    if not address:
        return helplines.get("default")
    country = extract_country_from_address(address)
    if country and country in helplines:
        return {"national": helplines[country]["national"]}
    return helplines.get("default")

def plot_progress(mood_records):
    if not mood_records: 
        st.info("No mood records yet")
        return
    
    today = datetime.now().date()
    dates = [today + timedelta(days=i) for i in range(7)]
    
    if len(mood_records) < 7:
        scores = [5] * 7
        for i, record in enumerate(mood_records):
            if i < 7:
                scores[i] = record["mood_score"]
    else:
        scores = [r["mood_score"] for r in mood_records[-7:]]
        dates = [r["date"].date() if isinstance(r["date"], datetime) else r["date"] for r in mood_records[-7:]]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(dates, scores, marker='o', color='#4A90E2', linewidth=2.5, markersize=8)
    ax.fill_between(dates, scores, alpha=0.2, color='#4A90E2')
    ax.set_ylim(0, 10)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Mood Score', fontsize=12)
    ax.set_title('Your Mood Progress (Last 7 Days)', fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    
    for i, score in enumerate(scores):
        ax.annotate(f'{score}', (dates[i], score), textcoords="offset points", xytext=(0,10), ha='center')
    
    buf = io.BytesIO()
    plt.xticks(rotation=45)
    plt.tight_layout()
    fig.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    st.image(buf)
    plt.close(fig)

with st.sidebar:
    st.markdown("### 🤖 Mindmate")
    st.markdown("---")
    
    if st.session_state.authenticated:
        st.markdown(f"Welcome, {st.session_state.user_info.get('name', 'User')}")
        st.markdown(f"Age: {st.session_state.user_info.get('age', '')}")
        
        if st.button("📊 View Progress"):
            st.session_state.show_progress = not st.session_state.show_progress
            
        st.markdown("---")
        st.markdown("#### 🆘 Crisis Resources")
        loc = st.session_state.user_info.get("address", "")
        contacts = localize_helpline(loc)
        
        if "national" in contacts:
            for contact in contacts["national"][:3]:
                st.markdown(f"• {contact}")
        
        st.markdown("---")
        st.markdown("#### 💡 Self-Help Resources")
        st.markdown("- [Breathing exercises](https://example.com)")
        st.markdown("- [Mindfulness techniques](https://example.com)")
        st.markdown("- [Coping strategies](https://example.com)")

st.markdown("<div class='header-container'>", unsafe_allow_html=True)
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    try:
        st.image("1000024964.jpg", width=200)
    except:
        st.markdown("<h1 style='color:#4A90E2;margin:0;'>🤖 Mindmate</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#666;margin:10px 0 0 0;'>Your AI companion for mental wellness</p>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

if not st.session_state.authenticated:
    with st.container():
        st.markdown("<h2 style='text-align:center;color:#4A90E2;'>🤖 Mindmate Registration</h2>", unsafe_allow_html=True)
        with st.form("user_info_form"):
            name=st.text_input("Full Name*")
            email=st.text_input("Email Address*")
            phone=st.text_input("Mobile Number*")
            age=st.number_input("Age*",min_value=10,max_value=100,value=18)
            address=st.text_area("Address*")
            emergency_contact=st.text_input("Emergency Contact*")
            gender=st.selectbox("Gender",["Prefer not to say","Male","Female","Non-binary","Other"])
            concerns=st.multiselect("Support with?",["Anxiety","Depression","Stress","Relationships","School/Work","Self-esteem","Sleep issues","Other"])
            previous_therapy=st.radio("Therapy before?",["No","Yes"])
            preferred_lang=st.selectbox("Preferred Language",["English","Hindi","Tamil","Telugu"],index=0)
            submitted=st.form_submit_button("Start Chatting")
            if submitted:
                if not all([name,email,phone,address,emergency_contact]):
                    st.error("Please fill all required fields")
                else:
                    lang_map = {"English": "en", "Hindi": "hi", "Tamil": "ta", "Telugu": "te"}
                    st.session_state.user_info={"name":name,"email":email,"phone":phone,"age":age,"address":address,"emergency_contact":emergency_contact,"gender":gender,"concerns":concerns,"previous_therapy":previous_therapy}
                    st.session_state.preferred_lang=lang_map[preferred_lang]
                    st.session_state.authenticated=True
                    st.success("Registration complete! Redirecting...")
                    time.sleep(1)
                    st.rerun()
    st.stop()

st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
for role, text, audio_data, timestamp in st.session_state.messages:
    container_class = "user-message-container" if role == "user" else "bot-message-container"
    avatar_class = "user-avatar" if role == "user" else "bot-avatar"
    avatar_text = "You" if role == "user" else "M"
    message_class = "user-message" if role == "user" else "bot-message"
    
    st.markdown(f"<div class='message-container {container_class}'>", unsafe_allow_html=True)
    
    if role == "bot":
        st.markdown(f"<div class='avatar bot-avatar'>{avatar_text}</div>", unsafe_allow_html=True)
    
    st.markdown(f"<div class='message {message_class}'><b>{'You' if role=='user' else 'Mindmate'}:</b><br>{text}<div class='message-time'>{timestamp.strftime('%H:%M')}</div></div>", unsafe_allow_html=True)
    
    if role == "user":
        st.markdown(f"<div class='avatar user-avatar'>{avatar_text}</div>", unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    if audio_data and role == "assistant":
        st.markdown(f'<div class="audio-player">{audio_data}</div>', unsafe_allow_html=True)

if st.session_state.processing:
    st.markdown("<div class='message-container bot-message-container'><div class='avatar bot-avatar'>M</div><div class='typing-indicator'><div class='typing-dot'></div><div class='typing-dot'></div><div class='typing-dot'></div></div></div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div class='quick-questions'>", unsafe_allow_html=True)
col1, col2, col3, col4 = st.columns(4)
with col1:
    if st.button("😰 Anxiety Help", disabled=st.session_state.processing or st.session_state.listening, use_container_width=True): 
        st.session_state.quick_question="I've been feeling anxious lately, what can I do?"
with col2:
    if st.button("📚 School Stress", disabled=st.session_state.processing or st.session_state.listening, use_container_width=True): 
        st.session_state.quick_question="How to manage stress from school?"
with col3:
    if st.button("😴 Sleep Tips", disabled=st.session_state.processing or st.session_state.listening, use_container_width=True): 
        st.session_state.quick_question="Tips for better sleep?"
with col4:
    if st.button("💪 Self-Confidence", disabled=st.session_state.processing or st.session_state.listening, use_container_width=True): 
        st.session_state.quick_question="How to build self-confidence?"
st.markdown("</div>", unsafe_allow_html=True)

input_cols = st.columns([5, 1])
with input_cols[0]:
    user_input = st.text_input(
        "Type your message here:", 
        value=st.session_state.quick_question if st.session_state.quick_question else "",
        key="user_input", 
        placeholder="How are you feeling today?", 
        disabled=st.session_state.processing or st.session_state.listening,
        label_visibility="collapsed"
    )
with input_cols[1]:
    if st.button("🎤", disabled=st.session_state.processing or st.session_state.listening, use_container_width=True):
        record_voice()

if st.button("Send", disabled=st.session_state.processing or st.session_state.listening or not user_input, use_container_width=True):
    st.session_state.processing = True
    current_time = datetime.now()
    st.session_state.messages.append(("user", user_input, None, current_time))
    score, stress, anxiety = analyze_text_emotion(user_input)
    st.session_state.mood_data.append({"date": current_time, "mood_score": score})
    
    if score <= 3: 
        st.session_state.emergency_triggered = True
    
    try: 
        response = chain.invoke({"query": user_input})
    except Exception: 
        response = "I'm having trouble responding right now. Please try again."
    
    lang = st.session_state.preferred_lang
    
    if lang != "en":
        response = translate_with_ollama(response, lang)
    
    lang_code = language_codes.get(lang, "en")
    
    audio_html = ""
    try:
        audio_bytes = text_to_speech(response, lang_code)
        if audio_bytes:
            audio_html = get_audio_player(audio_bytes)
    except:
        pass
    
    response_with_analysis = f"{response}\n\nMood score: {score}/10\nStress: {stress}\nAnxiety: {anxiety}"
    st.session_state.messages.append(("assistant", response_with_analysis, audio_html, datetime.now()))
    st.session_state.quick_question = None
    st.session_state.processing = False
    st.rerun()

if st.session_state.show_progress:
    st.markdown("<div class='progress-container'>", unsafe_allow_html=True)
    st.markdown("### 📈 Your Mood Progress")
    plot_progress(st.session_state.mood_data)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div class='crisis-section'>", unsafe_allow_html=True)
st.markdown("### 📞 Crisis Support")
st.markdown("If you're in immediate crisis, please contact:")

loc = st.session_state.user_info.get("address", "")
contacts = localize_helpline(loc)

detected_country = extract_country_from_address(loc)
if detected_country:
    st.markdown(f"Detected Location: {detected_country.title()}")

if "national" in contacts:
    st.markdown("National Helplines:")
    for contact in contacts["national"]:
        st.markdown(f"• {contact}")

st.markdown("Your Emergency Contact:")
st.markdown(f"• {st.session_state.user_info.get('emergency_contact', 'Not provided')}")

if st.session_state.emergency_triggered:
    loc = st.session_state.user_info.get("address", "")
    contacts = localize_helpline(loc)
    html_contacts = ""
    if "national" in contacts: 
        for c in contacts["national"]: 
            html_contacts += f"<p>{c}</p>"
    st.markdown(f"<div class='emergency-alert'><h3>🚨 Immediate Support Available</h3><p>If you're in crisis, please contact emergency services or a crisis hotline:</p>{html_contacts}</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)
