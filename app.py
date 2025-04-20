import requests
import streamlit as st
import os
import time
import json  # Needed for error handling

# Load Hugging Face token from secrets or environment
HF_TOKEN = st.secrets["HF_TOKEN"] if "HF_TOKEN" in st.secrets else os.getenv("HF_TOKEN")
API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-large"
headers = {"Authorization": f"Bearer {HF_TOKEN}"}

# Streamlit page config
st.set_page_config(page_title="🌍 Smart Travel Chatbot", layout="centered")

# Custom CSS for better UI
st.markdown("""
    <style>
    .stApp {
        background-color: #0f1117;
        color: #ffffff;
        font-family: 'Segoe UI', sans-serif;
    }
    .title {
        font-size: 2.5rem;
        font-weight: bold;
        color: #4CD3C2;
    }
    .desc {
        font-size: 1.1rem;
        color: #cccccc;
    }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.markdown("<div class='title'>🌍 Smart Travel Chatbot</div>", unsafe_allow_html=True)
st.markdown("<div class='desc'>Plan your next adventure! Ask anything about food, hotels, or attractions in any Pakistani city.</div>", unsafe_allow_html=True)

# Sidebar inputs + suggested examples
with st.sidebar:
    st.header("Plan Your Trip ✈️")
    city_input = st.text_input("Enter a City (e.g., Lahore, Hunza, Swat)")
    user_query = st.text_area("Ask something (e.g., best food, top hotels, tourist places)")
    
    st.markdown("#### 💡 Try Examples:")
    st.caption("• Best food spots in Quetta?")
    st.caption("• Places to visit in Swat during winter?")
    st.caption("• Famous hotels in Skardu?")
    st.caption("• Top attractions in Multan for families")

# Hugging Face API call with retry
@st.cache_data(show_spinner=False)
def ask_huggingface(city, query, max_retries=4, retry_delay=10):
    prompt = (
        f"As a Pakistani travel expert, list 3 to 5 beautiful tourist places someone should visit in {city} during summer. "
        f"Answer in a friendly tone, using full sentences. Do not repeat any place.\n\n"
        f"Question: {query}\nAnswer:"
    )

    for attempt in range(max_retries):
        try:
            response = requests.post(API_URL, headers=headers, json={"inputs": prompt})
            response.raise_for_status()
            result = response.json()

            if isinstance(result, dict) and result.get("error"):
                if "is overloaded" in result["error"]:
                    st.warning(f"Hugging Face API is overloaded. Retrying in {retry_delay} seconds... (Attempt {attempt + 1}/{max_retries})")
                    time.sleep(retry_delay)
                else:
                    return f"❌ Hugging Face API Error: {result['error']}"
            else:
                return result[0]["generated_text"]
        except requests.exceptions.RequestException as e:
            st.error(f"Request error (Attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                return f"❌ Request failed after {max_retries} attempts: {e}"
        except json.JSONDecodeError as e:
            st.error(f"JSON Decode Error (Attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                return f"❌ JSON Decode Error after {max_retries} attempts: {e}"

    return "❌ Failed to get a response from Hugging Face API."

# Button interaction
if st.button("Ask the Bot"):
    if not city_input.strip() or not user_query.strip():
        st.warning("⚠️ Please enter both a city and your question.")
    else:
        with st.spinner("Thinking..."):
            answer = ask_huggingface(city_input.strip(), user_query.strip())
        st.success(answer)
else:
    st.info("👈 Use the sidebar to ask your travel questions about any city in Pakistan.")
