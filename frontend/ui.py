import streamlit as st
import requests

# --- Configuration ---
# PASTE YOUR AWS IP HERE (No 'http://', just the numbers)
AWS_SERVER_IP = "51.20.64.173"  # <--- CHANGE THIS!
API_URL = f"http://{AWS_SERVER_IP}:8000/predict"

st.set_page_config(page_title="MarketPulse", page_icon="📈")

# --- UI Design ---
st.title("📈 MarketPulse: Financial Sentiment AI")
st.markdown("""
This AI analyzes financial news headlines to predict market sentiment.
*Backend running on AWS EC2 (t2.micro) with DistilBERT.*
""")

# --- User Input ---
news_text = st.text_area(
    "Enter a Financial Headline:",
    placeholder="e.g., Tesla creates a robot which helps humans..."
)

if st.button("Analyze Sentiment"):
    if not news_text:
        st.warning("Please enter some text first.")
    else:
        with st.spinner("Consulting the AI model on AWS..."):
            try:
                # Send request to your AWS API
                payload = {"text": news_text}
                response = requests.post(API_URL, json=payload)
                
                if response.status_code == 200:
                    result = response.json()
                    sentiment = result['sentiment']
                    confidence = result['confidence']
                    
                    # Display Result
                    st.success("Analysis Complete!")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Sentiment", sentiment.upper())
                    with col2:
                        st.metric("Confidence", f"{confidence * 100:.2f}%")
                        
                    # visual bar for fun
                    if sentiment == "positive":
                        st.progress(confidence)
                    elif sentiment == "negative":
                        st.progress(confidence) # You can customize color logic later
                        
                else:
                    st.error(f"Error {response.status_code}: {response.text}")
                    
            except Exception as e:
                st.error(f"Connection Error: Is the AWS server running? \nDetails: {e}")

# --- Sidebar Info ---
st.sidebar.header("About Project")
st.sidebar.info("End-to-End MLOps Project built with FastAPI, Docker, GitHub Actions, and AWS.")