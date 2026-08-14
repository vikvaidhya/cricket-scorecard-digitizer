import streamlit as st
import json
import pandas as pd
import google.generativeai as genai
from PIL import Image

st.set_page_config(layout="wide", page_title="Cricket Scorecard OCR Tester")

st.title("🏏 Paper Scorecard Digitizer & Reconstruction Engine")
st.write("Upload a handwritten scorecard picture to digitize and validate match statistics.")

# --- API KEY SECRETS HANDLING ---
# 1. Check if GEMINI_API_KEY exists in Streamlit Cloud Secrets
api_key = st.secrets.get("GEMINI_API_KEY", None)

# 2. If not found in Secrets, show the sidebar input box as a fallback
if not api_key:
    raw_api_key = st.sidebar.text_input("Enter Gemini API Key", type="password")
    api_key = raw_api_key.strip().strip('"').strip("'") if raw_api_key else ""
else:
    st.sidebar.success("🔑 Gemini API Key loaded automatically from Secrets!")

uploaded_file = st.file_uploader("Upload Scorecard Image (PNG/JPG)", type=["png", "jpg", "jpeg"])

if uploaded_file and api_key:
    try:
        genai.configure(api_key=api_key)
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Original Scorecard")
            st.image(image, use_container_width=True)
            
        if st.button("Extract & Reconstruct Scorecard"):
            with st.spinner("Analyzing scorecard and calculating match stats..."):
                model = genai.GenerativeModel('gemini-1.5-flash')
                
                prompt = """
                Extract this handwritten cricket scorecard into strict JSON format with keys:
                - 'match_details': {home_club, visitor_club, total_runs, wickets}
                - 'batting': list of {position, batter_name, runs, balls, how_out, bowler, fielder}
                - 'bowling': list of {position, bowler_name, overs, maidens, runs_conceded, wickets}
                Return ONLY raw JSON, no markdown formatting or extra text.
                """
                
                response = model.generate_content([prompt, image])
                
                raw_json = response.text.replace("```json", "").replace("```", "").strip()
                data = json.loads(raw_json)
                
                with col2:
                    st.subheader("Extracted Electronic Scorecard")
                    
                    st.write("**Match Summary:**", data.get("match_details", {}))
                    
                    df_bat = pd.DataFrame(data.get("batting", []))
                    df_bowl = pd.DataFrame(data.get("bowling", []))
                    
                    # Safe numeric conversions
                    if not df_bat.empty:
                        df_bat["runs"] = pd.to_numeric(df_bat["runs"], errors="coerce").fillna(0).astype(int)
                        df_bat["balls"] = pd.to_numeric(df_bat["balls"], errors="coerce").fillna(0).astype(int)
                        df_bat["strike_rate"] = df_bat.apply(
                            lambda row: round((row["runs"] / row["balls"] * 100), 2) if row["balls"] > 0 else 0.0, axis=1
                        )
                    
                    if not df_bowl.empty:
                        df_bowl["runs_conceded"] = pd.to_numeric(df_bowl["runs_conceded"], errors="coerce").fillna(0).astype(int)
                        df_bowl["wickets"] = pd.to_numeric(df_bowl["wickets"], errors="coerce").fillna(0).astype(int)
                    
                    st.markdown("### Batting Performance")
                    st.dataframe(df_bat, use_container_width=True)
                    
                    st.markdown("### Bowling Performance")
                    st.dataframe(df_bowl, use_container_width=True)
                    
                    st.download_button(
                        label="Download Clean JSON File",
                        data=json.dumps(data, indent=4),
                        file_name="scorecard_output.json",
                        mime="application/json"
                    )
    except Exception as e:
        st.error(f"Error executing API call: {e}")