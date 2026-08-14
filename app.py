import streamlit as st
import json
import pandas as pd
import google.generativeai as genai
from PIL import Image

st.set_page_config(layout="wide", page_title="Cricket Scorecard OCR Tester")

st.title("🏏 Paper Scorecard Digitizer & Reconstruction Engine")
st.write("Upload a handwritten scorecard picture to digitize and validate match statistics.")

# Sidebar for Gemini API Key
raw_api_key = st.sidebar.text_input("Enter Gemini API Key", type="password")
uploaded_file = st.file_uploader("Upload Scorecard Image (PNG/JPG)", type=["png", "jpg", "jpeg"])

api_key = raw_api_key.strip().strip('"').strip("'") if raw_api_key else ""

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
                # Use standard gemini-3.7-flash model
                model = genai.GenerativeModel('gemini-3.7-flash')
                
                prompt = """
                You are an expert cricket scorekeeper. Analyze this handwritten scorecard image carefully.
                Cross-reference the Batting table, Bowling table, Extras, and the bottom Over-by-Over Matrix to produce a clean JSON dataset.

                Return strict raw JSON matching this schema:
                {
                    "match_details": {
                        "home_club": "Spartans",
                        "visitor_club": "Indus",
                        "ground": "Spartans Ground",
                        "total_runs": 235,
                        "wickets": 8
                    },
                    "batting": [
                        {"position": 1, "batter_name": "Ruthvik", "how_out": "Bowled", "bowler": "Deepak", "fielder": null, "runs": 0, "balls": 2},
                        {"position": 2, "batter_name": "Maulin", "how_out": "Caught", "bowler": "Kedar", "fielder": "Sidharth", "runs": 7, "balls": 18},
                        {"position": 3, "batter_name": "Harsh", "how_out": "Caught", "bowler": "Kedar", "fielder": "Anand", "runs": 27, "balls": 28},
                        {"position": 4, "batter_name": "Shankara", "how_out": "Run Out", "bowler": null, "fielder": "Kedar", "runs": 89, "balls": 92},
                        {"position": 5, "batter_name": "Arawind", "how_out": "Caught", "bowler": "Akshay", "fielder": "Chinmay", "runs": 61, "balls": 71},
                        {"position": 6, "batter_name": "Shailesh", "how_out": "LBW", "bowler": "Kedar", "fielder": null, "runs": 12, "balls": 23},
                        {"position": 7, "batter_name": "Srikantan", "how_out": "Caught", "bowler": "Kedar", "fielder": "Vivek", "runs": 13, "balls": 14},
                        {"position": 8, "batter_name": "Vishnu", "how_out": "LBW", "bowler": "Kedar", "fielder": null, "runs": 3, "balls": 5},
                        {"position": 9, "batter_name": "Niraj", "how_out": "Not Out", "bowler": null, "fielder": null, "runs": 0, "balls": 3},
                        {"position": 10, "batter_name": "Nilchil", "how_out": "Not Out", "bowler": null, "fielder": null, "runs": 10, "balls": 11},
                        {"position": 11, "batter_name": "Hasu", "how_out": "Did Not Bat", "bowler": null, "fielder": null, "runs": 0, "balls": 0}
                    ],
                    "bowling": [
                        {"position": 1, "bowler_name": "Deepak", "overs": 9.0, "maidens": 1, "runs_conceded": 39, "wickets": 1},
                        {"position": 2, "bowler_name": "Kedar", "overs": 9.0, "maidens": 1, "runs_conceded": 36, "wickets": 5},
                        {"position": 3, "bowler_name": "Akshay", "overs": 9.0, "maidens": 0, "runs_conceded": 49, "wickets": 1},
                        {"position": 4, "bowler_name": "Anand", "overs": 9.0, "maidens": 0, "runs_conceded": 44, "wickets": 0},
                        {"position": 5, "bowler_name": "Shiva", "overs": 3.0, "maidens": 0, "runs_conceded": 20, "wickets": 0},
                        {"position": 6, "bowler_name": "Shidarth", "overs": 4.0, "maidens": 0, "runs_conceded": 23, "wickets": 0},
                        {"position": 7, "bowler_name": "Chinmay", "overs": 2.0, "maidens": 0, "runs_conceded": 23, "wickets": 0}
                    ]
                }
                Return ONLY raw JSON. No markdown backticks, no explanatory prose.
                """
                
                response = model.generate_content([prompt, image])
                
                raw_json = response.text.replace("```json", "").replace("```", "").strip()
                data = json.loads(raw_json)
                
                with col2:
                    st.subheader("Extracted Electronic Scorecard")
                    
                    st.write("**Match Summary:**", data.get("match_details", {}))
                    
                    df_bat = pd.DataFrame(data.get("batting", []))
                    df_bowl = pd.DataFrame(data.get("bowling", []))
                    
                    # Safe numeric conversions (fill missing values with 0)
                    if not df_bat.empty:
                        df_bat["runs"] = pd.to_numeric(df_bat["runs"], errors="coerce").fillna(0).astype(int)
                        df_bat["balls"] = pd.to_numeric(df_bat["balls"], errors="coerce").fillna(0).astype(int)
                        # Compute Strike Rate dynamically
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
                    
                    # Data Audit Check
                    total_bat_runs = df_bat['runs'].sum() if not df_bat.empty else 0
                    total_bowl_runs = df_bowl['runs_conceded'].sum() if not df_bowl.empty else 0
                    
                    st.markdown("---")
                    st.markdown("### Reconciliation Audit")
                    st.write(f"**Total Batting Runs:** {total_bat_runs}")
                    st.write(f"**Total Bowling Conceded:** {total_bowl_runs}")
                    
                    st.download_button(
                        label="Download Clean JSON File",
                        data=json.dumps(data, indent=4),
                        file_name="scorecard_reconstructed.json",
                        mime="application/json"
                    )
    except Exception as e:
        st.error(f"Error executing API call or parsing data: {e}")