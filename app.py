import st_express as st  # or import streamlit as st
import streamlit as st
import json
import pandas as pd
from google import genai
from google.genai import types
from PIL import Image

st.set_page_config(layout="wide", page_title="Cricket Scorecard OCR & Ball-by-Ball Engine")

st.title("🏏 Paper Scorecard Digitizer & Ball-by-Ball Reconstruction")
st.write("Upload a handwritten scorecard image to extract summaries and reconstruct the full ball-by-ball innings timeline.")

# --- API KEY SECRETS HANDLING ---
api_key = st.secrets.get("GEMINI_API_KEY", None)

if not api_key:
    raw_api_key = st.sidebar.text_input("Enter Gemini API Key", type="password")
    api_key = raw_api_key.strip().strip('"').strip("'") if raw_api_key else ""
else:
    st.sidebar.success("🔑 Gemini API Key loaded automatically!")

uploaded_file = st.file_uploader("Upload Scorecard Image (PNG/JPG)", type=["png", "jpg", "jpeg"])

if uploaded_file and api_key:
    try:
        # Initialize Google GenAI Client
        client = genai.Client(api_key=api_key)
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Original Scorecard")
            st.image(image, use_container_width=True)
            
        if st.button("Extract & Reconstruct Ball-by-Ball"):
            with st.spinner("AI is analyzing overs, strike rotations, and delivery boxes..."):
                
                prompt = """
                You are an expert cricket statistician and scorekeeper. Analyze this handwritten scorecard carefully.
                Cross-reference the Batting table, Bowling table, Extras, and the bottom Over-by-Over Matrix.
                
                Perform a full ball-by-ball reconstruction of the innings applying standard strike rotation rules (odd runs switch strike, end-of-over switches strike).
                
                Return ONLY raw strict JSON matching this schema:
                {
                    "match_details": {
                        "home_club": "Spartans",
                        "visitor_club": "Indus",
                        "ground": "Spartans Ground",
                        "total_runs": 235,
                        "wickets": 8
                    },
                    "batting": [
                        {"position": 1, "batter_name": "Ruthvik", "runs": 0, "balls": 2, "how_out": "Bowled", "bowler": "Deepak", "fielder": null},
                        {"position": 2, "batter_name": "Maulin", "runs": 7, "balls": 18, "how_out": "Caught", "bowler": "Kedar", "fielder": "Sidharth"},
                        {"position": 3, "batter_name": "Harsh", "runs": 27, "balls": 28, "how_out": "Caught", "bowler": "Kedar", "fielder": "Sunny"},
                        {"position": 4, "batter_name": "Shankara", "runs": 89, "balls": 92, "how_out": "Run Out", "bowler": null, "fielder": "Kedar"},
                        {"position": 5, "batter_name": "Arawind", "runs": 61, "balls": 71, "how_out": "Caught", "bowler": "Akshay", "fielder": "Chinmay"},
                        {"position": 6, "batter_name": "Shailesh", "runs": 12, "balls": 23, "how_out": "LBW", "bowler": "Kedar", "fielder": null},
                        {"position": 7, "batter_name": "Srikantan", "runs": 13, "balls": 14, "how_out": "Caught", "bowler": "Kedar", "fielder": "Vivek"},
                        {"position": 8, "batter_name": "Vishnu", "runs": 3, "balls": 5, "how_out": "LBW", "bowler": "Kedar", "fielder": null},
                        {"position": 9, "batter_name": "Nirav", "runs": 0, "balls": 3, "how_out": "Not Out", "bowler": null, "fielder": null},
                        {"position": 10, "batter_name": "Nikhil", "runs": 10, "balls": 11, "how_out": "Not Out", "bowler": null, "fielder": null},
                        {"position": 11, "batter_name": "Hasu", "runs": 0, "balls": 0, "how_out": "Did Not Bat", "bowler": null, "fielder": null}
                    ],
                    "bowling": [
                        {"position": 1, "bowler_name": "Deepak", "overs": 9.0, "maidens": 1, "runs_conceded": 39, "wickets": 1},
                        {"position": 2, "bowler_name": "Kedar", "overs": 9.0, "maidens": 1, "runs_conceded": 36, "wickets": 5},
                        {"position": 3, "bowler_name": "Akshay", "overs": 9.0, "maidens": 0, "runs_conceded": 49, "wickets": 1},
                        {"position": 4, "bowler_name": "Anand", "overs": 9.0, "maidens": 0, "runs_conceded": 44, "wickets": 0},
                        {"position": 5, "bowler_name": "Shiva", "overs": 3.0, "maidens": 0, "runs_conceded": 20, "wickets": 0},
                        {"position": 6, "bowler_name": "Shidarth", "overs": 4.0, "maidens": 0, "runs_conceded": 23, "wickets": 0},
                        {"position": 7, "bowler_name": "Chinmay", "overs": 2.0, "maidens": 0, "runs_conceded": 23, "wickets": 0}
                    ],
                    "ball_by_ball": [
                        {"over_num": 1, "ball_num": 1, "bowler": "Deepak", "striker": "Ruthvik", "non_striker": "Maulin", "runs": 0, "extras": 0, "event": "Dot", "score": 0, "wickets": 0},
                        {"over_num": 1, "ball_num": 2, "bowler": "Deepak", "striker": "Ruthvik", "non_striker": "Maulin", "runs": 0, "extras": 0, "event": "WICKET (Bowled)", "score": 0, "wickets": 1},
                        {"over_num": 1, "ball_num": 3, "bowler": "Deepak", "striker": "Maulin", "non_striker": "Harsh", "runs": 0, "extras": 1, "event": "Leg Bye", "score": 1, "wickets": 1},
                        {"over_num": 1, "ball_num": 4, "bowler": "Deepak", "striker": "Harsh", "non_striker": "Maulin", "runs": 0, "extras": 0, "event": "Dot", "score": 1, "wickets": 1},
                        {"over_num": 1, "ball_num": 5, "bowler": "Deepak", "striker": "Harsh", "non_striker": "Maulin", "runs": 0, "extras": 0, "event": "Dot", "score": 1, "wickets": 1},
                        {"over_num": 1, "ball_num": 6, "bowler": "Deepak", "striker": "Harsh", "non_striker": "Maulin", "runs": 0, "extras": 0, "event": "Dot", "score": 1, "wickets": 1}
                    ]
                }
                Reconstruct as many overs as recorded in the scorecard grid up to 43 overs.
                Do not include markdown code block syntax (like ```json).
                """
                
                response = client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=[prompt, image]
                )
                
                raw_json = response.text.replace("```json", "").replace("```", "").strip()
                data = json.loads(raw_json)
                
                with col2:
                    st.subheader("Extracted Electronic Scorecard")
                    st.write("**Match Details:**", data.get("match_details", {}))
                    
                    df_bat = pd.DataFrame(data.get("batting", []))
                    df_bowl = pd.DataFrame(data.get("bowling", []))
                    df_bbb = pd.DataFrame(data.get("ball_by_ball", []))
                    
                    # Safe Numeric Format
                    if not df_bat.empty:
                        df_bat["runs"] = pd.to_numeric(df_bat["runs"], errors="coerce").fillna(0).astype(int)
                        df_bat["balls"] = pd.to_numeric(df_bat["balls"], errors="coerce").fillna(0).astype(int)
                        df_bat["strike_rate"] = df_bat.apply(
                            lambda r: round((r["runs"] / r["balls"] * 100), 2) if r["balls"] > 0 else 0.0, axis=1
                        )
                    
                    st.markdown("### 🏏 Batting Summary")
                    st.dataframe(df_bat, use_container_width=True)
                    
                    st.markdown("### 🎯 Bowling Summary")
                    st.dataframe(df_bowl, use_container_width=True)
                    
                    st.markdown("### 📊 Ball-by-Ball Innings Reconstruction")
                    if not df_bbb.empty:
                        st.dataframe(df_bbb, use_container_width=True)
                        
                        # Add CSV Download for Ball-by-Ball
                        st.download_button(
                            label="Download Ball-by-Ball CSV",
                            data=df_bbb.to_csv(index=False),
                            file_name="ball_by_ball_reconstruction.csv",
                            mime="text/csv"
                        )
                    
                    st.download_button(
                        label="Download Full JSON",
                        data=json.dumps(data, indent=4),
                        file_name="full_scorecard.json",
                        mime="application/json"
                    )
    except Exception as e:
        st.error(f"Error executing API call: {e}")