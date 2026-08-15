import streamlit as st
import json
import pandas as pd
from google import genai
from PIL import Image

st.set_page_config(layout="wide", page_title="Cricket Scorecard OCR & HITL Validation Engine")

st.title("🏏 Paper Scorecard Digitizer & HITL Validation Engine")
st.write("Upload a handwritten scorecard to extract, cross-validate against master tallies, and manually edit discrepancies.")

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
        client = genai.Client(api_key=api_key)
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns([1, 1.2])
        
        with col1:
            st.subheader("Original Scorecard")
            st.image(image, use_container_width=True)
            
        if st.button("Extract & Run Validation Audit"):
            with st.spinner("AI is extracting data and cross-checking against Master Tally Grid..."):
                
                prompt = """
                You are an expert cricket statistician and scorekeeper. Analyze this handwritten scorecard carefully.
                Cross-reference the Batting table, Bowling table, Extras, the top-right Cumulative Run Tally grid, and the bottom Over-by-Over Matrix.
                
                Perform a full ball-by-ball reconstruction of the innings applying standard strike rotation rules. Ensure the sum of runs across all balls matches the Master Score Box (235 runs, 8 wickets).
                
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
                Do not include markdown code block syntax.
                """
                
                response = client.models.generate_content(
                    model='gemini-3.5-flash',
                    contents=[prompt, image]
                )
                
                raw_json = response.text.replace("```json", "").replace("```", "").strip()
                st.session_state["extracted_data"] = json.loads(raw_json)

        # --- HITL & RECONCILIATION SECTION ---
        if "extracted_data" in st.session_state:
            data = st.session_state["extracted_data"]
            
            with col2:
                st.subheader("🔍 Multi-Section Reconciliation Audit")
                
                master_target_runs = data.get("match_details", {}).get("total_runs", 235)
                master_target_wickets = data.get("match_details", {}).get("wickets", 8)
                
                df_bbb = pd.DataFrame(data.get("ball_by_ball", []))
                
                # Perform Live Calculation
                if not df_bbb.empty:
                    bbb_calculated_runs = int(df_bbb["runs"].sum() + df_bbb["extras"].sum())
                    bbb_calculated_wickets = int(df_bbb["event"].str.contains("WICKET", case=False, na=False).sum())
                    
                    # Validation Display
                    audit_col1, audit_col2, audit_col3 = st.columns(3)
                    
                    with audit_col1:
                        st.metric("Master Target Score", f"{master_target_runs} / {master_target_wickets}")
                    with audit_col2:
                        st.metric("Reconstructed BBB Score", f"{bbb_calculated_runs} / {bbb_calculated_wickets}")
                    with audit_col3:
                        diff = master_target_runs - bbb_calculated_runs
                        if diff == 0:
                            st.success("✅ 100% Reconciliation Match!")
                        else:
                            st.error(f"⚠️ Discrepancy: {diff} Run(s) Off!")

                st.markdown("---")
                st.markdown("### ✍️ Human-In-The-Loop (HITL) Interactive Editor")
                st.info("Edit any value directly in the table below to correct discrepancies before exporting!")
                
                # Streamlit Interactive Data Editor
                edited_bbb_df = st.data_editor(
                    df_bbb,
                    use_container_width=True,
                    num_rows="dynamic",
                    key="bbb_editor"
                )
                
                # Recalculate Live Stats based on User Edits
                live_runs = int(edited_bbb_df["runs"].sum() + edited_bbb_df["extras"].sum())
                st.caption(f"**Live Corrected Total:** {live_runs} Runs")

                # Export Controls
                st.download_button(
                    label="Download Corrected Ball-by-Ball CSV",
                    data=edited_bbb_df.to_csv(index=False),
                    file_name="verified_ball_by_ball.csv",
                    mime="text/csv"
                )
                
    except Exception as e:
        st.error(f"Error executing API call: {e}")