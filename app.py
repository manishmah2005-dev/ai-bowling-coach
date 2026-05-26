import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

from pose_analysis import analyze_video
from rule_engine import get_strategies, generate_report
from ball_tracker import track_ball, analyze_ball_trajectory

# ── Page config ──────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Bowling Coach",
    page_icon="🏏",
    layout="wide"
)

st.markdown("""
<style>
    .score-card {
        background: #1e1e2e;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        margin: 4px;
    }
    .score-good { border: 2px solid #00ff88; color: #00ff88; }
    .score-weak { border: 2px solid #ff4444; color: #ff4444; }
    .strategy-card {
        background: #1e1e2e;
        border-left: 4px solid #f39c12;
        padding: 14px;
        border-radius: 8px;
        margin: 8px 0;
    }
    .combo-card { border-left-color: #e74c3c; }
    .high-card  { border-left-color: #e67e22; }
    .medium-card{ border-left-color: #3498db; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar navigation ───────────────────────────────────────────
st.sidebar.title("🏏 AI Bowling Coach")
page = st.sidebar.radio("Navigate", [
    "Analyze Video",
    "Progress Report",
    "Compare Players"
])

# ════════════════════════════════════════════════════════════════
# PAGE 1 — ANALYZE VIDEO
# ════════════════════════════════════════════════════════════════
if page == "Analyze Video":
    st.title("🏏 AI Bowling Coach — Video Analysis")

    col1, col2 = st.columns([2, 1])

    with col1:
        player_name = st.text_input("Player Name", placeholder="e.g. Virat Kohli")
        session_num = st.number_input("Session Number", min_value=1, value=1)

    with col2:
        batting_style = st.selectbox("Batting Style", ["Right Hand", "Left Hand"])
        batting_position = st.selectbox("Position", ["Opener", "Top Order", "Middle Order", "Lower Order"])

    uploaded_file = st.file_uploader("Upload Batting Video", type=['mp4', 'avi', 'mov'])

    if uploaded_file and st.button("Analyze", type="primary", use_container_width=True):

        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
            tfile.write(uploaded_file.read())
            video_path = tfile.name

        with st.spinner("Analyzing batting technique..."):
            weaknesses, frames, avg_scores, _ = analyze_video(video_path, session_num)

        os.unlink(video_path)

        if not avg_scores:
            st.error("Could not detect pose. Please upload a clear batting video.")
        else:
            player_display = player_name if player_name else "Player"
            st.success(f"Analysis complete for {player_display}!")

            # ── SCORES ──────────────────────────────────────────
            st.subheader("📊 Performance Scores")
            cols = st.columns(5)
            score_keys = ['balance', 'footwork', 'timing', 'bat_swing', 'overall']
            labels = ['Balance', 'Footwork', 'Timing', 'Bat Swing', 'Overall']

            for i, (key, label) in enumerate(zip(score_keys, labels)):
                val = avg_scores.get(key, 0)
                css = "score-good" if val >= 65 else "score-weak"
                with cols[i]:
                    st.markdown(f"""
                    <div class="score-card {css}">
                        <h2>{val}</h2>
                        <p>{label}</p>
                    </div>
                    """, unsafe_allow_html=True)

            # ── RADAR CHART ──────────────────────────────────────
            st.subheader("📈 Skill Radar")
            categories = ['Balance', 'Footwork', 'Timing', 'Bat Swing']
            values = [
                avg_scores.get('balance', 0),
                avg_scores.get('footwork', 0),
                avg_scores.get('timing', 0),
                avg_scores.get('bat_swing', 0)
            ]
            values_closed = values + [values[0]]
            categories_closed = categories + [categories[0]]

            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=values_closed,
                theta=categories_closed,
                fill='toself',
                name=player_display,
                line_color='#00ff88'
            ))
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                height=350
            )
            st.plotly_chart(fig, use_container_width=True)

            # ── WEAKNESSES ───────────────────────────────────────
            st.subheader("⚠️ Weaknesses Detected")
            if weaknesses:
                for w in weaknesses:
                    st.error(f"• {w}")
            else:
                st.success("No major weaknesses detected! Technically sound batsman.")

            # ── STRATEGIES ───────────────────────────────────────
            st.subheader("🎯 Recommended Bowling Strategies")
            strategies = get_strategies(avg_scores, weaknesses)

            for s in strategies:
                priority = s['priority']
                css_class = f"{priority.lower()}-card"
                badge_color = {
                    'COMBO': '#e74c3c',
                    'HIGH': '#e67e22',
                    'MEDIUM': '#3498db',
                    'STANDARD': '#2ecc71'
                }.get(priority, '#888')

                st.markdown(f"""
                <div class="strategy-card {css_class}">
                    <span style="background:{badge_color};padding:2px 8px;border-radius:4px;font-size:12px;font-weight:bold">{priority}</span>
                    <br><br>
                    <b>Line:</b> {s['line']} &nbsp;|&nbsp;
                    <b>Length:</b> {s['length']} &nbsp;|&nbsp;
                    <b>Speed:</b> {s['speed']}<br>
                    <b>Variation:</b> {s['variation']}<br>
                    <small style="color:#aaa">💡 {s['why']}</small>
                </div>
                """, unsafe_allow_html=True)

            # ── ANNOTATED FRAMES ─────────────────────────────────
            if frames:
                st.subheader("🎬 Annotated Frames")
                cols = st.columns(min(4, len(frames)))
                for i, frame in enumerate(frames[:4]):
                    with cols[i % 4]:
                        st.image(frame, caption=f"Frame {i+1}", use_container_width=True)

# ════════════════════════════════════════════════════════════════
# PAGE 2 — PROGRESS REPORT
# ════════════════════════════════════════════════════════════════
elif page == "Progress Report":
    st.title("📈 Progress Report")

    if not os.path.exists('progress_log.csv'):
        st.info("No sessions recorded yet. Analyze a video first!")
    else:
        df = pd.read_csv('progress_log.csv')
        st.dataframe(df, use_container_width=True)

        # Line chart — scores over sessions
        st.subheader("Score Trends Over Sessions")
        score_cols = [c for c in ['Balance', 'Footwork', 'Timing', 'BatSwing', 'Overall'] if c in df.columns]

        if score_cols:
            fig = px.line(df, x='Session', y=score_cols,
                         title="Performance Scores Over Time",
                         markers=True)
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white')
            )
            st.plotly_chart(fig, use_container_width=True)

        # Latest session summary
        if len(df) > 0:
            latest = df.iloc[-1]
            st.subheader("Latest Session Summary")
            cols = st.columns(4)
            for i, col in enumerate(['Balance', 'Footwork', 'Timing', 'BatSwing']):
                if col in latest:
                    val = int(latest[col])
                    css = "score-good" if val >= 65 else "score-weak"
                    with cols[i]:
                        st.markdown(f"""
                        <div class="score-card {css}">
                            <h2>{val}</h2>
                            <p>{col}</p>
                        </div>
                        """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
# PAGE 3 — COMPARE PLAYERS
# ════════════════════════════════════════════════════════════════
elif page == "Compare Players":
    st.title("⚖️ Compare Players")
    st.info("Upload two batting videos to compare their technique side by side.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Player 1")
        name1 = st.text_input("Name", key="n1", placeholder="e.g. Player A")
        video1 = st.file_uploader("Video", type=['mp4', 'avi', 'mov'], key="v1")

    with col2:
        st.subheader("Player 2")
        name2 = st.text_input("Name", key="n2", placeholder="e.g. Player B")
        video2 = st.file_uploader("Video", type=['mp4', 'avi', 'mov'], key="v2")

    if video1 and video2 and st.button("Compare", type="primary", use_container_width=True):

        scores_list = []
        names = [name1 or "Player 1", name2 or "Player 2"]

        for i, (video, name) in enumerate(zip([video1, video2], names)):
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
                tfile.write(video.read())
                vpath = tfile.name

            with st.spinner(f"Analyzing {name}..."):
                _, _, avg_scores, _ = analyze_video(vpath, i+1)
            os.unlink(vpath)
            scores_list.append(avg_scores)

        if all(scores_list):
            # Side by side scores
            st.subheader("Score Comparison")
            categories = ['Balance', 'Footwork', 'Timing', 'Bat Swing']
            keys = ['balance', 'footwork', 'timing', 'bat_swing']

            fig = go.Figure()
            colors = ['#00ff88', '#ff6b6b']

            for i, (scores, name) in enumerate(zip(scores_list, names)):
                vals = [scores.get(k, 0) for k in keys]
                vals_closed = vals + [vals[0]]
                fig.add_trace(go.Scatterpolar(
                    r=vals_closed,
                    theta=categories + [categories[0]],
                    fill='toself',
                    name=name,
                    line_color=colors[i],
                    opacity=0.7
                ))

            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)

            # Bar chart comparison
            comp_data = []
            for key, label in zip(keys, categories):
                comp_data.append({
                    'Skill': label,
                    names[0]: scores_list[0].get(key, 0),
                    names[1]: scores_list[1].get(key, 0)
                })

            comp_df = pd.DataFrame(comp_data)
            fig2 = px.bar(comp_df, x='Skill', y=[names[0], names[1]],
                         barmode='group', title="Score Breakdown")
            fig2.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white')
            )
            st.plotly_chart(fig2, use_container_width=True)

            # Individual strategies
            col1, col2 = st.columns(2)
            for i, (scores, name) in enumerate(zip(scores_list, names)):
                weaknesses = []
                if scores.get('balance', 100) < 65:
                    weaknesses.append('Balance issue')
                if scores.get('footwork', 100) < 65:
                    weaknesses.append('Footwork issue')
                if scores.get('timing', 100) < 65:
                    weaknesses.append('Timing issue')
                if scores.get('bat_swing', 100) < 65:
                    weaknesses.append('Bat swing issue')

                strategies = get_strategies(scores, weaknesses)
                target_col = col1 if i == 0 else col2

                with target_col:
                    st.subheader(f"🎯 {name} — Top Strategy")
                    if strategies:
                        s = strategies[0]
                        st.markdown(f"""
                        **Line:** {s['line']}  
                        **Length:** {s['length']}  
                        **Speed:** {s['speed']}  
                        **Variation:** {s['variation']}  
                        💡 _{s['why']}_
                        """)