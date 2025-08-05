import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from backend import run_analysis

st.title("Cursor IDE Feedback Analyzer")

reddit_client_id = st.text_input("Reddit Client ID")
reddit_client_secret = st.text_input("Reddit Client Secret", type="password")
user_agent = st.text_input("User Agent", "cursor_feedback_app/0.1 by your_reddit_username")

if st.button("Run Analysis"):
    if reddit_client_id and reddit_client_secret:
        with st.spinner("Running analysis..."):
            df, reddit_summary, so_summary, wordcloud_buf = run_analysis(
                reddit_client_id, reddit_client_secret, user_agent
            )

        st.success("Analysis Complete!")

        st.subheader("Sentiment Distribution")
        fig1, ax1 = plt.subplots()
        df["sentiment"].value_counts().plot(kind="bar", ax=ax1)
        st.pyplot(fig1)

        st.subheader("Topic Distribution")
        fig2, ax2 = plt.subplots()
        df["topic"].value_counts().plot(kind="bar", ax=ax2)
        st.pyplot(fig2)

        st.subheader("Word Cloud")
        st.image(wordcloud_buf)

        st.subheader("Reddit Summary")
        st.write(reddit_summary)

        st.subheader("Stack Overflow Summary")
        st.write(so_summary)

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", csv, "cursor_feedback_final.csv", "text/csv")
    else:
        st.error("Please provide Reddit API credentials.")
