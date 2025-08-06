import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import praw
import requests
import time
import re
import os
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from transformers import pipeline
import warnings
warnings.filterwarnings("ignore")

# Page configuration
st.set_page_config(
    page_title="Feedback Summarizer & Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2e86c1;
        border-bottom: 2px solid #2e86c1;
        padding-bottom: 0.5rem;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_collected' not in st.session_state:
    st.session_state.data_collected = False
if 'df_all' not in st.session_state:
    st.session_state.df_all = None

def clean_text(text):
    """Clean and preprocess text data"""
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text

@st.cache_data
def collect_reddit_data(client_id, client_secret, user_agent, subreddits, time_ranges, target_posts=3000):
    """Collect data from Reddit with enhanced collection strategy"""
    try:
        reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent
        )
        
        reddit_posts = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Enhanced collection strategy
        collection_methods = ['top', 'hot', 'new', 'rising']
        total_operations = len(subreddits) * len(time_ranges) * len(collection_methods)
        current_operation = 0
        posts_per_operation = max(100, target_posts // total_operations)
        
        for subreddit_name in subreddits:
            try:
                subreddit = reddit.subreddit(subreddit_name)
                
                for time_filter in time_ranges:
                    for method in collection_methods:
                        status_text.text(f"Collecting from r/{subreddit_name} ({method} - {time_filter})...")
                        
                        try:
                            if method == 'top':
                                posts = subreddit.top(time_filter=time_filter, limit=posts_per_operation)
                            elif method == 'hot':
                                posts = subreddit.hot(limit=posts_per_operation)
                            elif method == 'new':
                                posts = subreddit.new(limit=posts_per_operation)
                            elif method == 'rising':
                                posts = subreddit.rising(limit=posts_per_operation)
                            
                            for post in posts:
                                reddit_posts.append({
                                    "source": "Reddit",
                                    "subreddit": subreddit_name,
                                    "title": post.title,
                                    "text": post.selftext,
                                    "score": post.score,
                                    "link": f"https://www.reddit.com{post.permalink}",
                                    "method": method
                                })
                                
                                # Stop if we've reached target
                                if len(reddit_posts) >= target_posts:
                                    progress_bar.empty()
                                    status_text.empty()
                                    return reddit_posts
                                    
                        except Exception as e:
                            st.warning(f"Error collecting from r/{subreddit_name} ({method}-{time_filter}): {str(e)}")
                        
                        current_operation += 1
                        progress_bar.progress(min(current_operation / total_operations, 0.99))
                        time.sleep(0.05)  # Reduced delay for faster collection
                        
            except Exception as e:
                st.warning(f"Error accessing subreddit r/{subreddit_name}: {str(e)}")
        
        progress_bar.empty()
        status_text.empty()
        return reddit_posts
    
    except Exception as e:
        st.error(f"Reddit API Error: {str(e)}")
        return []

@st.cache_data
def collect_stackoverflow_data(tags, target_posts=2000):
    """Collect data from Stack Overflow - simplified and reliable approach"""
    stack_posts = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Simplified search strategies that are more reliable
    search_configs = [
        {"endpoint": "search", "sort": "activity", "order": "desc"},
        {"endpoint": "search", "sort": "votes", "order": "desc"},
        {"endpoint": "search", "sort": "creation", "order": "desc"},
        {"endpoint": "questions", "sort": "activity", "order": "desc"},
        {"endpoint": "questions", "sort": "votes", "order": "desc"}
    ]
    
    posts_per_config = target_posts // len(search_configs)
    max_pages = max(5, posts_per_config // 100)
    
    for config_idx, config in enumerate(search_configs):
        endpoint = config["endpoint"]
        sort_method = config["sort"]
        order = config["order"]
        
        status_text.text(f"Collecting Stack Overflow data ({sort_method})...")
        
        # Base URL for Stack Exchange API
        base_url = f"https://api.stackexchange.com/2.3/{endpoint}"
        
        for page in range(1, max_pages + 1):
            try:
                # Build parameters
                params = {
                    "order": order,
                    "sort": sort_method,
                    "site": "stackoverflow",
                    "pagesize": 100,
                    "page": page,
                    "tagged": tags
                }
                
                # Make API request
                response = requests.get(base_url, params=params)
                
                # Debug: Show the actual URL being called
                if page == 1:
                    st.write(f"Debug: Calling {response.url}")
                
                if response.status_code != 200:
                    st.warning(f"Stack Overflow API returned status {response.status_code}")
                    if response.status_code == 429:  # Rate limit
                        time.sleep(5)
                        continue
                    break
                
                try:
                    data = response.json()
                except:
                    st.error(f"Failed to parse JSON response: {response.text[:200]}")
                    break
                
                # Check for API errors
                if "error_message" in data:
                    st.error(f"Stack Overflow API Error: {data['error_message']}")
                    break
                
                # Get items from response
                items = data.get("items", [])
                if not items:
                    st.info(f"No more items found for {sort_method} on page {page}")
                    break
                
                # Process each item
                for item in items:
                    stack_posts.append({
                        "source": "Stack Overflow",
                        "title": item.get("title", ""),
                        "text": "",  # Stack Overflow search API doesn't return body by default
                        "score": item.get("score", 0),
                        "comments": "",
                        "link": item.get("link", ""),
                        "tags": ", ".join(item.get("tags", [])),
                        "method": sort_method,
                        "view_count": item.get("view_count", 0),
                        "answer_count": item.get("answer_count", 0)
                    })
                    
                    # Stop if we've reached target
                    if len(stack_posts) >= target_posts:
                        progress_bar.empty()
                        status_text.empty()
                        st.success(f"Stack Overflow: Collected {len(stack_posts)} posts (target reached)")
                        return stack_posts
                
                # Check if there are more pages
                if not data.get("has_more", False):
                    break
                    
                # Rate limiting
                time.sleep(0.1)
                
            except requests.exceptions.RequestException as e:
                st.error(f"Network error collecting Stack Overflow data: {str(e)}")
                time.sleep(2)
                continue
            except Exception as e:
                st.error(f"Unexpected error collecting Stack Overflow data: {str(e)}")
                continue
        
        # Update progress
        progress_bar.progress((config_idx + 1) / len(search_configs))
        
        # Show current count
        st.info(f"Stack Overflow: Collected {len(stack_posts)} posts so far...")
        
        # Brief pause between different search methods
        time.sleep(0.5)
    
    progress_bar.empty()
    status_text.empty()
    
    if stack_posts:
        st.success(f"Stack Overflow: Successfully collected {len(stack_posts)} posts")
    else:
        st.warning("Stack Overflow: No posts were collected. This might be due to API issues or invalid tags.")
    
    return stack_posts

@st.cache_data
def perform_sentiment_analysis(texts):
    """Perform sentiment analysis on texts"""
    try:
        sentiment_classifier = pipeline("sentiment-analysis", 
                                      model="distilbert-base-uncased-finetuned-sst-2-english")
        
        sentiments = []
        progress_bar = st.progress(0)
        
        for i, text in enumerate(texts):
            try:
                result = sentiment_classifier(text[:512])
                sentiments.append(result[0]['label'])
            except:
                sentiments.append("NEUTRAL")
            
            if i % 10 == 0:
                progress_bar.progress((i + 1) / len(texts))
        
        progress_bar.empty()
        return sentiments
    
    except Exception as e:
        st.error(f"Sentiment analysis error: {str(e)}")
        return ["NEUTRAL"] * len(texts)

@st.cache_data
def perform_topic_classification(texts):
    """Perform topic classification on texts"""
    try:
        topic_classifier = pipeline("zero-shot-classification", 
                                   model="facebook/bart-large-mnli")
        candidate_labels = ["bug report", "feature request", "praise", 
                          "performance issue", "usability feedback"]
        
        topics = []
        progress_bar = st.progress(0)
        
        for i, text in enumerate(texts):
            try:
                result = topic_classifier(text[:512], candidate_labels)
                topics.append(result['labels'][0])
            except:
                topics.append("general")
            
            if i % 10 == 0:
                progress_bar.progress((i + 1) / len(texts))
        
        progress_bar.empty()
        return topics
    
    except Exception as e:
        st.error(f"Topic classification error: {str(e)}")
        return ["general"] * len(texts)

@st.cache_data
def generate_summaries(df_all):
    """Generate summaries for different sources"""
    try:
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        
        summaries = {}
        
        # Reddit Summary
        reddit_data = df_all[df_all["source"] == "Reddit"]
        if not reddit_data.empty:
            reddit_text = " ".join(reddit_data["content_clean"].tolist())[:3000]
            if reddit_text.strip():
                reddit_summary = summarizer(reddit_text, max_length=130, min_length=50, do_sample=False)
                summaries["Reddit"] = reddit_summary[0]['summary_text']
        
        # Stack Overflow Summary
        so_data = df_all[df_all["source"] == "Stack Overflow"]
        if not so_data.empty:
            so_text = " ".join(so_data["content_clean"].tolist())[:3000]
            if so_text.strip():
                so_summary = summarizer(so_text, max_length=130, min_length=50, do_sample=False)
                summaries["Stack Overflow"] = so_summary[0]['summary_text']
        
        return summaries
    
    except Exception as e:
        st.error(f"Summarization error: {str(e)}")
        return {}

def perform_advanced_categorization(df):
    """Perform advanced NLP categorization"""
    feedback_texts = df["content_clean"].dropna().astype(str).tolist()
    
    # Seed training data
    train_texts = [
        "App crashes when I click on settings", "There is a bug in the login page", "Frequent error when uploading file",
        "Can you add dark mode feature", "Please support multiple languages", "Need integration with Slack",
        "The core search functionality is slow", "Essential features missing from dashboard", "Critical performance issue",
        "This is high priority for our team", "It is urgent to fix these bugs", "Must have an option to export data",
        "Great app overall", "Nice work, keep it up", "I really like the product",
        "5 stars for this app", "Excellent performance", "Very bad experience, 1 star",
        "Smooth experience so far", "Frustrating to use sometimes", "Easy to use and fast",
        "Better than ChatGPT", "Not as good as Copilot", "OpenAI does it better"
    ]
    
    train_labels = [
        "bug_report","bug_report","bug_report",
        "feature_request","feature_request","feature_request",
        "core_functionality","core_functionality","core_functionality",
        "user_priority","user_priority","user_priority",
        "general_feedback","general_feedback","general_feedback",
        "rating","rating","rating",
        "experience_feedback","experience_feedback","experience_feedback",
        "competitor_comparison","competitor_comparison","competitor_comparison"
    ]
    
    # Train classifier
    vectorizer = TfidfVectorizer(stop_words="english")
    X_train = vectorizer.fit_transform(train_texts)
    clf = LogisticRegression(max_iter=300)
    clf.fit(X_train, train_labels)
    
    # Predict categories
    X_feedback = vectorizer.transform(feedback_texts)
    predicted_labels = clf.predict(X_feedback)
    pattern_counts = dict(Counter(predicted_labels))
    
    return pattern_counts, predicted_labels

def generate_recommendations(pattern_counts):
    """Generate recommendations based on feedback patterns"""
    recommendations = []
    if pattern_counts.get('bug_report', 0) > 5:
        recommendations.append("🐛 Investigate and resolve frequently occurring bugs to improve stability.")
    if pattern_counts.get('feature_request', 0) > 3:
        recommendations.append("✨ Prioritize most requested features to improve user satisfaction.")
    if pattern_counts.get('core_functionality', 0) > 2:
        recommendations.append("⚙️ Ensure core functionalities are robust and well-documented.")
    if pattern_counts.get('user_priority', 0) > 2:
        recommendations.append("🎯 Align roadmap with top user priorities.")
    if pattern_counts.get('rating', 0) > 2:
        recommendations.append("⭐ Analyze ratings feedback to improve overall sentiment.")
    if pattern_counts.get('experience_feedback', 0) > 2:
        recommendations.append("🎨 Improve user experience flows based on feedback.")
    if pattern_counts.get('competitor_comparison', 0) > 1:
        recommendations.append("🏆 Benchmark against competitors to address perceived gaps.")
    if not recommendations:
        recommendations.append("📊 Feedback does not indicate strong patterns; continue monitoring trends.")
    return recommendations

# Main App
def main():
    st.markdown('<h1 class="main-header">📊 Feedback Summarizer & Analyzer</h1>', unsafe_allow_html=True)
    
    # Sidebar configuration
    st.sidebar.title("Configuration")
    
    # Data Collection Section
    st.markdown('<div class="section-header">🔍 Data Collection</div>', unsafe_allow_html=True)
    
    with st.expander("Reddit API Configuration", expanded=not st.session_state.data_collected):
        col1, col2 = st.columns(2)
        
        # Try to get credentials from secrets, fallback to user input
        default_client_id = ""
        default_client_secret = ""
        default_user_agent = "feedback_analyzer_app"
        
        try:
            default_client_id = st.secrets["reddit"]["client_id"]
            default_client_secret = st.secrets["reddit"]["client_secret"]
            default_user_agent = st.secrets["reddit"]["user_agent"]
            st.info("✅ Using Reddit credentials from secrets")
        except:
            st.warning("⚠️ No Reddit credentials found in secrets. Please enter manually or configure secrets for deployment.")
        
        with col1:
            reddit_client_id = st.text_input("Reddit Client ID", value=default_client_id)
            reddit_client_secret = st.text_input("Reddit Client Secret", type="password", 
                                                value=default_client_secret)
        with col2:
            user_agent = st.text_input("User Agent", value=default_user_agent)
            subreddits = st.multiselect("Subreddits", 
                                       options=["vscode", "VisualStudio", "programming", "webdev"],
                                       default=["vscode"])
    
    with st.expander("Stack Overflow Configuration", expanded=not st.session_state.data_collected):
        so_tags = st.text_input("Stack Overflow Tags", value="visual-studio-code")
        
        # Add a test button for Stack Overflow API
        if st.button("🧪 Test Stack Overflow API"):
            with st.spinner("Testing Stack Overflow API..."):
                test_url = f"https://api.stackexchange.com/2.3/search?order=desc&sort=activity&tagged={so_tags}&site=stackoverflow&pagesize=5"
                try:
                    response = requests.get(test_url)
                    st.write(f"**Test URL:** {test_url}")
                    st.write(f"**Status Code:** {response.status_code}")
                    if response.status_code == 200:
                        data = response.json()
                        items = data.get("items", [])
                        st.write(f"**Items Found:** {len(items)}")
                        if items:
                            st.write("**Sample Item:**")
                            st.json(items[0])
                        else:
                            st.warning("No items returned. Try different tags.")
                    else:
                        st.error(f"API Error: {response.text}")
                except Exception as e:
                    st.error(f"Test failed: {str(e)}")
        
    with st.expander("Collection Settings", expanded=not st.session_state.data_collected):
        col1, col2, col3 = st.columns(3)
        with col1:
            reddit_target = st.number_input("Reddit Posts Target", min_value=500, max_value=10000, value=3000, step=500)
        with col2:
            stackoverflow_target = st.number_input("Stack Overflow Posts Target", min_value=500, max_value=10000, value=2000, step=500)
        with col3:
            total_target = reddit_target + stackoverflow_target
            st.metric("Total Target", f"{total_target:,}")
            
        # Add more subreddit options
        additional_subreddits = st.multiselect("Additional Subreddits (for more data)", 
                                             options=["VisualStudio", "programming", "webdev", "learnprogramming", 
                                                    "javascript", "Python", "reactjs", "node", "webdevelopment"],
                                             help="Select more subreddits to reach your target posts")
    
    # Data collection button
    if st.button("🚀 Collect Data", type="primary"):
        if reddit_client_id and reddit_client_secret and subreddits:
            with st.spinner("Collecting data..."):
                # Combine all selected subreddits
                all_subreddits = list(set(subreddits + additional_subreddits))
                
                # Show collection plan
                st.info(f"📊 Collection Plan: {len(all_subreddits)} subreddits, targeting {reddit_target:,} Reddit posts and {stackoverflow_target:,} Stack Overflow posts")
                
                # Collect Reddit data
                time_ranges = ["day", "week", "month", "year"]  # Added 'year' for more data
                reddit_posts = collect_reddit_data(reddit_client_id, reddit_client_secret, 
                                                 user_agent, all_subreddits, time_ranges, reddit_target)
                
                # Collect Stack Overflow data
                stack_posts = collect_stackoverflow_data(so_tags, stackoverflow_target)
                
                # Combine and process data
                all_posts = reddit_posts + stack_posts
                df_all = pd.DataFrame(all_posts)
                
                # Ensure required columns exist
                for col in ["title", "text", "comments"]:
                    if col not in df_all.columns:
                        df_all[col] = ""
                
                # Convert to string and handle NaN
                df_all[["title", "text", "comments"]] = (
                    df_all[["title", "text", "comments"]]
                    .fillna("")
                    .astype(str)
                )
                
                # Combine content fields
                df_all["content"] = df_all["title"] + " " + df_all["text"] + " " + df_all["comments"]
                df_all["content_clean"] = df_all["content"].apply(clean_text)
                
                # Remove duplicates
                df_all = df_all.drop_duplicates(subset=["link"]) if "link" in df_all.columns else df_all.drop_duplicates()
                
                st.session_state.df_all = df_all
                st.session_state.data_collected = True
                
                # Show detailed collection results
                reddit_count = len([p for p in all_posts if p['source'] == 'Reddit'])
                so_count = len([p for p in all_posts if p['source'] == 'Stack Overflow'])
                
                st.success(f"✅ Data collection completed!")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Reddit Posts", f"{reddit_count:,}", f"{reddit_count - reddit_target:+,}")
                with col2:
                    st.metric("Stack Overflow Posts", f"{so_count:,}", f"{so_count - stackoverflow_target:+,}")
                with col3:
                    total_collected = reddit_count + so_count
                    st.metric("Total Collected", f"{total_collected:,}", f"{total_collected - total_target:+,}")
                
                if total_collected >= 5000:
                    st.balloons()
                    st.success("🎉 Target of 5000+ posts achieved!")
                elif total_collected >= total_target:
                    st.success("🎯 Collection target met!")
        else:
            st.error("Please provide Reddit API credentials and select subreddits.")
    
    # Analysis Section
    if st.session_state.data_collected and st.session_state.df_all is not None:
        df_all = st.session_state.df_all
        
        st.markdown('<div class="section-header">📊 Data Overview</div>', unsafe_allow_html=True)
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Posts", len(df_all))
        with col2:
            reddit_count = len(df_all[df_all["source"] == "Reddit"])
            st.metric("Reddit Posts", reddit_count)
        with col3:
            so_count = len(df_all[df_all["source"] == "Stack Overflow"])
            st.metric("Stack Overflow Posts", so_count)
        with col4:
            avg_score = df_all["score"].mean() if "score" in df_all.columns else 0
            st.metric("Avg Score", f"{avg_score:.1f}")
        
        # Display sample data
        st.subheader("Sample Data")
        st.dataframe(df_all.head(10))
        
        # NLP Analysis Section
        st.markdown('<div class="section-header">🤖 NLP Analysis</div>', unsafe_allow_html=True)
        
        analysis_options = st.multiselect(
            "Select Analysis Types",
            ["Sentiment Analysis", "Topic Classification", "Text Summarization", "Advanced Categorization"],
            default=["Sentiment Analysis"]
        )
        
        if st.button("🔬 Run Analysis", type="primary"):
            analysis_results = {}
            
            if "Sentiment Analysis" in analysis_options:
                with st.spinner("Performing sentiment analysis..."):
                    sentiments = perform_sentiment_analysis(df_all["content_clean"].tolist())
                    df_all["sentiment"] = sentiments
                    analysis_results["sentiment"] = True
            
            if "Topic Classification" in analysis_options:
                with st.spinner("Performing topic classification..."):
                    topics = perform_topic_classification(df_all["content_clean"].tolist())
                    df_all["topic"] = topics
                    analysis_results["topic"] = True
            
            if "Text Summarization" in analysis_options:
                with st.spinner("Generating summaries..."):
                    summaries = generate_summaries(df_all)
                    analysis_results["summaries"] = summaries
            
            if "Advanced Categorization" in analysis_options:
                with st.spinner("Performing advanced categorization..."):
                    pattern_counts, predicted_labels = perform_advanced_categorization(df_all)
                    df_all["category"] = predicted_labels
                    analysis_results["patterns"] = pattern_counts
            
            st.session_state.df_all = df_all
            st.success("✅ Analysis completed!")
        
        # Results Section
        if "sentiment" in df_all.columns or "topic" in df_all.columns:
            st.markdown('<div class="section-header">📈 Results & Visualizations</div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            # Sentiment Distribution
            if "sentiment" in df_all.columns:
                with col1:
                    st.subheader("Sentiment Distribution")
                    sentiment_counts = df_all["sentiment"].value_counts()
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sentiment_counts.plot(kind="bar", ax=ax, color=['#ff6b6b', '#4ecdc4', '#45b7d1'])
                    ax.set_title("Sentiment Distribution")
                    ax.set_xlabel("Sentiment")
                    ax.set_ylabel("Count")
                    plt.xticks(rotation=45)
                    st.pyplot(fig)
            
            # Topic Distribution
            if "topic" in df_all.columns:
                with col2:
                    st.subheader("Topic Distribution")
                    topic_counts = df_all["topic"].value_counts()
                    fig, ax = plt.subplots(figsize=(8, 6))
                    topic_counts.plot(kind="bar", ax=ax, color=['#96ceb4', '#feca57', '#ff9ff3', '#54a0ff', '#5f27cd'])
                    ax.set_title("Topic Distribution")
                    ax.set_xlabel("Topic")
                    ax.set_ylabel("Count")
                    plt.xticks(rotation=45)
                    st.pyplot(fig)
            
            # Advanced Categorization Results
            if "category" in df_all.columns:
                st.subheader("Advanced Feedback Categorization")
                pattern_counts, _ = perform_advanced_categorization(df_all)
                
                # Visualization
                sorted_items = sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True)
                categories, counts = zip(*sorted_items) if sorted_items else ([], [])
                
                if categories:
                    fig, ax = plt.subplots(figsize=(12, 6))
                    bars = ax.bar(categories, counts, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f'])
                    ax.set_title("Feedback Pattern Analysis")
                    ax.set_ylabel("Count")
                    plt.xticks(rotation=45, ha='right')
                    
                    # Add count labels on bars
                    for bar, count in zip(bars, counts):
                        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                                str(count), ha='center', va='bottom', fontsize=10)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Recommendations
                    st.subheader("📝 Recommendations")
                    recommendations = generate_recommendations(pattern_counts)
                    for rec in recommendations:
                        st.write(f"• {rec}")
        
        # Summaries Section
        if st.session_state.get('summaries'):
            st.markdown('<div class="section-header">📄 Generated Summaries</div>', unsafe_allow_html=True)
            
            summaries = st.session_state['summaries']
            for source, summary in summaries.items():
                st.subheader(f"{source} Summary")
                st.write(summary)
        
        # Export Section
        st.markdown('<div class="section-header">💾 Export Data</div>', unsafe_allow_html=True)
        
        if st.button("📁 Download Results as CSV"):
            csv = df_all.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="feedback_analysis_results.csv",
                mime="text/csv"
            )
    
    # Upload existing data option
    st.markdown('<div class="section-header">📤 Upload Existing Data</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload a CSV file with feedback data", type=['csv'])
    
    if uploaded_file is not None:
        try:
            df_uploaded = pd.read_csv(uploaded_file)
            st.session_state.df_all = df_uploaded
            st.session_state.data_collected = True
            st.success(f"✅ Uploaded {len(df_uploaded)} records successfully!")
            st.dataframe(df_uploaded.head())
        except Exception as e:
            st.error(f"Error uploading file: {str(e)}")

if __name__ == "__main__":
    main()
