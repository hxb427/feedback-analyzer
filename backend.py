import praw
import requests
import pandas as pd
import re
from transformers import pipeline
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from io import BytesIO

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text

def run_analysis(reddit_client_id, reddit_client_secret, user_agent):
    # ==== Collect Reddit Posts ====
    reddit = praw.Reddit(
        client_id=reddit_client_id,
        client_secret=reddit_client_secret,
        user_agent=user_agent
    )
    reddit_posts = []
    for post in reddit.subreddit("all").search("Cursor IDE", limit=100):
        post.comments.replace_more(limit=0)
        comments = " ".join([comment.body for comment in post.comments.list()])
        reddit_posts.append({
            "source": "Reddit",
            "title": post.title,
            "text": post.selftext,
            "score": post.score,
            "comments": comments,
            "link": f"https://www.reddit.com{post.permalink}"
        })

    # ==== Collect Stack Overflow Posts ====
    url = "https://api.stackexchange.com/2.3/search"
    params = {"order": "desc", "sort": "activity", "intitle": "Cursor IDE", "site": "stackoverflow"}
    response = requests.get(url, params=params)
    data = response.json()
    stack_posts = [{
        "source": "Stack Overflow",
        "title": item.get("title", ""),
        "text": "",
        "score": item.get("score", 0),
        "comments": "",
        "link": item.get("link", "")
    } for item in data.get("items", [])]

    # ==== Merge both sources ====
    df_all = pd.DataFrame(reddit_posts + stack_posts)
    df_all["content_clean"] = df_all.apply(
        lambda row: clean_text((row.get("title","")+" "+row.get("text","")+" "+row.get("comments",""))),
        axis=1
    )

    # ==== Sentiment Analysis ====
    sentiment_classifier = pipeline("sentiment-analysis")
    df_all["sentiment"] = df_all["content_clean"].apply(lambda x: sentiment_classifier(x[:512])[0]['label'])

    # ==== Topic Classification ====
    topic_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    candidate_labels = ["bug report", "feature request", "praise", "performance issue", "usability feedback"]
    df_all["topic"] = df_all["content_clean"].apply(
        lambda x: topic_classifier(x[:512], candidate_labels)['labels'][0]
    )

    # ==== Word Cloud ====
    words = " ".join(df_all["content_clean"]).split()
    wordcloud = WordCloud(width=800, height=400).generate(" ".join(words))
    buf = BytesIO()
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.savefig(buf, format="png")
    buf.seek(0)

    # ==== Summarization ====
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    reddit_text = " ".join(df_all[df_all["source"]=="Reddit"]["content_clean"].tolist())[:3000]
    so_text = " ".join(df_all[df_all["source"]=="Stack Overflow"]["content_clean"].tolist())[:3000]
    reddit_summary = summarizer(reddit_text, max_length=130, min_length=50, do_sample=False)[0]['summary_text']
    so_summary = summarizer(so_text, max_length=130, min_length=50, do_sample=False)[0]['summary_text']

    return df_all, reddit_summary, so_summary, buf
