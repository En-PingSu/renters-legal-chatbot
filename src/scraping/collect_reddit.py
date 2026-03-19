"""
Collect real renter questions from Reddit for the evaluation benchmark.
NOTE: Reddit's JSON API now blocks unauthenticated requests (403).
Use PRAW with API credentials, or manually curate questions.
The evaluation questions in data/evaluation/reddit_questions.json were
manually written to represent realistic MA tenant law scenarios.
"""

import json
import re
import time
from pathlib import Path

import requests

from .utils import PROJECT_ROOT, USER_AGENT

EVAL_DIR = PROJECT_ROOT / "data" / "evaluation"

# Subreddits to search
SUBREDDITS = ["legaladvice", "renting", "bostonhousing", "boston"]

# Search queries for MA/Boston renter issues
SEARCH_QUERIES = [
    "security deposit Massachusetts",
    "eviction notice Boston",
    "landlord won't fix",
    "lease breaking MA",
    "rent increase Boston",
    "mold apartment Massachusetts",
    "bed bugs tenant rights",
    "landlord entering apartment",
    "heat not working tenant",
    "repair request landlord Massachusetts",
    "last month rent deposit",
    "tenant rights Boston",
    "withholding rent Massachusetts",
    "small claims landlord",
    "lease renewal Massachusetts",
]

# Keywords that indicate MA/Boston relevance
MA_KEYWORDS = [
    "massachusetts", "boston", "mass", " ma ", "cambridge", "somerville",
    "worcester", "springfield", "lowell", "quincy", "mgl", "masslegalhelp",
]

REQUEST_DELAY = 3  # Reddit is stricter about rate limiting


def search_reddit(subreddit: str, query: str, limit: int = 25) -> list[dict]:
    """Search a subreddit using Reddit JSON API."""
    url = f"https://www.reddit.com/r/{subreddit}/search.json"
    params = {
        "q": query,
        "restrict_sr": "on",
        "sort": "relevance",
        "t": "all",
        "limit": limit,
    }
    headers = {"User-Agent": USER_AGENT}

    time.sleep(REQUEST_DELAY)
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        posts = data.get("data", {}).get("children", [])
        return [p["data"] for p in posts if p["kind"] == "t3"]
    except Exception as e:
        print(f"  [ERROR] Reddit search failed ({subreddit}/{query}): {e}")
        return []


def is_ma_relevant(post: dict) -> bool:
    """Check if a post is relevant to Massachusetts/Boston."""
    text = f"{post.get('title', '')} {post.get('selftext', '')}".lower()
    # Check for flair
    flair = (post.get("link_flair_text") or "").lower()
    if "massachusetts" in flair or "boston" in flair:
        return True
    return any(kw in text for kw in MA_KEYWORDS)


def is_housing_question(post: dict) -> bool:
    """Check if the post is about a concrete housing/renter legal issue."""
    text = f"{post.get('title', '')} {post.get('selftext', '')}".lower()
    housing_kw = [
        "landlord", "tenant", "rent", "lease", "evict", "deposit",
        "apartment", "housing", "repair", "mold", "bed bug", "heat",
        "utility", "notice", "move out", "security deposit",
    ]
    return any(kw in text for kw in housing_kw)


def extract_question(post: dict) -> dict:
    """Extract relevant fields from a Reddit post."""
    return {
        "id": post["id"],
        "subreddit": post["subreddit"],
        "title": post["title"],
        "selftext": post.get("selftext", "")[:2000],  # truncate very long posts
        "url": f"https://www.reddit.com{post['permalink']}",
        "score": post.get("score", 0),
        "num_comments": post.get("num_comments", 0),
        "created_utc": post.get("created_utc"),
    }


def run(target_count: int = 50):
    """Collect Reddit evaluation questions."""
    print("=" * 60)
    print("Collecting Reddit Evaluation Questions")
    print("=" * 60)

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    seen_ids = set()
    candidates = []

    for subreddit in SUBREDDITS:
        for query in SEARCH_QUERIES:
            print(f"  Searching r/{subreddit} for: {query}")
            posts = search_reddit(subreddit, query, limit=10)

            for post in posts:
                if post["id"] in seen_ids:
                    continue
                seen_ids.add(post["id"])

                if not is_housing_question(post):
                    continue

                # For non-boston subreddits, require MA relevance
                if subreddit not in ["bostonhousing", "boston"]:
                    if not is_ma_relevant(post):
                        continue

                candidates.append(extract_question(post))

            if len(candidates) >= target_count * 2:
                break
        if len(candidates) >= target_count * 2:
            break

    # Sort by score (community validation) and take top candidates
    candidates.sort(key=lambda x: x["score"], reverse=True)
    selected = candidates[:target_count]

    # Save all candidates (for manual curation)
    all_path = EVAL_DIR / "reddit_candidates_all.json"
    with open(all_path, "w", encoding="utf-8") as f:
        json.dump(candidates, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved {len(candidates)} total candidates to {all_path.name}")

    # Save selected questions
    selected_path = EVAL_DIR / "reddit_questions.json"
    with open(selected_path, "w", encoding="utf-8") as f:
        json.dump(selected, f, indent=2, ensure_ascii=False)
    print(f"  Saved {len(selected)} selected questions to {selected_path.name}")

    print(f"\n{'=' * 60}")
    print(f"Done! {len(selected)} Reddit questions collected")
    print(f"{'=' * 60}")
    return selected


if __name__ == "__main__":
    run()
