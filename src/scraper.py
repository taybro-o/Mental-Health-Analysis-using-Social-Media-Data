"""
Keyless Data Scraping Engine for Contextual Emotion Distribution Vectorizer (CEDV)

Cascading zero-auth acquisition strategy:
1. Primary: Arctic Shift Search API (Recent 2024-2026 posts with pagination)
2. Secondary: Public Reddit .rss feeds
3. Conversational Anchors & Seed Fallback: Ensures both long Reddit posts and short conversational inputs are accurately learned.
"""

import hashlib
import json
import logging
import os
import re
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("CEDV_Scraper")

# 8-Class Affective Taxonomy Mapping with diversified subreddits
CLASS_TAXONOMY: Dict[int, Dict[str, object]] = {
    0: {
        "label": "depression",
        "subreddits": ["depression"],
        "description": "Persistent low mood, lethargy, loss of purpose, emotional numbness",
    },
    1: {
        "label": "anxiety",
        "subreddits": ["anxiety", "socialanxiety", "PanicAttack"],
        "description": "Panic, generalized worry, social apprehension, physiological tension",
    },
    2: {
        "label": "stress_burnout",
        "subreddits": ["stress", "burnout", "offmychest"],
        "description": "Overwork exhaustion, emotional exhaustion, feeling overwhelmed",
    },
    3: {
        "label": "adhd",
        "subreddits": ["adhd", "adhdwomen"],
        "description": "Executive dysfunction, focus/dopamine struggles, hyperactivity",
    },
    4: {
        "label": "autism",
        "subreddits": ["autism", "aspergers"],
        "description": "Neurodivergence, sensory overload, masking, social communication",
    },
    5: {
        "label": "loneliness",
        "subreddits": ["lonely", "Needafriend"],
        "description": "Isolation, lack of belonging, alienation",
    },
    6: {
        "label": "anger",
        "subreddits": ["anger", "vent"],
        "description": "Irritation, indignation, venting, emotional reactivity",
    },
    7: {
        "label": "positive_neutral",
        "subreddits": ["happy", "CasualConversation", "MadeMeSmile", "Gratitude"],
        "description": "Uplifting, grounded, gratitude, everyday non-clinical discourse",
    },
}

LABEL_TO_ID = {info["label"]: idx for idx, info in CLASS_TAXONOMY.items()}
ID_TO_LABEL = {idx: info["label"] for idx, info in CLASS_TAXONOMY.items()}

# Curated conversational anchors (ensures short conversational prompts are well-represented)
CONVERSATIONAL_ANCHORS: Dict[str, List[str]] = {
    "positive_neutral": [
        "I am feeling so good today!",
        "Feeling fantastic and loving life right now.",
        "Today has been such a wonderful and productive day.",
        "I am so happy and grateful for everything today.",
        "Small things made my day today: sunshine, good coffee, and smiles.",
        "Feeling joyful and relaxed after a great conversation.",
        "Life feels peaceful, balanced, and good.",
        "Celebrating a big win today! Feeling accomplished and cheerful.",
        "Had a wonderful morning walk, feeling refreshed and content.",
        "I love spending time with family, feeling blessed.",
        "Grateful for good health, supportive friends, and a sunny afternoon.",
        "Everything went smoothly today and I am in such a good mood.",
        "Just wanted to share some positivity and good vibes with everyone.",
        "Feeling energized, motivated, and happy about the future.",
        "Enjoying a quiet, peaceful evening with a great book.",
        "Today was great, I feel happy and at peace.",
        "Feeling proud of myself for reaching this milestone.",
        "Such a pleasant day today! Feeling happy and lighthearted.",
        "Woke up feeling positive, optimistic, and full of energy.",
        "I feel so thankful and blessed for my friends.",
    ],
    "depression": [
        "I want to die, nothing matters anymore.",
        "I feel so empty, hollow, and dead inside.",
        "Woke up wishing I hadn't. Depression is suffocating me.",
        "Everything feels completely hopeless and meaningless.",
        "I can't get out of bed, I have no energy or reason to live.",
        "The sadness is crushing me and I see no way forward.",
        "I have lost all interest in things I used to love.",
        "Feeling profoundly depressed and detached from reality.",
        "I am tired of pretending to be okay when I am broken inside.",
        "Crying uncontrollably and feeling an abyss of darkness.",
        "Depression is draining every drop of life out of me.",
        "I feel numb, lifeless, and invisible.",
        "There is no point in trying anymore, I always fail.",
        "I'm suffocating under this constant heavy cloud of depression.",
        "I feel like a burden to everyone and wish I could disappear.",
    ],
    "anxiety": [
        "I am having a severe panic attack, heart racing.",
        "My chest is tight and I can't catch my breath from panic.",
        "I am shaking with extreme fear and nervous dread.",
        "Overwhelmed with crippling anxiety about tomorrow.",
        "Can't stop spiraling into worst-case scenarios and terror.",
        "My hands are trembling and I feel lightheaded with anxiety.",
        "Sudden wave of panic out of nowhere, feeling like I'm dying.",
        "Paralyzed by social anxiety, terrified to speak or leave home.",
        "Irrational fear is consuming my mind all day long.",
        "Heart palpitations and dizziness from constant generalized anxiety.",
        "The anxiety is unbearable, I can't calm my nervous system down.",
        "Woke up in cold sweat with racing pulse and intense fear.",
        "Constantly on edge, anticipating disaster at any second.",
        "Terrified of judgment, my social phobia is ruining my life.",
        "Nausea and trembling from acute anxiety and panic.",
    ],
    "stress_burnout": [
        "I am completely burnt out from working 60 hours a week.",
        "Exhausted to the bone, I have nothing left to give.",
        "The work stress and endless deadlines are killing me.",
        "I feel overwhelmed by responsibilities, bills, and pressure.",
        "Chronic stress is giving me physical migraines and fatigue.",
        "I am at my breaking point, completely depleted and drained.",
        "My job has destroyed my mental health through toxic overwork.",
        "No matter how much I sleep, I wake up utterly exhausted.",
        "Drowning in unread emails, tasks, and demands with zero support.",
        "Burnout has turned me into an empty shell of a human.",
        "I just need a break from life, the pressure is unbearable.",
        "Stressed beyond measure, juggling too many impossible duties.",
        "Sunday evening dread about going to work is ruining my weekends.",
        "Mental exhaustion is paralyzing, I can't think straight.",
        "Pushed way past my limits, I am burning out fast.",
    ],
    "adhd": [
        "My ADHD executive dysfunction is paralyzing today.",
        "I cannot focus on anything, my brain is jumping everywhere.",
        "Sitting for hours trying to start one simple task, impossible.",
        "Hyperfocusing on useless rabbit holes instead of my work.",
        "My thoughts are racing at 100 miles an hour with no brakes.",
        "Constant brain fog and forgetfulness, lost my keys again.",
        "Procrastinating until the last panic-fueled minute as always.",
        "Dopamine deficiency makes boring routine tasks physically painful.",
        "I have 30 tabs open and can't finish a single project.",
        "Time blindness ruined my schedule completely today.",
        "Struggling with attention deficit, spacing out mid-conversation.",
        "ADHD paralysis has me stuck on the couch for 5 hours.",
        "Impulsive, restless, and completely unable to organize my life.",
        "My mind never shuts up, constantly bouncing between ideas.",
        "Executive dysfunction is not laziness, my brain just won't start.",
    ],
    "autism": [
        "Extreme sensory overload from fluorescent lights and loud noises.",
        "I am exhausted from masking all day around neurotypicals.",
        "Textures of this fabric make my skin crawl and panic.",
        "I struggle with eye contact, it feels intensely overwhelming.",
        "People speak in sarcasm and social games that I don't understand.",
        "Stimming helps regulate my nervous system after an overload.",
        "Routine change threw me into an immediate autistic meltdown.",
        "I feel like an alien observing humans from another planet.",
        "Special interest in trains, I can research it for 12 hours straight.",
        "Social cues and small talk drain my social battery instantly.",
        "Need my noise-cancelling headphones to survive outside.",
        "Masking my autistic traits for years has caused severe burnout.",
        "Sensory sensitivities to sounds, smells, and textures.",
        "Struggling with nonverbal communication and body language.",
        "Meltdown after a chaotic day in an overstimulating environment.",
    ],
    "loneliness": [
        "I am so lonely, I have zero friends to talk to.",
        "Nobody has texted me or checked on me in weeks.",
        "Sitting alone on a Friday night crying from sheer isolation.",
        "I feel completely invisible and abandoned by everyone.",
        "Surrounded by people but feeling utterly disconnected and alone.",
        "The silence in my room is deafening, I crave a human voice.",
        "I just want one genuine friend who actually cares about me.",
        "Eating alone, walking alone, living in complete solitude.",
        "Loneliness is eating away at my soul every single day.",
        "I feel like nobody in this entire world would notice if I vanished.",
        "Nobody reaches out unless they want something from me.",
        "The emptiness of having no one to share life with is painful.",
        "Deep yearning for emotional intimacy and companionship.",
        "Isolated from the world, watching everyone else live with friends.",
        "Heartbreakingly lonely, longing for genuine connection.",
    ],
    "anger": [
        "I am so furious and angry at everyone right now!",
        "My blood is boiling with intense rage and frustration.",
        "Tired of people disrespecting my boundaries and walking over me.",
        "I want to punch a wall, I am so goddamn furious.",
        "The blatant hypocrisy and injustice makes me sick with anger.",
        "Infuriated by how unfairly and cruelly I was treated today.",
        "Venting here because I am ready to explode on someone.",
        "Fuming with rage at terrible management and deceitful liars.",
        "I can't stand people who talk down to me with condescension.",
        "Losing my temper, I can't take this bullshit anymore.",
        "Blinded by rage and hatred for what they did to me.",
        "Screaming in my car to release the intense rage inside.",
        "Disgusted and irritated by rude, entitled, obnoxious people.",
        "Simmering with resentment and anger that won't go away.",
        "So damn mad my hands are shaking with adrenaline.",
    ],
}


def sanitize_text(text: str) -> str:
    """Basic text sanitization for deduplication and storage."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"\[(deleted|removed)\]", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compute_hash(text: str) -> str:
    """Generate MD5 hash of lowercased stripped text for deduplication."""
    clean = re.sub(r"\W+", "", text.lower())
    return hashlib.md5(clean.encode("utf-8")).hexdigest()


class CEDVScraper:
    """
    Cascading zero-auth Reddit data acquisition engine with pagination.
    """

    def __init__(self, request_delay: float = 0.5, user_agent: Optional[str] = None):
        self.request_delay = request_delay
        self.user_agent = user_agent or (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36 CEDVResearch/1.0"
        )
        self.headers = {"User-Agent": self.user_agent}

    def fetch_arctic_shift(
        self,
        subreddit: str,
        limit: int = 100,
        before_timestamp: Optional[int] = None,
    ) -> List[Dict]:
        """
        Fetch posts via Arctic Shift with pagination.
        """
        endpoint = "https://arctic-shift.photon-reddit.com/api/posts/search"
        params = {
            "subreddit": subreddit,
            "limit": min(limit, 100),
        }
        if before_timestamp:
            params["before"] = before_timestamp

        posts: List[Dict] = []
        try:
            resp = requests.get(endpoint, params=params, headers=self.headers, timeout=12)
            if resp.status_code == 200:
                data = resp.json()
                raw_items = data.get("data", [])
                for item in raw_items:
                    title = item.get("title", "") or ""
                    selftext = item.get("selftext", "") or ""
                    if selftext.strip() in ["[deleted]", "[removed]"]:
                        selftext = ""
                    combined = f"{title}. {selftext}".strip()
                    cleaned = sanitize_text(combined)
                    if len(cleaned.split()) >= 4:
                        posts.append({
                            "id": str(item.get("id", "")),
                            "text": cleaned,
                            "subreddit": subreddit,
                            "timestamp": int(item.get("created_utc", 0)),
                            "score": int(item.get("score", 0)),
                        })
                logger.info("Arctic Shift fetched %d posts from r/%s", len(posts), subreddit)
            else:
                logger.warning("Arctic Shift status %s for r/%s", resp.status_code, subreddit)
        except Exception as e:
            logger.warning("Arctic Shift error for r/%s: %s", subreddit, e)
        return posts

    def fetch_reddit_rss(self, subreddit: str, limit: int = 50) -> List[Dict]:
        """
        Fetch recent posts from public unauthenticated Reddit RSS feeds.
        """
        url = f"https://www.reddit.com/r/{subreddit}/new.rss?limit={limit}"
        posts: List[Dict] = []
        try:
            resp = requests.get(url, headers=self.headers, timeout=10)
            if resp.status_code == 200:
                root = ET.fromstring(resp.content)
                ns = {"atom": "http://www.w3.org/2005/Atom"}
                for entry in root.findall("atom:entry", ns):
                    title_elem = entry.find("atom:title", ns)
                    content_elem = entry.find("atom:content", ns)
                    id_elem = entry.find("atom:id", ns)
                    title = title_elem.text if title_elem is not None and title_elem.text else ""
                    raw_content = content_elem.text if content_elem is not None and content_elem.text else ""
                    clean_content = re.sub(r"<[^>]+>", " ", raw_content)
                    combined = f"{title}. {clean_content}".strip()
                    cleaned = sanitize_text(combined)
                    post_id = id_elem.text.split("/")[-1] if id_elem is not None and id_elem.text else ""
                    if len(cleaned.split()) >= 4:
                        posts.append({
                            "id": post_id,
                            "text": cleaned,
                            "subreddit": subreddit,
                            "timestamp": int(time.time()),
                            "score": 1,
                        })
                logger.info("Reddit RSS fetched %d posts from r/%s", len(posts), subreddit)
            else:
                logger.warning("Reddit RSS status %s for r/%s", resp.status_code, subreddit)
        except Exception as e:
            logger.warning("Reddit RSS error for r/%s: %s", subreddit, e)
        return posts

    def get_conversational_anchors(self, label: str) -> List[Dict]:
        """
        Return curated conversational statements for the label.
        """
        anchors = CONVERSATIONAL_ANCHORS.get(label, [])
        records = []
        for i, text in enumerate(anchors):
            records.append({
                "id": f"anchor_{label}_{i}",
                "text": text,
                "subreddit": f"anchor_{label}",
                "timestamp": int(time.time()),
                "score": 100,
            })
        return records

    def scrape_class(
        self,
        class_id: int,
        target_count: int = 250,
        seen_hashes: Optional[Set[str]] = None,
    ) -> List[Dict]:
        """
        Acquire target count of deduplicated posts for a specific class with pagination.
        """
        if seen_hashes is None:
            seen_hashes = set()

        info = CLASS_TAXONOMY[class_id]
        label = info["label"]
        subreddits = info["subreddits"]
        collected: List[Dict] = []

        logger.info("Collecting data for class %d (%s) from %s", class_id, label, subreddits)

        # 1. Inject conversational anchors first (ensures grounding)
        anchor_batch = self.get_conversational_anchors(label)
        for p in anchor_batch:
            h = compute_hash(p["text"])
            if h not in seen_hashes:
                seen_hashes.add(h)
                p["label"] = label
                p["class_id"] = class_id
                collected.append(p)

        # 2. Scrape subreddits with pagination
        for sub in subreddits:
            if len(collected) >= target_count:
                break
            
            # Paginate up to 3 pages per subreddit
            before_ts = None
            for page in range(3):
                if len(collected) >= target_count:
                    break
                batch = self.fetch_arctic_shift(sub, limit=100, before_timestamp=before_ts)
                if not batch:
                    break
                for p in batch:
                    h = compute_hash(p["text"])
                    if h not in seen_hashes:
                        seen_hashes.add(h)
                        p["label"] = label
                        p["class_id"] = class_id
                        collected.append(p)
                        if len(collected) >= target_count:
                            break
                # Set timestamp for next page
                timestamps = [p["timestamp"] for p in batch if p.get("timestamp")]
                if timestamps:
                    before_ts = min(timestamps)
                else:
                    break
                time.sleep(self.request_delay)

        # 3. Secondary Reddit RSS fallback if needed
        if len(collected) < target_count:
            for sub in subreddits:
                if len(collected) >= target_count:
                    break
                batch = self.fetch_reddit_rss(sub, limit=50)
                for p in batch:
                    h = compute_hash(p["text"])
                    if h not in seen_hashes:
                        seen_hashes.add(h)
                        p["label"] = label
                        p["class_id"] = class_id
                        collected.append(p)
                        if len(collected) >= target_count:
                            break
                time.sleep(self.request_delay)

        logger.info("Completed class %s: Total = %d posts", label, len(collected))
        return collected

    def scrape_all(
        self,
        samples_per_class: int = 250,
        output_csv: str = "data/dataset.csv",
    ) -> pd.DataFrame:
        """
        Scrape and assemble unified 8-class dataset.
        """
        seen_hashes: Set[str] = set()
        all_records: List[Dict] = []

        for class_id in sorted(CLASS_TAXONOMY.keys()):
            class_records = self.scrape_class(
                class_id=class_id,
                target_count=samples_per_class,
                seen_hashes=seen_hashes,
            )
            all_records.extend(class_records)

        df = pd.DataFrame(all_records)
        os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
        df.to_csv(output_csv, index=False)
        logger.info("Saved unified dataset (%d rows) to %s", len(df), output_csv)
        return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CEDV Keyless Dataset Scraper")
    parser.add_argument("--samples", type=int, default=250, help="Target samples per class")
    parser.add_argument("--output", type=str, default="data/dataset.csv", help="Path to output CSV")
    parser.add_argument("--delay", type=float, default=0.3, help="Delay between HTTP requests in seconds")
    args = parser.parse_args()

    scraper = CEDVScraper(request_delay=args.delay)
    df = scraper.scrape_all(samples_per_class=args.samples, output_csv=args.output)
    print("\nDataset Summary:")
    print(df["label"].value_counts())
