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
        "subreddits": ["depression", "depression_help", "sad"],
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

# Curated conversational anchors (ensures short conversational prompts & slang are well-represented)
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
        "Vibing today, everything is falling into place.",
        "Huge W today, aced my test and got lunch with besties.",
        "This weather is immaculate, pure good vibes.",
        "Feeling blessed fr, life isn't perfect but I'm smiling.",
        "Today slaps, got so much done and relaxed after.",
        "Good morning world, let's make today an awesome day.",
        "Finished all my chores, now chilling with warm tea and Netflix.",
        "Nothing better than laughing until your stomach hurts with friends.",
        "Kindness really makes the world go round, someone paid for my coffee.",
        "Feeling hopeful and optimistic about this new job offer.",
        "Cooked a delicious homemade dinner, feeling very satisfied.",
        "Beautiful sunset tonight, took a moment just to appreciate it.",
        "My puppy learned a new trick today, so proud and wholesome.",
        "Finally submitted my project, huge weight off my chest and feeling great.",
        "Just a friendly reminder that you are loved and worthy.",
        "Peace of mind is truly priceless, loving this quiet weekend.",
        "Such a cozy rainy day with hot chocolate and music.",
        "Feeling calm, centered, and content with where I am.",
        "Everything turned out way better than expected, so relieved and glad.",
        "Gratitude attitude today, thankful for another breath and fresh air.",
        "Had the best sleep in weeks, ready to conquer the day.",
        "Simple joys: clean sheets, fresh breeze, and cold water.",
        "Got promoted today! Celebrated with pizza and ice cream.",
        "Life has its ups and downs, but right now I'm loving the view.",
        "Listening to my favorite upbeat playlist on full blast.",
        "The garden is blooming so nicely this spring.",
        "Random acts of kindness restore my faith in humanity.",
        "Feeling creative, inspired, and excited to paint again.",
        "Sending love and good energy to everyone reading this.",
        "Had a deep wholesome talk with an old friend, felt so recharging.",
        "So proud of my sibling graduating with honors today.",
        "Enjoying this tranquil Sunday morning before the week starts.",
        "The food at this new cafe was heavenly, 10/10 recommendation.",
        "Feeling serene, grateful, and aligned with my goals.",
        "Life is good, no complaints here.",
        "A good workout really resets the brain, feeling strong and upbeat.",
        "Smile at strangers, you never know whose day you might brighten.",
        "Embracing the slow moments and savoring every sip.",
        "What a delightful surprise receiving a handwritten letter in the mail.",
        "Everything is alright, taking it one happy step at a time.",
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
        "Lowkey depressed and can't even get out of my bed rn.",
        "L life honestly, why was I even born.",
        "Everything feels gray, flat, and hollow.",
        "I haven't showered in four days because what is the point.",
        "Existing is just painful, I wish I could go to sleep and not wake up.",
        "I am so tired of fighting this chronic emptiness inside.",
        "No motivation to eat, no motivation to move, just rotting in bed.",
        "Nobody would even notice or care if I was gone tomorrow.",
        "Deep down I feel like a complete waste of space.",
        "The emotional pain in my chest feels so heavy and physical.",
        "Staring at the ceiling for hours feeling utterly hopeless.",
        "I feel like a ghost haunting my own life.",
        "I hate myself so much, I can't look in the mirror without crying.",
        "Every morning brings dread because living is just too exhausting.",
        "Drowning in sorrow and darkness, unable to reach the surface.",
        "My soul feels bruised and exhausted beyond repair.",
        "The depression relapse hit hard this week, back to rock bottom.",
        "Nothing brings me joy anymore, not gaming, not music, nothing.",
        "Just sitting in a dark room crying silently so nobody hears.",
        "I feel fundamentally broken and unlovable.",
        "Self-hatred is eating me alive from the inside out.",
        "I don't see a future for myself at all.",
        "It feels like everyone else has a purpose except me.",
        "I am tired, not just physically, but deep in my spirit.",
        "Can't remember the last time I felt genuine happiness.",
        "Bed rotting all weekend because depression won again.",
        "I push everyone away because I don't want to drag them down.",
        "Crying in the shower so nobody sees how pathetic I am.",
        "I am suffocating in silence, too numb to even ask for help.",
        "The void inside keeps getting wider and colder.",
        "Life is just a chore list with zero reward at the end.",
        "I hate waking up, sleep is the only escape from my thoughts.",
        "I feel completely detached, like an empty vessel walking around.",
        "Lost in a fog of despair, wishing it would all just stop.",
        "There is no light at the end of this tunnel for me.",
        "I am so sick of feeling like this every single day.",
        "Broken beyond words, just waiting for the day to end.",
        "Nobody understands the depth of this darkness.",
        "I feel like an unwanted extra in the movie of life.",
        "The sadness is so heavy it feels like gravity doubled.",
        "I gave up on my dreams, I just don't care anymore.",
        "Why does breathing have to hurt this much emotionally?",
        "Rotting away in my bedroom while the world moves on.",
        "Hopelessness is the only constant companion I have.",
        "I feel completely finished, done with everything.",
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
        "I literally can't breathe rn, my heart is beating through my ribs.",
        "Having an anxiety spiral over an email I sent 10 minutes ago.",
        "Social anxiety is so bad I can't even order food or make a phone call.",
        "Dread is pooling in my stomach, feel like something horrible is about to happen.",
        "Hyperventilating in the bathroom, trying to ground myself with 54321.",
        "Panic attack hit me in the middle of the grocery store aisle.",
        "My mind is racing with catastrophic thoughts non-stop.",
        "Physical symptoms of anxiety are mimicking a heart attack again.",
        "Can't swallow, throat feels constricted from intense nervous panic.",
        "Paranoid that everyone is staring at me and whispering.",
        "Woke up at 3am with severe adrenaline surge and trembling.",
        "Agoraphobia is kicking in, stepping outside feels terrifying.",
        "Every notification on my phone sends a jolt of panic through my chest.",
        "Terrified I made a fatal mistake at work and will get fired.",
        "Can't calm down, pacing back and forth in my apartment.",
        "Trembling hands, ringing ears, cold sweats from pure panic.",
        "Social battery instantly wiped out by crippling social anxiety.",
        "Paralyzed by what-if thoughts, spiraling into absolute doom.",
        "I hate health anxiety, every twinge makes me think I'm terminally ill.",
        "Feeling like I'm losing control of my mind and body.",
        "Deep breathing isn't helping, the panic is overwhelming.",
        "Claustrophobia and panic trapped in this crowded subway.",
        "Can't eat anything because my stomach is in knots from nervous dread.",
        "Terrified of giving this presentation, voice shaking and legs wobbling.",
        "Adrenaline dump with no actual danger, my nervous system is fried.",
        "Feeling hyper-aware of every single heartbeat, so scared.",
        "Overthinking every single interaction I had today with intense anxiety.",
        "The panic came out of nowhere, hands numb and tingling.",
        "Dizziness and vertigo from hyperventilating during panic.",
        "Scared to drive on the highway because of sudden panic attacks.",
        "Walking on eggshells in my own mind, anxiety never lets up.",
        "Terrified of crowds, feels like the walls are closing in on me.",
        "Crippling fear of rejection and embarrassment paralyzing me.",
        "Chest pain from anxiety, wishing this terror would just end.",
        "Worrying about things that will never happen, but I can't stop.",
        "Panic disorder has made my world so small and confining.",
        "Feeling detached and derealized during this severe panic episode.",
        "My nervous system feels like an exposed live wire sparking.",
        "Afraid of going to sleep because I wake up gasping in terror.",
        "Anticipatory anxiety is ruining every special occasion for me.",
        "I feel so jumpy, every loud noise makes me leap out of my skin.",
        "Severe imposter syndrome and social dread consuming me.",
        "Nervous nausea is ruining my morning once again.",
        "Trapped in a cycle of fight-or-flight with no escape.",
        "Can't relax even on vacation, my anxiety follows me everywhere.",
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
        "The school sucks, the teachers give impossible homework and zero support.",
        "Work sucks so bad, corporate demands are completely unrealistic.",
        "This job is draining my soul, endless overtime for pennies.",
        "College is burning me out, midterm season is a nightmare.",
        "I have three exams tomorrow and haven't slept in 48 hours.",
        "Crying at my desk because the workload is literally impossible.",
        "School is killing my sanity, too many assignments and deadlines.",
        "The constant grind culture is toxic and destroying young people.",
        "I am working two jobs and going to university, surviving on caffeine.",
        "My boss expects 24/7 availability, no boundaries or respect.",
        "Burnout is real, I have zero creativity or energy left.",
        "Tired of carrying the weight of the entire department on my back.",
        "The school system sucks, all test cramming and no actual learning.",
        "Overwhelmed by student loans, rent, inflation, and bills piling up.",
        "Drowning in deadlines, feels like running on a treadmill at max speed.",
        "Physical burnout: back aches, headaches, and chronic eye strain.",
        "Can't switch my brain off after 12 hours of spreadsheets and meetings.",
        "Too tired to cook, too tired to clean, living in survival mode.",
        "My toxic workplace is pushing me toward an emotional breakdown.",
        "Every single day is the same exhausting grind with no relief.",
        "Exhausted from pretending to care about corporate buzzwords.",
        "Studying 14 hours a day for this licensing exam is soul crushing.",
        "I feel like a cog in a machine grinding down to dust.",
        "The academic pressure from my parents and teachers is suffocating.",
        "Workplace stress has ruined my physical health and immune system.",
        "No time for friends, no time for hobbies, just work and sleep.",
        "Deadlines are breathing down my neck and I can't keep up.",
        "Burning the candle at both ends, running on fumes.",
        "High school curriculum is completely broken and outdated, it sucks.",
        "I am so overwhelmed by life admin, taxes, and adult responsibilities.",
        "Chronic burnout is making me forget basic things and feel dumb.",
        "My job gave me ulcers from non-stop high-stakes stress.",
        "Semester from hell, five projects due on the same Monday.",
        "Unreasonable expectations and micromanagement are suffocating me.",
        "I am mentally and physically depleted, desperate for a vacation.",
        "The pressure to be high-achieving is literally breaking my mind.",
        "Working 12-hour shifts without a lunch break is illegal and exhausting.",
        "Stress acne, stress hair loss, stress insomnia — my body is warning me.",
        "Can't even enjoy my day off because I'm stressing about Monday.",
        "This university program is designed to weed people out through torture.",
        "Stressed to the max, feel like a rubber band stretched to snapping point.",
        "Toxic productivity guilt won't let me rest for even ten minutes.",
        "Endless Zoom calls and emails are melting my brain.",
        "I need a sabbatical, my battery is at negative twenty percent.",
        "Workplace exploitation has pushed me past my human limits.",
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
        "My brain is fried ngl, ADHD dopamine seeking is at an all-time high.",
        "Can't focus on this lecture to save my life, doodling on margins.",
        "Went to kitchen for water, ended up reorganizing the spice rack.",
        "Forgot to reply to a text for 3 weeks because of object permanence.",
        "ADHD tax is real: paid late fees on three bills I forgot existed.",
        "Hyperfixated on crochet for 10 hours and forgot to eat lunch.",
        "Starting 20 different hobbies and abandoning them within two weeks.",
        "My attention span is cooked, can't even sit through a 5-minute video.",
        "Rejection sensitive dysphoria (RSD) hitting hard after minor feedback.",
        "ADHD medication shortage is making work and school an absolute nightmare.",
        "Spacing out while someone is talking directly to my face, again.",
        "Restless legs and fidgeting constantly, I cannot sit still.",
        "Lost my glasses while they were on top of my own head.",
        "Understimulated and bored out of my mind, needing dopamine instantly.",
        "Cleaning my room turned into finding an old diary and reading it for 4 hours.",
        "Procrastivity: doing everything except the one urgent task on my list.",
        "Brain feels like a browser with 50 tabs open and music playing somewhere.",
        "Interrupting people accidentally because I'll forget the thought if I wait.",
        "Impulsively bought 100 dollars worth of craft supplies I will never use.",
        "Time blindness is awful, I genuinely thought 10 minutes passed, it was 2 hours.",
        "Executive dysfunction makes brushing teeth feel like climbing Mount Everest.",
        "Struggling with working memory, what was I just doing two seconds ago?",
        "Can't prioritize tasks, everything feels either urgent or impossible.",
        "Overwhelmed by messy clutter but paralyzed to clean it up.",
        "Binge eating junk food just for the quick dopamine hit.",
        "Hyperfocused until 4 AM researching random historical events.",
        "ADHD paralysis: screaming in my head to move, but body won't budge.",
        "Changing tasks requires immense mental energy and causes friction.",
        "Forgot my laundry in the washing machine for the third time this week.",
        "My brain refuses to engage with boring paperwork, it feels painful.",
        "Constantly misplacing wallet, phone, keys, and water bottle.",
        "Emotional dysregulation from ADHD makes small frustrations feel catastrophic.",
        "Can't follow verbal multi-step directions, please write it down.",
        "Zoning out in every meeting, catching maybe 10% of what was said.",
        "The dopamine crash after hyperfocus is so physically exhausting.",
        "Living with an ADHD brain is like driving a Ferrari with bicycle brakes.",
        "Stimulant meds wore off, brain is officially scrambled eggs.",
        "Started writing an email, got distracted by research, forgot to send it.",
        "Fidgeting with pen, clicking it repeatedly to stay alert in class.",
        "ADHD burnout is when your hyperactive mind finally crashes hard.",
        "Struggling to regulate attention: either zero focus or 500% laser focus.",
        "Chronically late to everything despite waking up two hours early.",
        "Daydreaming vivid scenarios instead of doing my actual assignments.",
        "Why can I code for 12 hours straight but can't do 5 minutes of dishes?",
        "Neurodivergent brain just operates on interest, urgency, and novelty.",
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
        "The grocery store was pure torture: flickering lights, beeping, crowds.",
        "Sensory shutdown: lost my ability to speak, going nonverbal.",
        "Why can't neurotypicals just say what they mean directly?",
        "My special interest is paleontology, I could talk about it for days.",
        "Rocking back and forth and hand-flapping to self-soothe.",
        "Food textures are a sensory nightmare, I can only eat safe foods.",
        "Masking at work is slowly killing my authentic identity.",
        "Unexpected visitors showed up and completely shattered my routine.",
        "Eye contact feels like an intimate, aggressive laser beam into my soul.",
        "Hyperacusis makes chewing sounds and refrigerator hum physically painful.",
        "Feeling deeply misunderstood because I speak very literally.",
        "Autistic burnout after years of camouflaging my natural traits.",
        "Small talk about the weather feels like an illogical waste of time.",
        "Sensory friendly clothing with no tags and soft seams is a lifesaver.",
        "Need hours of alone time in a quiet dark room to recover from socialization.",
        "Social interaction feels like performing in a play where I never got the script.",
        "Stimming with weighted blanket and fidget spinner to ground myself.",
        "Having an autistic shutdown, sitting in a dark closet for quiet.",
        "People think I'm blunt or rude when I'm just stating objective facts.",
        "I can't filter out background noise, every conversation enters my brain equally.",
        "Infodumping about my special interest to anyone who will listen.",
        "The sound of buzzing lights is driving me insane right now.",
        "Struggling with unwritten social rules that nobody explicitly taught me.",
        "Autistic joy when engaging in my deep special interest is unmatched.",
        "Changes in plans give me acute panic and distress.",
        "Wearing sunglasses indoors because the office lighting gives me migraines.",
        "Echolalia: repeating phrases from movies and memes out loud.",
        "I feel so disconnected from neurotypical socialization.",
        "Meltdown from being touched without warning or consent.",
        "Texture of soggy vegetables makes me instantly gag.",
        "Social exhaustion after just one hour of networking.",
        "People misinterpret my flat affect and monotone voice as anger.",
        "Needing explicit step-by-step instructions with zero ambiguity.",
        "Autism diagnosis as an adult finally explained my whole life.",
        "Hyper-focused on my collection, organizing it brings deep peace.",
        "The world is designed for neurotypicals, everything is too loud and bright.",
        "Mask slipped at work and people gave me weird looks.",
        "Overstimulated by perfume, hairspray, and cologne in the elevator.",
        "Social battery ran out, can't make small talk with coworkers.",
        "Safe food hyperfixation: eating the exact same meal for six months.",
        "Flapping my hands when excited, unmasking feels so freeing.",
        "Struggling to read between the lines and decode hidden hints.",
        "Meltdown recovery takes days of complete isolation.",
        "Sensory overload from crying babies, sirens, and blaring TVs.",
        "Neurodivergent pride: embracing my autistic brain quirks.",
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
        "My phone has zero unread notifications except spam and ads.",
        "Another weekend spent completely alone in my four walls.",
        "Watching groups of friends laughing outside my window hurts so badly.",
        "I crave a hug or physical touch so much it physically aches.",
        "Feeling like an outcast who doesn't belong anywhere or to anyone.",
        "I haven't had a real deep conversation with anyone in months.",
        "Nobody ever invites me anywhere, always an afterthought.",
        "Moving to a new city alone has been brutally isolating.",
        "I talk to my pet because there is literally no human in my life.",
        "Sitting by myself in the college cafeteria watching everyone socialize.",
        "Loneliness is like a cold winter inside my chest that won't thaw.",
        "It's my birthday today and not a single person remembered.",
        "I feel like a ghost drifting through a world full of connected people.",
        "Scrolling social media seeing everyone's friend groups makes me weep.",
        "I just want someone to ask me 'how was your day?' and care.",
        "Empty apartment, eating cereal alone in the dark.",
        "I am so tired of doing every single activity by myself.",
        "Longing for a partner, feeling like I am unlovable and undesirable.",
        "Left on read by the only person I thought was my friend.",
        "Alienated from my family and having no community to lean on.",
        "I feel like a stray dog begging for scraps of friendship.",
        "The emotional ache of extreme isolation is slowly killing me.",
        "Going days without uttering a single spoken word out loud.",
        "Everyone already has established friend circles, no room for me.",
        "I feel so disposable to people, easily forgotten and replaced.",
        "Solitude was peaceful at first, now it feels like a prison cell.",
        "Sitting in the park watching couples and families, feeling invisible.",
        "Nobody to share my good news with, nobody to share my grief with.",
        "Desperate for emotional connection, tired of this solitary existence.",
        "I have 500 followers online but zero people to call in an emergency.",
        "The silence in this house is overwhelming and heavy.",
        "Feeling forgotten by everyone I went to high school or college with.",
        "I eat all my meals at restaurants alone with headphones on.",
        "I just want a community where I feel welcomed and accepted.",
        "Chronic loneliness hurts more than physical injury.",
        "Nobody knows the real me because nobody has ever asked.",
        "Yearning for a shoulder to lean on, someone who gets me.",
        "I am so starved of companionship and genuine warmth.",
        "Fading into the background, a spectator to other people's lives.",
        "Tired of being the one who always initiates and reaches out first.",
        "Crying into my pillow because I have no one to talk to.",
        "Feeling completely cut off from humanity, like an astronaut lost in space.",
        "Why is it so hard to make genuine, loyal adult friends?",
        "The loneliness epidemic is real and it has swallowed me whole.",
        "Surrounded by a crowd in the city, but utterly alone inside.",
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
        "The school sucks, the administration is corrupt and completely useless.",
        "This place sucks so much, everything is broken and nobody cares.",
        "My landlord is a greedy parasitic scammer and I'm ready to sue.",
        "I hate this toxic company and every incompetent manager in it.",
        "People who cut in traffic without signaling deserve the worst.",
        "Sick and tired of being taken advantage of by selfish freeloaders.",
        "I am so pissed off, my blood pressure is through the roof.",
        "Lying, manipulative cheaters make me absolutely sick to my stomach.",
        "Furious at how the medical system dismissed my symptoms for years.",
        "I am tired of playing nice, I am going to tear them a new one.",
        "This customer service is pure garbage, wasted two hours of my life.",
        "Enraged by people who abuse animals or mistreat the vulnerable.",
        "Stop telling me to calm down, I have every right to be furious!",
        "The audacity of this person to blame me for their own screw up.",
        "Sick of being gaslit by narcissists who never take accountability.",
        "I am seething with anger over this scam, money stolen.",
        "Flipping tables angry, the level of disrespect is intolerable.",
        "Nothing pisses me off more than arrogant condescending know-it-alls.",
        "I swear people lose all common sense and decency nowadays.",
        "Betrayed by someone I trusted with everything, pure burning rage.",
        "The referee completely cheated us, worst officiating in history.",
        "Raging at my computer, this software crash deleted 4 hours of work.",
        "Fed up with lazy coworkers who do zero work and take all the credit.",
        "I want to scream at the top of my lungs, so enraged right now.",
        "How dare they talk behind my back and spread vicious rumors.",
        "Disrespectful neighbors blasting music at 3 AM, so tempted to call police.",
        "Corrupt politicians selling out ordinary citizens makes me so angry.",
        "I am done keeping the peace, gloves are coming off.",
        "Infuriating how rich criminals get away with everything while poor get punished.",
        "Don't test my patience today, I am on the verge of snapping.",
        "Spite and anger are fueling me to prove every single doubter wrong.",
        "Irritated by every little sound, everything is grating on my nerves.",
        "The total incompetence of this airline ruined my whole trip.",
        "Livid that they stole my design and claimed it as their own work.",
        "Stop making excuses for shitty behavior, hold people accountable!",
        "I hate feeling powerless against greedy landlords and corporations.",
        "My temper flared up and I slammed the door so hard it rattled.",
        "Fuming at the blatant double standards and bias.",
        "Rude retail customers treating workers like garbage makes me furious.",
        "I have had it up to here with this constant disrespect and mockery.",
        "Burning bridges with toxic relatives who only bring drama and venom.",
        "Furious that my hard earned money was wasted on defective trash.",
        "The sheer stupidity of this decision is astounding and infuriating.",
        "Ready to confront them face to face, I'm not backing down anymore.",
        "Anger is an understatement, I am utterly disgusted and hostile.",
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
        for attempt in range(2):
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
                    break
                else:
                    logger.warning("Arctic Shift status %s for r/%s (attempt %d)", resp.status_code, subreddit, attempt + 1)
                    time.sleep(1.0)
            except Exception as e:
                logger.warning("Arctic Shift error for r/%s (attempt %d): %s", subreddit, attempt + 1, e)
                time.sleep(1.0)
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
        target_count: int = 550,
        seen_hashes: Optional[Set[str]] = None,
    ) -> List[Dict]:
        """
        Acquire target count of deduplicated posts for a specific class with deep pagination
        distributed across all associated subreddits.
        """
        if seen_hashes is None:
            seen_hashes = set()

        info = CLASS_TAXONOMY[class_id]
        label = info["label"]
        subreddits = info["subreddits"]
        collected: List[Dict] = []

        logger.info("Collecting data for class %d (%s) from %s", class_id, label, subreddits)

        # 1. Inject conversational anchors first (ensures grounding across slang & colloquial speech)
        anchor_batch = self.get_conversational_anchors(label)
        for p in anchor_batch:
            h = compute_hash(p["text"])
            if h not in seen_hashes:
                seen_hashes.add(h)
                p["label"] = label
                p["class_id"] = class_id
                collected.append(p)

        remaining_needed = max(0, target_count - len(collected))
        per_sub_target = max(60, remaining_needed // len(subreddits) + 30)

        # 2. Scrape each subreddit with up to 8 pages of pagination
        for sub in subreddits:
            if len(collected) >= target_count:
                break
            
            sub_count = 0
            before_ts = None
            for page in range(8):
                if len(collected) >= target_count or sub_count >= per_sub_target:
                    break
                batch = self.fetch_arctic_shift(sub, limit=100, before_timestamp=before_ts)
                if not batch:
                    break
                added_in_page = 0
                for p in batch:
                    h = compute_hash(p["text"])
                    if h not in seen_hashes:
                        seen_hashes.add(h)
                        p["label"] = label
                        p["class_id"] = class_id
                        collected.append(p)
                        sub_count += 1
                        added_in_page += 1
                        if len(collected) >= target_count or sub_count >= per_sub_target:
                            break
                if added_in_page == 0:
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
        samples_per_class: int = 550,
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
    parser.add_argument("--samples", type=int, default=550, help="Target samples per class")
    parser.add_argument("--output", type=str, default="data/dataset.csv", help="Path to output CSV")
    parser.add_argument("--delay", type=float, default=0.25, help="Delay between HTTP requests in seconds")
    args = parser.parse_args()

    scraper = CEDVScraper(request_delay=args.delay)
    df = scraper.scrape_all(samples_per_class=args.samples, output_csv=args.output)
    print("\nDataset Summary:")
    print(df["label"].value_counts())
