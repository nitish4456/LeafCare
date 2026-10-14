"""
LeafCare - Natural Language Processing (NLP) Advisory Assistant
College Applied AI Project - BCA AI & ML Viva Documentation

This module implements a transparent, rule-based NLP pipeline:
1. Knowledge Base Loading (from knowledge_base.json)
2. Text Preprocessing:
   - Lowercase normalization
   - Punctuation removal
   - Tokenization
   - Stopword filtering (preserving semantic intent indicators)
3. Keyword-Based Intent Identification:
   - Matches tokens and key phrases against domain vocabularies for 4 categories:
     * symptoms
     * causes
     * prevention
     * management
4. Knowledge-Base Retrieval:
   - Retrieves exact advisory guidelines corresponding to predicted leaf class.
   - Enforces strict groundedness: avoids external hallucination or unsupported claims.
   - Handles ambiguous, general, or unrecognized queries gracefully.
"""

import json
import os
import re
from typing import Dict, List, Optional, Tuple, Any


# Standard English stopwords to filter (excluding intent signals like 'why')
STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an",
    "and", "any", "are", "aren't", "as", "at", "be", "because", "been",
    "before", "being", "below", "between", "both", "but", "by", "can",
    "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does",
    "doesn't", "doing", "don't", "down", "during", "each", "few", "for",
    "from", "further", "had", "hadn't", "has", "hasn't", "have", "haven't",
    "having", "he", "her", "here", "hers", "herself", "him", "himself",
    "his", "how", "i", "if", "in", "into", "is", "isn't", "it", "its",
    "itself", "let's", "me", "more", "most", "mustn't", "my", "myself",
    "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other",
    "ought", "our", "ours", "ourselves", "out", "over", "own", "same",
    "shan't", "she", "should", "shouldn't", "so", "some", "such", "than",
    "that", "the", "their", "theirs", "them", "themselves", "then", "there",
    "these", "they", "this", "those", "through", "to", "too", "under",
    "until", "up", "very", "was", "wasn't", "we", "were", "weren't", "what",
    "when", "where", "which", "while", "who", "whom", "with", "won't",
    "would", "wouldn't", "you", "your", "yours", "yourself", "yourselves",
    "tell", "please", "know", "want"
}

# Mapping between metadata class names and knowledge base keys
CLASS_TO_KB_KEY = {
    "Tomato___healthy": "healthy",
    "Tomato___Early_blight": "early_blight",
    "Tomato___Late_blight": "late_blight",
    "healthy": "healthy",
    "early_blight": "early_blight",
    "late_blight": "late_blight"
}

# Human-readable labels for interface display
CLASS_DISPLAY_NAMES = {
    "healthy": "Tomato - Healthy",
    "early_blight": "Tomato - Early Blight",
    "late_blight": "Tomato - Late Blight"
}

# Domain-specific keyword vocabularies for rule-based intent classification
INTENT_KEYWORDS: Dict[str, List[str]] = {
    "symptoms": [
        "symptom", "symptoms", "sign", "signs", "look", "looks", "looking",
        "appearance", "appear", "appears", "spot", "spots", "ring", "rings",
        "concentric", "lesion", "lesions", "yellow", "yellowing", "brown",
        "dark", "water-soaked", "soak", "soaked", "foliage", "visual",
        "feature", "features", "identify", "identification", "detect",
        "observe", "characteristic", "characteristics", "pale", "recognize"
    ],
    "causes": [
        "cause", "causes", "caused", "causing", "reason", "reasons", "why",
        "occur", "occurs", "occurred", "occurring", "happen", "happens",
        "happened", "pathogen", "pathogens", "fungus", "fungi", "fungal",
        "organism", "alternaria", "solani", "phytophthora", "infestans",
        "source", "origin", "favour", "favor", "favorable", "conditions",
        "humidity", "humid", "cool", "warm", "wet", "wetness", "agent",
        "biology", "spread", "spreading", "develop", "development"
    ],
    "prevention": [
        "prevent", "prevention", "preventative", "preventive", "avoid",
        "avoiding", "avoided", "stop", "stopping", "safeguard", "protect",
        "protection", "protective", "precaution", "precautions", "guard",
        "guarding", "airflow", "air", "spacing", "debris", "hygiene",
        "sanitize", "material", "seed", "seedling", "prolonged", "alert",
        "alerts"
    ],
    "management": [
        "manage", "management", "treat", "treatment", "cure", "curing",
        "curative", "control", "controlling", "handle", "handling", "action",
        "actions", "remedy", "remedies", "solution", "care", "caring",
        "remove", "removing", "removal", "spray", "spraying", "overhead",
        "watering", "advice", "guidance", "measures", "steps", "recommendation",
        "recommendations", "deal"
    ]
}

# Multi-word key phrases for direct high-confidence intent routing
KEY_PHRASES: Dict[str, List[str]] = {
    "symptoms": [
        "what does it look like", "how does it look", "what are the symptoms",
        "signs and symptoms", "what are the signs", "visual signs"
    ],
    "causes": [
        "what causes", "why does it happen", "why does it occur", "why does this happen",
        "why does", "what is the cause", "how does it spread", "where does it come from"
    ],
    "prevention": [
        "how to prevent", "how can i prevent", "how do i prevent", "how to avoid",
        "prevention measures", "how to stop", "how to protect"
    ],
    "management": [
        "how to treat", "how to cure", "how to manage", "how to control",
        "what to do", "what should i do", "management practices", "treatment steps"
    ]
}


class NLPAssistant:
    """
    Lightweight rule-based NLP engine for plant disease advisory.
    Complies strictly with syllabus requirements: no external LLMs or black-box APIs.
    """

    def __init__(self, knowledge_base_path: str = "knowledge_base.json"):
        """
        Loads and verifies the prepared knowledge base JSON file.
        """
        self.knowledge_base_path = knowledge_base_path
        self.knowledge_base = self._load_knowledge_base()

    def _load_knowledge_base(self) -> Dict[str, Dict[str, str]]:
        """
        Reads knowledge_base.json with safety error checks.
        """
        if not os.path.exists(self.knowledge_base_path):
            raise FileNotFoundError(
                f"Knowledge base file not found at '{self.knowledge_base_path}'. "
                f"Please ensure knowledge_base.json is present in the project root."
            )

        with open(self.knowledge_base_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Validate structure
        for category in ["healthy", "early_blight", "late_blight"]:
            if category not in data:
                raise ValueError(
                    f"Knowledge base is missing required category '{category}'."
                )
            for intent in ["symptoms", "causes", "prevention", "management"]:
                if intent not in data[category]:
                    raise ValueError(
                        f"Knowledge base category '{category}' is missing '{intent}'."
                    )
        return data

    @staticmethod
    def clean_text(text: str) -> str:
        """
        Preprocesses raw user input:
        1. Converts text to lowercase
        2. Removes punctuation and special characters
        3. Normalizes whitespace
        """
        if not text:
            return ""
        # Lowercase conversion
        lowered = text.lower()
        # Remove punctuation marks and symbols, replacing with spaces
        cleaned = re.sub(r"[^\w\s]", " ", lowered)
        # Collapse multiple spaces
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    @staticmethod
    def tokenize(cleaned_text: str, remove_stopwords: bool = True) -> List[str]:
        """
        Splits preprocessed text into tokens and optionally filters stopwords.
        """
        tokens = cleaned_text.split()
        if remove_stopwords:
            tokens = [t for t in tokens if t not in STOPWORDS]
        return tokens

    def identify_intents(self, cleaned_text: str, tokens: List[str]) -> List[Tuple[str, int]]:
        """
        Scores user query against known intent keyword dictionaries and key phrases.
        Returns a sorted list of (intent, score) pairs in descending order of score.
        """
        scores: Dict[str, int] = {intent: 0 for intent in INTENT_KEYWORDS}

        # Step A: Check multi-word phrase patterns (higher weight)
        for intent, phrases in KEY_PHRASES.items():
            for phrase in phrases:
                if phrase in cleaned_text:
                    scores[intent] += 3

        # Step B: Check individual token matches
        for token in tokens:
            for intent, keywords in INTENT_KEYWORDS.items():
                if token in keywords:
                    scores[intent] += 2
                # Sub-string or stemming match (e.g. "treat" in "treatment")
                elif any(kw.startswith(token) or token.startswith(kw) for kw in keywords if len(token) >= 4 and len(kw) >= 4):
                    scores[intent] += 1

        # Sort by score descending
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return ranked

    def detect_category_override(self, cleaned_text: str) -> Optional[str]:
        """
        Checks if the user explicitly mentions a specific disease in their question
        (e.g., 'What are the symptoms of early blight?').
        """
        if "healthy" in cleaned_text:
            return "healthy"
        if "early blight" in cleaned_text or "early" in cleaned_text or "alternaria" in cleaned_text:
            return "early_blight"
        if "late blight" in cleaned_text or "late" in cleaned_text or "phytophthora" in cleaned_text:
            return "late_blight"
        return None

    def query(
        self,
        question: str,
        predicted_class: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Processes a natural language question and retrieves advisory information.

        Parameters:
            question (str): User's natural language question
            predicted_class (str, optional): Model prediction (e.g. 'Tomato___Early_blight' or 'early_blight')

        Returns:
            dict containing:
                - status: 'success', 'ambiguous', 'unsupported', 'missing_context'
                - category: determined category key (e.g., 'early_blight')
                - intent: identified intent (or 'all'/'general')
                - response: advisory text strictly from the knowledge base
                - tokens: processed tokens for viva transparency
                - intent_scores: raw score mapping for viva demonstration
        """
        if not question or not question.strip():
            return {
                "status": "empty",
                "category": None,
                "intent": None,
                "response": "Please enter a question to receive advisory information.",
                "tokens": [],
                "intent_scores": {}
            }

        # Step 1: Preprocessing & Tokenization
        cleaned = self.clean_text(question)
        tokens = self.tokenize(cleaned, remove_stopwords=True)
        all_tokens = self.tokenize(cleaned, remove_stopwords=False)

        # Step 2: Determine target category (either from user explicit query or current prediction)
        explicit_category = self.detect_category_override(cleaned)
        resolved_category = None

        if explicit_category:
            resolved_category = explicit_category
        elif predicted_class:
            resolved_category = CLASS_TO_KB_KEY.get(predicted_class, None)

        if not resolved_category:
            return {
                "status": "missing_context",
                "category": None,
                "intent": None,
                "response": (
                    "Please upload a tomato leaf image for classification first, or specify one of the supported "
                    "classes in your question (Healthy, Early Blight, or Late Blight)."
                ),
                "tokens": tokens,
                "intent_scores": {}
            }

        # Step 3: Intent Identification
        ranked_intents = self.identify_intents(cleaned, tokens)
        top_intent, top_score = ranked_intents[0]
        scores_dict = dict(ranked_intents)

        category_data = self.knowledge_base[resolved_category]
        category_name = CLASS_DISPLAY_NAMES.get(resolved_category, resolved_category.title())

        # Check for general overview queries (e.g., "tell me about this disease", "overview", "what is this")
        general_phrases = [
            "tell me about", "about this disease", "about this plant", "what is this disease",
            "overview", "general summary", "general info", "general information", "full details",
            "explain this disease", "all details", "summary of"
        ]
        has_general_indicator = any(p in cleaned for p in general_phrases)

        # Case A: Unsupported / Zero Keyword Match
        if top_score == 0:
            if has_general_indicator:
                # Provide a full structured summary of all 4 sections for that category
                full_summary = (
                    f"### Comprehensive Advisory Overview for {category_name}\n\n"
                    f"**Symptoms:** {category_data['symptoms']}\n\n"
                    f"**Causes:** {category_data['causes']}\n\n"
                    f"**Prevention:** {category_data['prevention']}\n\n"
                    f"**Management:** {category_data['management']}"
                )
                return {
                    "status": "success",
                    "category": resolved_category,
                    "intent": "general_overview",
                    "response": full_summary,
                    "tokens": tokens,
                    "intent_scores": scores_dict
                }

            return {
                "status": "unsupported",
                "category": resolved_category,
                "intent": None,
                "response": (
                    f"I could not identify the specific topic of your question for **{category_name}**.\n\n"
                    f"LeafCare's rule-based NLP assistant can answer questions related to:\n"
                    f"- **Symptoms** (e.g., *'What are the symptoms?'*, *'How does it look?'*)\n"
                    f"- **Causes** (e.g., *'What causes this?'*, *'Why does late blight occur?'*)\n"
                    f"- **Prevention** (e.g., *'How to prevent this disease?'*, *'How to protect the plant?'*)\n"
                    f"- **Management** (e.g., *'What is the treatment?'*, *'How do I manage it?'*)\n\n"
                    f"*Note: Queries are matched strictly against prepared local knowledge.*"
                ),
                "tokens": tokens,
                "intent_scores": scores_dict
            }

        # Case B: Multi-intent match (e.g., user asks "What are the causes and prevention?")
        matching_intents = [intent for intent, score in ranked_intents if score == top_score and score > 0]
        if len(matching_intents) > 1 and len(matching_intents) <= 2:
            combined_responses = []
            for intent in matching_intents:
                title = intent.capitalize()
                text = category_data[intent]
                combined_responses.append(f"**{title} ({category_name}):**\n{text}")

            return {
                "status": "success",
                "category": resolved_category,
                "intent": "+".join(matching_intents),
                "response": "\n\n".join(combined_responses),
                "tokens": tokens,
                "intent_scores": scores_dict
            }

        # Case C: Ambiguity check (more than 2 intents tied with equal score)
        if len(matching_intents) > 2:
            return {
                "status": "ambiguous",
                "category": resolved_category,
                "intent": None,
                "response": (
                    f"Your question matches multiple topics ({', '.join(matching_intents)}) for **{category_name}**. "
                    f"Please refine your question to focus on Symptoms, Causes, Prevention, or Management."
                ),
                "tokens": tokens,
                "intent_scores": scores_dict
            }

        # Case D: Single clear intent match
        retrieved_text = category_data[top_intent]
        formatted_response = (
            f"**{top_intent.capitalize()} ({category_name}):**\n\n{retrieved_text}"
        )

        return {
            "status": "success",
            "category": resolved_category,
            "intent": top_intent,
            "response": formatted_response,
            "tokens": tokens,
            "intent_scores": scores_dict
        }

    def get_all_guidance(self, category_key: str) -> Dict[str, str]:
        """
        Retrieves all 4 knowledge sections for a given category.
        Useful for displaying quick reference tabs or full advisory cards in Streamlit.
        """
        kb_key = CLASS_TO_KB_KEY.get(category_key, category_key)
        if kb_key in self.knowledge_base:
            return self.knowledge_base[kb_key]
        return {}
