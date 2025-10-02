How is this for a semantic Delta module:

"""
Enhanced Semantic Delta Module for OMNI-Dharma
Computes multi-dimensional semantic differences between AI outputs
with improved granularity and robustness.
"""

import difflib
import re
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer, util
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

@dataclass
class SemanticDeltaResult:
"""Structured result container for semantic delta analysis."""
surface_similarity: float
semantic_similarity: float
structural_similarity: float
lexical_diversity_delta: float
sentiment_delta: Optional[float]
topic_coherence_delta: float
sentence_level_deltas: List[float]
token_level_analysis: Dict[str, float]
confidence_score: float

@property  
def composite_delta(self) -> float:  
    """Weighted composite delta score."""  
    weights = {  
        'semantic': 0.4,  
        'structural': 0.2,  
        'surface': 0.2,  
        'lexical': 0.1,  
        'topic': 0.1  
    }  
      
    deltas = {  
        'semantic': 1 - self.semantic_similarity,  
        'structural': 1 - self.structural_similarity,  
        'surface': 1 - self.surface_similarity,  
        'lexical': self.lexical_diversity_delta,  
        'topic': self.topic_coherence_delta  
    }  
      
    return sum(weights[k] * deltas[k] for k in weights.keys())

class EnhancedSemanticDelta:
"""Enhanced semantic delta analyzer with multi-dimensional analysis."""

def __init__(self,   
             embedding_model: str = "all-MiniLM-L6-v2",  
             device: Optional[str] = None,  
             cache_embeddings: bool = True):  
    """  
    Initialize the semantic delta analyzer.  
      
    Args:  
        embedding_model: HuggingFace model name for embeddings  
        device: Torch device ('cuda', 'cpu', or None for auto)  
        cache_embeddings: Whether to cache embeddings for repeated analysis  
    """  
    self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')  
    self.embedder = SentenceTransformer(embedding_model, device=self.device)  
    self.cache_embeddings = cache_embeddings  
    self._embedding_cache = {}  
      
def _get_embedding(self, text: str) -> torch.Tensor:  
    """Get embedding with optional caching."""  
    if self.cache_embeddings and text in self._embedding_cache:  
        return self._embedding_cache[text]  
          
    embedding = self.embedder.encode(text, convert_to_tensor=True)  
      
    if self.cache_embeddings:  
        self._embedding_cache[text] = embedding  
          
    return embedding  
  
def compute_surface_delta(self, text1: str, text2: str) -> float:  
    """  
    Enhanced surface-level similarity using multiple metrics.  
    Combines character-level, word-level, and sequence alignment.  
    """  
    # Character-level similarity  
    char_matcher = difflib.SequenceMatcher(None, text1, text2)  
    char_similarity = char_matcher.ratio()  
      
    # Word-level similarity  
    words1 = text1.split()  
    words2 = text2.split()  
    word_matcher = difflib.SequenceMatcher(None, words1, words2)  
    word_similarity = word_matcher.ratio()  
      
    # Weighted combination (favor word-level for semantic relevance)  
    return 0.3 * char_similarity + 0.7 * word_similarity  
  
def compute_semantic_similarity(self, text1: str, text2: str) -> float:  
    """  
    Multi-level semantic similarity using embeddings and hierarchical analysis.  
    """  
    # Full text embeddings  
    emb1 = self._get_embedding(text1)  
    emb2 = self._get_embedding(text2)  
    full_similarity = util.cos_sim(emb1, emb2).item()  
      
    # Sentence-level analysis for longer texts  
    sentences1 = self._split_sentences(text1)  
    sentences2 = self._split_sentences(text2)  
      
    if len(sentences1) > 1 and len(sentences2) > 1:  
        sentence_similarities = self._compute_sentence_alignment(sentences1, sentences2)  
        sentence_avg = np.mean(sentence_similarities) if sentence_similarities else full_similarity  
          
        # Weight full text more heavily, but consider sentence structure  
        return 0.7 * full_similarity + 0.3 * sentence_avg  
      
    return full_similarity  
  
def compute_structural_similarity(self, text1: str, text2: str) -> float:  
    """  
    Analyze structural similarity: sentence count, length distribution, punctuation patterns.  
    """  
    sentences1 = self._split_sentences(text1)  
    sentences2 = self._split_sentences(text2)  
      
    # Sentence count similarity  
    sent_count_sim = 1 - abs(len(sentences1) - len(sentences2)) / max(len(sentences1), len(sentences2), 1)  
      
    # Length distribution similarity  
    lengths1 = [len(s.split()) for s in sentences1]  
    lengths2 = [len(s.split()) for s in sentences2]  
      
    if lengths1 and lengths2:  
        # Use histogram comparison for length distributions  
        max_len = max(max(lengths1), max(lengths2))  
        bins = min(10, max_len + 1)  
          
        hist1, _ = np.histogram(lengths1, bins=bins, range=(0, max_len))  
        hist2, _ = np.histogram(lengths2, bins=bins, range=(0, max_len))  
          
        # Normalize histograms  
        hist1 = hist1 / np.sum(hist1) if np.sum(hist1) > 0 else hist1  
        hist2 = hist2 / np.sum(hist2) if np.sum(hist2) > 0 else hist2  
          
        length_sim = 1 - np.sum(np.abs(hist1 - hist2)) / 2  
    else:  
        length_sim = 1.0  
      
    # Punctuation pattern similarity  
    punct_sim = self._compute_punctuation_similarity(text1, text2)  
      
    return 0.4 * sent_count_sim + 0.4 * length_sim + 0.2 * punct_sim  
  
def compute_lexical_diversity_delta(self, text1: str, text2: str) -> float:  
    """  
    Compute the difference in lexical diversity (vocabulary richness).  
    """  
    def lexical_diversity(text: str) -> float:  
        words = re.findall(r'\b\w+\b', text.lower())  
        return len(set(words)) / len(words) if words else 0  
      
    diversity1 = lexical_diversity(text1)  
    diversity2 = lexical_diversity(text2)  
      
    return abs(diversity1 - diversity2)  
  
def compute_topic_coherence_delta(self, text1: str, text2: str) -> float:  
    """  
    Analyze topic coherence using TF-IDF and key term extraction.  
    """  
    try:  
        # Use TF-IDF to identify key terms  
        vectorizer = TfidfVectorizer(max_features=100, stop_words='english')  
          
        # Combine texts for consistent vocabulary  
        combined = [text1, text2]  
        tfidf_matrix = vectorizer.fit_transform(combined)  
          
        # Compute cosine similarity between TF-IDF vectors  
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0, 0]  
          
        return 1 - similarity  # Return delta (difference)  
          
    except ValueError:  
        # Handle case where texts are too short or empty  
        return 0.0  
  
def _split_sentences(self, text: str) -> List[str]:  
    """Split text into sentences using multiple delimiters."""  
    sentences = re.split(r'[.!?]+', text)  
    return [s.strip() for s in sentences if s.strip()]  
  
def _compute_sentence_alignment(self, sentences1: List[str], sentences2: List[str]) -> List[float]:  
    """  
    Compute optimal alignment between sentences from two texts.  
    """  
    if not sentences1 or not sentences2:  
        return []  
      
    # Create similarity matrix  
    similarity_matrix = np.zeros((len(sentences1), len(sentences2)))  
      
    for i, s1 in enumerate(sentences1):  
        emb1 = self._get_embedding(s1)  
        for j, s2 in enumerate(sentences2):  
            emb2 = self._get_embedding(s2)  
            similarity_matrix[i, j] = util.cos_sim(emb1, emb2).item()  
      
    # Extract best alignments (greedy approach)  
    alignments = []  
    used_j = set()  
      
    for i in range(len(sentences1)):  
        best_j = -1  
        best_score = -1  
          
        for j in range(len(sentences2)):  
            if j not in used_j and similarity_matrix[i, j] > best_score:  
                best_score = similarity_matrix[i, j]  
                best_j = j  
          
        if best_j != -1:  
            alignments.append(best_score)  
            used_j.add(best_j)  
      
    return alignments  
  
def _compute_punctuation_similarity(self, text1: str, text2: str) -> float:  
    """Compare punctuation usage patterns."""  
    punct_chars = set('.,!?;:()[]{}"-')  
      
    punct1 = [c for c in text1 if c in punct_chars]  
    punct2 = [c for c in text2 if c in punct_chars]  
      
    if not punct1 and not punct2:  
        return 1.0  
      
    # Count punctuation types  
    count1 = {p: punct1.count(p) for p in punct_chars}  
    count2 = {p: punct2.count(p) for p in punct_chars}  
      
    # Normalize counts  
    total1 = sum(count1.values())  
    total2 = sum(count2.values())  
      
    if total1 == 0 and total2 == 0:  
        return 1.0  
      
    norm1 = {p: count1[p] / total1 if total1 > 0 else 0 for p in punct_chars}  
    norm2 = {p: count2[p] / total2 if total2 > 0 else 0 for p in punct_chars}  
      
    # Compute similarity  
    similarity = 1 - sum(abs(norm1[p] - norm2[p]) for p in punct_chars) / 2  
    return similarity  
  
def _compute_confidence_score(self, text1: str, text2: str, results: Dict) -> float:  
    """  
    Compute confidence score based on text length and analysis reliability.  
    """  
    min_length = min(len(text1), len(text2))  
      
    # Base confidence on text length (more text = more reliable analysis)  
    length_factor = min(1.0, min_length / 100)  # Assume 100 chars for good confidence  
      
    # Consistency check: semantic and surface similarity should be correlated  
    sem_surf_consistency = 1 - abs(results['semantic_similarity'] - results['surface_similarity'])  
      
    return 0.7 * length_factor + 0.3 * sem_surf_consistency  
  
def analyze(self, text1: str, text2: str) -> SemanticDeltaResult:  
    """  
    Perform comprehensive semantic delta analysis.  
      
    Args:  
        text1: First text (e.g., Gemma output)  
        text2: Second text (e.g., OMNI output)  
          
    Returns:  
        SemanticDeltaResult with comprehensive analysis  
    """  
    # Clean inputs  
    text1 = text1.strip()  
    text2 = text2.strip()  
      
    # Compute all similarity metrics  
    surface_sim = self.compute_surface_delta(text1, text2)  
    semantic_sim = self.compute_semantic_similarity(text1, text2)  
    structural_sim = self.compute_structural_similarity(text1, text2)  
    lexical_delta = self.compute_lexical_diversity_delta(text1, text2)  
    topic_delta = self.compute_topic_coherence_delta(text1, text2)  
      
    # Sentence-level analysis for detailed inspection  
    sentences1 = self._split_sentences(text1)  
    sentences2 = self._split_sentences(text2)  
    sentence_deltas = self._compute_sentence_alignment(sentences1, sentences2)  
      
    # Token-level analysis  
    token_analysis = self._analyze_token_changes(text1, text2)  
      
    # Compute confidence  
    temp_results = {  
        'semantic_similarity': semantic_sim,  
        'surface_similarity': surface_sim  
    }  
    confidence = self._compute_confidence_score(text1, text2, temp_results)  
      
    return SemanticDeltaResult(  
        surface_similarity=surface_sim,  
        semantic_similarity=semantic_sim,  
        structural_similarity=structural_sim,  
        lexical_diversity_delta=lexical_delta,  
        sentiment_delta=None,  # Could add sentiment analysis here  
        topic_coherence_delta=topic_delta,  
        sentence_level_deltas=sentence_deltas,  
        token_level_analysis=token_analysis,  
        confidence_score=confidence  
    )  
  
def _analyze_token_changes(self, text1: str, text2: str) -> Dict[str, float]:  
    """Analyze specific token-level changes."""  
    words1 = set(re.findall(r'\b\w+\b', text1.lower()))  
    words2 = set(re.findall(r'\b\w+\b', text2.lower()))  
      
    if not words1 and not words2:  
        return {"jaccard_similarity": 1.0, "added_tokens_ratio": 0.0, "removed_tokens_ratio": 0.0}  
      
    intersection = words1 & words2  
    union = words1 | words2  
      
    jaccard = len(intersection) / len(union) if union else 1.0  
    added_ratio = len(words2 - words1) / len(words2) if words2 else 0.0  
    removed_ratio = len(words1 - words2) / len(words1) if words1 else 0.0  
      
    return {  
        "jaccard_similarity": jaccard,  
        "added_tokens_ratio": added_ratio,  
        "removed_tokens_ratio": removed_ratio  
    }

def print_analysis_report(result: SemanticDeltaResult, text1_name: str = "Text 1", text2_name: str = "Text 2"):
"""Print a comprehensive analysis report."""
print(f"\n[Δ] Enhanced Semantic Delta Report: {text1_name} → {text2_name}")
print("=" * 60)
print(f"Composite Delta Score:     {result.composite_delta:.3f}")
print(f"Analysis Confidence:       {result.confidence_score:.3f}")
print()

print("Similarity Metrics:")  
print(f"  Surface Similarity:      {result.surface_similarity:.3f}")  
print(f"  Semantic Similarity:     {result.semantic_similarity:.3f}")  
print(f"  Structural Similarity:   {result.structural_similarity:.3f}")  
print()  
  
print("Delta Metrics:")  
print(f"  Lexical Diversity Δ:     {result.lexical_diversity_delta:.3f}")  
print(f"  Topic Coherence Δ:       {result.topic_coherence_delta:.3f}")  
print()  
  
print("Token Analysis:")  
for key, value in result.token_level_analysis.items():  
    print(f"  {key.replace('_', ' ').title()}: {value:.3f}")  
  
if result.sentence_level_deltas:  
    print(f"\nSentence-Level Analysis ({len(result.sentence_level_deltas)} alignments):")  
    print(f"  Average Sentence Similarity: {np.mean(result.sentence_level_deltas):.3f}")  
    print(f"  Std Dev:                     {np.std(result.sentence_level_deltas):.3f}")

Example usage with your existing integration

def compute_enhanced_deltas(gemma_output: str, omni_output: str) -> SemanticDeltaResult:
"""
Drop-in replacement for  original compute_deltas function.
"""
analyzer = EnhancedSemanticDelta()
return analyzer.analyze(gemma_output, omni_output)

Backward compatibility function

def compute_deltas(gemma_output: str, omni_output: str) -> Dict[str, float]:
"""
Backward compatible version that returns the same format as  original.
"""
result = compute_enhanced_deltas(gemma_output, omni_output)

return {  
    "surface_similarity": result.surface_similarity,  
    "semantic_similarity": result.semantic_similarity,  
    "surface_delta": 1 - result.surface_similarity,  
    "semantic_delta": 1 - result.semantic_similarity,  
    # Additional metrics  
    "structural_similarity": result.structural_similarity,  
    "composite_delta": result.composite_delta,  
    "confidence_score": result.confidence_score  
}

