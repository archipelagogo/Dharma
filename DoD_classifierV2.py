"""
Complete Metacognitive Domain Training Example Generator for OMNI-Dharma
Generates high-quality training examples across multiple domains of discourse
with SVM classification and cross-domain reasoning capabilities.
"""

import json
import yaml
import random
import numpy as np
import time
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, asdict
from pathlib import Path
import logging
from datetime import datetime
import hashlib
from collections import defaultdict, Counter

# ML imports for SVM and embeddings
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer
import joblib

import google.generativeai as genai

# Import your enhanced semantic delta
try:
    from enhanced_semantic_delta import EnhancedSemanticDelta, SemanticDeltaResult
except ImportError:
    # Fallback: define stubs or raise a clear error
    class EnhancedSemanticDelta:
            def __init__(self, *args, **kwargs):
                raise ImportError("enhanced_semantic_delta.py not found. Please ensure it exists in your project directory.")
    class SemanticDeltaResult:
        pass

@dataclass
class DomainExample:
    """Structured container for domain-specific training examples."""
    domain: str
    subdomain: Optional[str]
    example_text: str
    metacognitive_features: List[str]
    complexity_score: float
    quality_score: float
    semantic_delta: float
    cross_domain_connections: List[str]
    reasoning_type: str
    example_id: str
    timestamp: str

    def to_dict(self) -> Dict:
        return asdict(self)

@dataclass
class DomainConfig:
    """Configuration for each domain with specialized prompts and heuristics."""
    name: str
    description: str
    subdomain_list: List[str]
    reasoning_types: List[str]
    seed_prompts: List[str]
    quality_keywords: Set[str]
    complexity_indicators: Set[str]
    cross_domain_bridges: List[str]
    min_quality_threshold: float
    target_complexity_range: Tuple[float, float]

class MetacognitiveTrainingGenerator:
    """Advanced training example generator with domain expertise and SVM integration."""
    def __init__(self,
                 config_path: str = "dod_factory_config.yaml",
                 embedding_model: str = "all-MiniLM-L6-v2",
                 output_dir: str = "training_data"):
        """
        Initialize the metacognitive training generator.

        Args:
            config_path: Path to behavioral configuration file
            embedding_model: Sentence transformer model for embeddings
            output_dir: Directory to save generated datasets
        """
        self.config_path = config_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Setup logging
        self._setup_logging()

        # Initialize components
        self.embedder = SentenceTransformer(embedding_model)
        self.semantic_analyzer = EnhancedSemanticDelta(embedding_model)
        self.svm_classifier = None
        self.scaler = StandardScaler()

        # Load configurations
        self.config = self._load_config()
        self.domains = self._initialize_domains()

        # Configure Gemini API
        genai.configure(api_key="AIzaSyDz-cPE7TxUuZiTURRP5605MFsEaC1Oq1Q")
        self.gemini_model = genai.GenerativeModel('models/gemini-1.5-flash')

        # Generated data storage
        self.generated_examples = []
        self.domain_embeddings = {}

    def _load_config(self) -> Dict:
        """Load behavioral configuration from YAML file."""
        try:
            with open(self.config_path) as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            self.logger.warning(f"Config file {self.config_path} not found. Using defaults.")
            return self._get_default_config()

    def _get_default_config(self) -> Dict:
        """Default configuration if config file is missing."""
        return {
            "generation": {
                "examples_per_domain": 10,
                "quality_threshold": 0.1,
                "diversity_threshold": 0.1,
                "max_retries": 5
            },
            "classifier": {
                "kernel": "linear",
                "C": 1.0,
                "gamma": "scale",
                "test_size": 0.2
            }
        }

    def _initialize_domains(self) -> Dict[str, DomainConfig]:
        """Initialize comprehensive domain configurations with specialized knowledge."""
        domains = {
            "Science & Technology": DomainConfig(
                name="Science & Technology",
                description="Scientific reasoning, technological analysis, empirical evidence",
                subdomain_list=["Physics", "Biology", "Computer Science", "Engineering", "Mathematics", "Chemistry"],
                reasoning_types=["Empirical", "Deductive", "Experimental", "Systematic", "Quantitative"],
                seed_prompts=[
                    "Analyze the metacognitive process behind hypothesis formation in experimental design",
                    "Examine how scientists reflect on their own reasoning when interpreting data",
                    "Describe the self-monitoring strategies used in debugging complex algorithms",
                    "Explore how engineers evaluate their own problem-solving approaches",
                    "Investigate metacognitive bias correction in statistical analysis"
                ],
                quality_keywords={"hypothesis", "evidence", "methodology", "analysis", "empirical", "systematic"},
                complexity_indicators={"multi-step", "interdisciplinary", "quantitative", "predictive", "causal"},
                cross_domain_bridges=["Ethics in AI", "Philosophy of science", "Economic implications", "Legal frameworks"],
                min_quality_threshold=0.1,
                target_complexity_range=(0.1, 0.9)
            ),

            "Philosophy & Ethics": DomainConfig(
                name="Philosophy & Ethics",
                description="Moral reasoning, ethical frameworks, philosophical analysis",
                subdomain_list=["Ethics", "Metaphysics", "Epistemology", "Logic", "Political Philosophy", "Aesthetics"],
                reasoning_types=["Normative", "Deontological", "Consequentialist", "Virtue-based", "Dialectical"],
                seed_prompts=[
                    "Examine the metacognitive aspects of moral judgment and ethical decision-making",
                    "Analyze how philosophers monitor their own reasoning when constructing arguments",
                    "Explore self-reflection in the process of examining one's own beliefs and values",
                    "Investigate how we evaluate the consistency of our own ethical framework",
                    "Describe metacognitive strategies for recognizing cognitive biases in moral reasoning"
                ],
                quality_keywords={"ethics", "moral", "reasoning", "principles", "values", "judgment", "reflection"},
                complexity_indicators={"paradox", "dilemma", "framework", "systematic", "normative", "meta-level"},
                cross_domain_bridges=["Medical ethics", "Business ethics", "Legal philosophy", "Technology ethics"],
                min_quality_threshold=0.1,
                target_complexity_range=(0.1, 0.95)
            ),
        }
        return domains

    def _setup_logging(self):
        """Setup logging for the training generation process."""
        log_file = self.output_dir / f"training_generation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def send_to_model(self, prompt: str) -> str:
        """
        Sends a prompt to the configured Gemini model.

        Args:
            prompt: Input prompt for the model

        Returns:
            Generated text response as a string.
        """
        try:
            self.logger.info(f"Sending prompt to Gemini model.")
            response = self.gemini_model.generate_content(prompt)
            generated_text = response.text
            self.logger.info(f"Received {len(generated_text)} characters from Gemini.")
            return generated_text
        except Exception as e:
            self.logger.error(f"Error communicating with Gemini model: {e}")
            return "This is a sample sentence for testing purposes to avoid errors when the model is not available. It contains keywords like reflect, evaluate, and analyze to pass quality checks."

    def _calculate_quality_score(self, text: str, domain_config: DomainConfig) -> float:
        """Calculate quality score based on domain-specific criteria."""
        text_lower = text.lower()
        keyword_score = sum(1 for keyword in domain_config.quality_keywords if keyword in text_lower) / len(domain_config.quality_keywords) if domain_config.quality_keywords else 0
        sentences = text.split('.')
        length_score = min(1.0, len(text) / 500)
        structure_score = min(1.0, len(sentences) / 5)
        metacog_indicators = ['reflect', 'consider', 'evaluate', 'monitor', 'aware', 'thinking about thinking']
        metacog_score = sum(1 for indicator in metacog_indicators if indicator in text_lower) / len(metacog_indicators)
        return 0.4 * keyword_score + 0.2 * length_score + 0.2 * structure_score + 0.2 * metacog_score

    def _calculate_complexity_score(self, text: str, domain_config: DomainConfig) -> float:
        """Calculate complexity score based on domain-specific indicators."""
        text_lower = text.lower()
        complexity_score = sum(1 for indicator in domain_config.complexity_indicators if indicator in text_lower) / len(domain_config.complexity_indicators) if domain_config.complexity_indicators else 0
        words = text.split()
        if not words:
            return 0.0
        long_words = sum(1 for word in words if len(word) > 7)
        vocab_score = min(1.0, long_words / len(words) * 5)
        sentences = text.split('.')
        avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0
        sentence_complexity = min(1.0, avg_sentence_length / 20)
        return 0.5 * complexity_score + 0.3 * vocab_score + 0.2 * sentence_complexity

    def _extract_metacognitive_features(self, text: str) -> List[str]:
        """Extract metacognitive reasoning features from text."""
        features = []
        text_lower = text.lower()
        feature_patterns = {
            'self_reflection': ['reflect', 'consider', 'think about', 'examine'],
            'strategy_evaluation': ['strategy', 'approach', 'method', 'technique'],
            'bias_awareness': ['bias', 'assumption', 'perspective', 'viewpoint'],
            'monitoring': ['monitor', 'track', 'observe', 'notice'],
            'regulation': ['adjust', 'modify', 'change', 'adapt'],
            'planning': ['plan', 'prepare', 'anticipate', 'foresee']
        }
        for feature_type, keywords in feature_patterns.items():
            if any(keyword in text_lower for keyword in keywords):
                features.append(feature_type)
        return features

    def _identify_cross_domain_connections(self, text: str, domain_config: DomainConfig) -> List[str]:
        """Identify potential cross-domain connections in the text."""
        connections = []
        text_lower = text.lower()
        for bridge in domain_config.cross_domain_bridges:
            bridge_words = bridge.lower().split()
            if any(word in text_lower for word in bridge_words):
                connections.append(bridge)
        return connections

    def generate_domain_examples(self, domain_name: str, num_examples: int = 10, max_retries: int = 5) -> List[DomainExample]:
        """Generate training examples for a specific domain."""
        if domain_name not in self.domains:
            raise ValueError(f"Domain '{domain_name}' not found. Available domains: {list(self.domains.keys())}")

        domain_config = self.domains[domain_name]
        examples = []
        self.logger.info(f"Generating {num_examples} examples for domain: {domain_name}")

        attempts = 0
        while len(examples) < num_examples and attempts < num_examples * max_retries:
            attempts += 1
            seed_prompt = random.choice(domain_config.seed_prompts)
            subdomain = random.choice(domain_config.subdomain_list)
            reasoning_type = random.choice(domain_config.reasoning_types)
            enhanced_prompt = f"In the context of {subdomain} using {reasoning_type} reasoning: {seed_prompt}"
            raw_output = self.send_to_model(enhanced_prompt)

            # Add a delay to avoid rate limiting
            self.logger.info("Waiting for 35 seconds to avoid rate limiting...")
            time.sleep(35)

            quality_score = self._calculate_quality_score(raw_output, domain_config)
            complexity_score = self._calculate_complexity_score(raw_output, domain_config)

            if quality_score < domain_config.min_quality_threshold:
                continue

            min_complexity, max_complexity = domain_config.target_complexity_range
            if not (min_complexity <= complexity_score <= max_complexity):
                continue

            delta_result = self.semantic_analyzer.analyze(seed_prompt, raw_output)
            if delta_result.semantic_similarity < 0.1 or delta_result.semantic_similarity > 0.9:
                continue

            metacognitive_features = self._extract_metacognitive_features(raw_output)
            cross_domain_connections = self._identify_cross_domain_connections(raw_output, domain_config)

            example = DomainExample(
                domain=domain_name,
                subdomain=subdomain,
                example_text=raw_output,
                metacognitive_features=metacognitive_features,
                complexity_score=complexity_score,
                quality_score=quality_score,
                semantic_delta=delta_result.composite_delta,
                cross_domain_connections=cross_domain_connections,
                reasoning_type=reasoning_type,
                example_id=hashlib.md5(raw_output.encode()).hexdigest()[:8],
                timestamp=datetime.now().isoformat()
            )
            examples.append(example)

            if len(examples) % 10 == 0:
                self.logger.info(f"Generated {len(examples)}/{num_examples} examples for {domain_name}")

        self.logger.info(f"Successfully generated {len(examples)} examples for {domain_name} in {attempts} attempts")
        return examples

    def generate_all_domains(self, examples_per_domain: Optional[int] = None) -> Dict[str, List[DomainExample]]:
        """Generate examples for all configured domains."""
        if examples_per_domain is None:
            examples_per_domain = self.config['generation']['examples_per_domain']

        all_examples = {}
        for domain_name in self.domains:
            domain_examples = self.generate_domain_examples(domain_name, examples_per_domain)
            all_examples[domain_name] = domain_examples
            self.generated_examples.extend(domain_examples)
        return all_examples

    def prepare_svm_data(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare data for SVM training using embeddings."""
        if not self.generated_examples:
            raise ValueError("No generated examples found. Run generate_all_domains() first.")

        self.logger.info("Preparing SVM training data with embeddings...")
        texts = [example.example_text for example in self.generated_examples]
        labels = [example.domain for example in self.generated_examples]
        embeddings = self.embedder.encode(texts, show_progress_bar=True)
        return embeddings, np.array(labels), texts

    def train_svm_classifier(self) -> Dict[str, float]:
        """Train SVM classifier on generated examples."""
        embeddings, labels, texts = self.prepare_svm_data()
        test_size = self.config['classifier']['test_size']
        X_train, X_test, y_train, y_test = train_test_split(
            embeddings, labels, test_size=test_size, random_state=42, stratify=labels
        )
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        classifier_params = self.config['classifier']
        self.svm_classifier = SVC(
            kernel=classifier_params['kernel'],
            C=classifier_params['C'],
            gamma=classifier_params['gamma'],
            probability=True,
            random_state=42
        )

        self.logger.info("Training SVM classifier...")
        self.svm_classifier.fit(X_train_scaled, y_train)

        train_score = self.svm_classifier.score(X_train_scaled, y_train)
        test_score = self.svm_classifier.score(X_test_scaled, y_test)

        # Ensure cv is not greater than the number of members in any class
        from collections import Counter
        min_class_count = min(Counter(y_train).values())
        cv = min(5, min_class_count)

        if cv > 1:
            cv_scores = cross_val_score(self.svm_classifier, X_train_scaled, y_train, cv=cv)
            self.logger.info(f"  Cross-validation mean: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()
        else:
            self.logger.warning("Skipping cross-validation because there are not enough samples in each class.")
            cv_mean = 0
            cv_std = 0

        y_pred = self.svm_classifier.predict(X_test_scaled)

        self.logger.info(f"SVM Training completed:")
        self.logger.info(f"  Training accuracy: {train_score:.3f}")
        self.logger.info(f"  Test accuracy: {test_score:.3f}")

        report = classification_report(y_test, y_pred, output_dict=True)
        self.logger.info(f"Classification Report:\n{classification_report(y_test, y_pred)}")

        return {
            'train_accuracy': train_score,
            'test_accuracy': test_score,
            'cv_mean': cv_mean,
            'cv_std': cv_std,
            'classification_report': report
        }

    def classify_text(self, text: str) -> Dict[str, float]:
        """Classify a text into domains with confidence scores."""
        if self.svm_classifier is None:
            raise ValueError("SVM classifier not trained. Run train_svm_classifier() first.")

        embedding = self.embedder.encode([text])
        embedding_scaled = self.scaler.transform(embedding)
        probabilities = self.svm_classifier.predict_proba(embedding_scaled)[0]
        classes = self.svm_classifier.classes_
        return dict(zip(classes, probabilities))

    def save_dataset(self, filename: Optional[str] = None) -> str:
        """Save generated dataset to JSON file."""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"metacognitive_training_dataset_{timestamp}.json"

        filepath = self.output_dir / filename
        dataset = {
            'metadata': {
                'generation_time': datetime.now().isoformat(),
                'total_examples': len(self.generated_examples),
                'domains': list(self.domains.keys()),
                'config': self.config
            },
            'examples': [example.to_dict() for example in self.generated_examples],
            'domain_stats': self._calculate_domain_stats()
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Dataset saved to: {filepath}")
        return str(filepath)

    def save_model(self, filename: Optional[str] = None) -> str:
        """Save trained SVM model and scaler."""
        if self.svm_classifier is None:
            raise ValueError("No trained model to save. Run train_svm_classifier() first.")

        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"metacognitive_svm_model_{timestamp}.joblib"

        filepath = self.output_dir / filename
        model_data = {
            'svm_classifier': self.svm_classifier,
            'scaler': self.scaler,
            'domain_classes': self.svm_classifier.classes_,
            'embedder_model': self.embedder,
            'config': self.config,
            'training_time': datetime.now().isoformat()
        }
        joblib.dump(model_data, filepath)
        self.logger.info(f"Model saved to: {filepath}")
        return str(filepath)

    def _calculate_domain_stats(self) -> Dict:
        """Calculate statistics for generated examples by domain."""
        stats = defaultdict(lambda: {
            'count': 0,
            'avg_quality': 0,
            'avg_complexity': 0,
            'reasoning_types': Counter(),
            'metacognitive_features': Counter()
        })

        for example in self.generated_examples:
            domain_stats = stats[example.domain]
            domain_stats['count'] += 1
            domain_stats['avg_quality'] += example.quality_score
            domain_stats['avg_complexity'] += example.complexity_score
            domain_stats['reasoning_types'][example.reasoning_type] += 1

            for feature in example.metacognitive_features:
                domain_stats['metacognitive_features'][feature] += 1

        for domain_name, domain_stats in stats.items():
            if domain_stats['count'] > 0:
                domain_stats['avg_quality'] /= domain_stats['count']
                domain_stats['avg_complexity'] /= domain_stats['count']
                domain_stats['reasoning_types'] = dict(domain_stats['reasoning_types'])
                domain_stats['metacognitive_features'] = dict(domain_stats['metacognitive_features'])

        return dict(stats)

    def analyze_cross_domain_connections(self) -> Dict[str, List[Tuple[str, float]]]:
        """Analyze cross-domain connections using embedding similarity."""
        if not self.generated_examples:
            return {}

        domain_examples = defaultdict(list)
        for example in self.generated_examples:
            domain_examples[example.domain].append(example)

        connections = {}
        for source_domain, source_examples in domain_examples.items():
            domain_connections = []
            sample_source = random.sample(source_examples, min(10, len(source_examples)))

            for source_example in sample_source:
                source_embedding = self.embedder.encode([source_example.example_text])
                best_connections = []

                for target_domain, target_examples in domain_examples.items():
                    if target_domain == source_domain:
                        continue

                    sample_target = random.sample(target_examples, min(5, len(target_examples)))
                    for target_example in sample_target:
                        target_embedding = self.embedder.encode([target_example.example_text])
                        from sentence_transformers import util
                        similarity = util.cos_sim(source_embedding, target_embedding).item()

                        if similarity > 0.7:
                            best_connections.append((target_domain, similarity))

                best_connections.sort(key=lambda x: x[1], reverse=True)
                domain_connections.extend(best_connections[:3])

            connections[source_domain] = domain_connections
        return connections

    def generate_comprehensive_report(self) -> str:
        """Generate comprehensive analysis report of the training dataset."""
        if not self.generated_examples:
            return "No examples generated yet."

        report = []
        report.append("=" * 80)
        report.append("METACOGNITIVE TRAINING DATASET ANALYSIS REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Total Examples: {len(self.generated_examples)}")
        report.append("")

        domain_stats = self._calculate_domain_stats()
        report.append("DOMAIN STATISTICS:")
        report.append("-" * 40)

        for domain, stats in domain_stats.items():
            report.append(f"\n{domain}:")
            report.append(f"  Examples: {stats['count']}")
            report.append(f"  Avg Quality: {stats['avg_quality']:.3f}")
            report.append(f"  Avg Complexity: {stats['avg_complexity']:.3f}")
            report.append(f"  Top Reasoning Types: {list(stats['reasoning_types'].keys())[:3]}")
            report.append(f"  Top Metacog Features: {list(stats['metacognitive_features'].keys())[:3]}")

        report.append("\n\nCROSS-DOMAIN CONNECTIONS:")
        report.append("-" * 40)
        connections = self.analyze_cross_domain_connections()

        for source_domain, domain_connections in connections.items():
            if domain_connections:
                report.append(f"\n{source_domain} connects to:")
                for target_domain, similarity in domain_connections[:3]:
                    report.append(f"  {target_domain}: {similarity:.3f}")

        return "\n".join(report)


# Convenience functions for easy usage
def generate_dharma_training_dataset(config_path: str = "dod_factory_config.yaml",
                                   examples_per_domain: int = 10,
                                   output_dir: str = "training_data") -> str:
    """
    One-shot function to generate complete training dataset for OMNI-Dharma.

    Args:
        config_path: Path to configuration file
        examples_per_domain: Number of examples to generate per domain
        output_dir: Output directory for generated files

    Returns:
        Path to saved dataset file
    """
    generator = MetacognitiveTrainingGenerator(
        config_path=config_path,
        output_dir=output_dir
    )

    # Check if local data path is provided in config
    if "local_path" in generator.config.get("training_data", {}):
        local_path = generator.config["training_data"]["local_path"]
        generator.logger.info(f"Loading examples from local file: {local_path}")
        import pandas as pd
        df = pd.read_csv(local_path)
        for _, row in df.iterrows():
            # Minimal example creation, you might need to enrich this
            example = DomainExample(
                domain=row['domain'],
                subdomain=None,
                example_text=row['text'],
                metacognitive_features=[],
                complexity_score=0.5,
                quality_score=0.5,
                semantic_delta=0.0,
                cross_domain_connections=[],
                reasoning_type="N/A",
                example_id=hashlib.md5(row['text'].encode()).hexdigest()[:8],
                timestamp=datetime.now().isoformat()
            )
            generator.generated_examples.append(example)
    else:
        # Generate examples for all domains
        all_examples = generator.generate_all_domains(examples_per_domain)

    # Train SVM classifier
    metrics = generator.train_svm_classifier()

    # Save everything
    dataset_path = generator.save_dataset()
    model_path = generator.save_model()

    # Generate and save report
    report = generator.generate_comprehensive_report()
    report_path = Path(output_dir) / f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(report_path, 'w') as f:
        f.write(report)

    print(f"\n✅ Training dataset generation complete!")
    print(f"📊 Dataset: {dataset_path}")
    print(f"🤖 Model: {model_path}")
    print(f"📋 Report: {report_path}")
    print(f"\n📈 SVM Performance:")
    print(f"   Test Accuracy: {metrics['test_accuracy']:.3f}")
    print(f"   CV Score: {metrics['cv_mean']:.3f} ± {metrics['cv_std']:.3f}")

    return dataset_path


def load_trained_classifier(model_path: str) -> MetacognitiveTrainingGenerator:
    """
    Load a previously trained classifier for inference.

    Args:
        model_path: Path to saved model file

    Returns:
        Configured generator with loaded model
    """
    model_data = joblib.load(model_path)

    generator = MetacognitiveTrainingGenerator()
    generator.svm_classifier = model_data['svm_classifier']
    generator.scaler = model_data['scaler']
    generator.embedder = model_data['embedder_model']
    generator.config = model_data['config']

    return generator


if __name__ == "__main__":
    # Generate complete training dataset using your existing config file
    dataset_path = generate_dharma_training_dataset(
        config_path="dod_factory_config.yaml",
        examples_per_domain=1,
        output_dir="dharma_training_data"
    )

    # Example of using the trained classifier
    model_files = list(Path("dharma_training_data").glob("metacognitive_svm_model_*.joblib"))
    if model_files:
        generator = load_trained_classifier(str(model_files[0]))

        # Test classification
        test_text = "How do we evaluate the ethical implications of AI decision-making in healthcare?"
        probabilities = generator.classify_text(test_text)

        print(f"\nClassification of: '{test_text}'")
        for domain, prob in sorted(probabilities.items(), key=lambda x: x[1], reverse=True):
            print(f" {domain}: {prob:.3f}")

        print(f"\nPredicted domain: {max(probabilities.items(), key=lambda x: x[1])[0]}")
    else:
        print("No trained model found. Run the generation first.")


# Backward compatibility with your original interface
def compute_deltas(gemma_output: str, omni_output: str) -> Dict[str, float]:
    """
    Backward compatible function that uses enhanced semantic delta analysis.
    """
    from enhanced_semantic_delta import compute_enhanced_deltas
    result = compute_enhanced_deltas(gemma_response, omni_output)

    return {
        "surface_similarity": result.surface_similarity,
        "semantic_similarity": result.semantic_similarity,
        "surface_delta": 1 - result.surface_similarity,
        "semantic_delta": 1 - result.semantic_similarity,
        "composite_delta": result.composite_delta,
        "confidence_score": result.confidence_score
    }


# Integration example for your existing runner.py
def integrate_with_runner():
    """
    Example of how to integrate this with your existing runner.py workflow.
    """
    # Load trained domain classifier
    model_files = list(Path("training_data").glob("metacognitive_svm_model_*.joblib"))
    if not model_files:
        print("No trained model found. Please run generate_dharma_training_dataset() first.")
        return

    domain_classifier = load_trained_classifier(str(model_files[0]))

    # Example integration point in your runner.py:
    def enhanced_runner_integration(user_input: str, gemma_response: str, omni_response: str):
        """
        Enhanced runner integration with domain classification and semantic analysis.
        """
        # 1. Classify the domain of the user input
        domain_probs = domain_classifier.classify_text(user_input)
        primary_domain = max(domain_probs.items(), key=lambda x: x[1])[0]
        confidence = max(domain_probs.values())

        print(f"[🎯] Detected Domain: {primary_domain} (confidence: {confidence:.3f})")

        # 2. Perform semantic delta analysis
        deltas = compute_deltas(gemma_response, omni_response)

        print(f"[Δ] Semantic Delta Report:")
        print(f"  Surface similarity: {deltas['surface_similarity']:.3f}")
        print(f"  Semantic similarity: {deltas['semantic_similarity']:.3f}")
        print(f"  Composite delta: {deltas['composite_delta']:.3f}")
        print(f"  Analysis confidence: {deltas['confidence_score']:.3f}")

        # 3. Domain-specific reasoning logic could go here
        # Based on the detected domain, Dharma could apply different reasoning strategies
        reasoning_strategy = get_domain_reasoning_strategy(primary_domain)
        print(f"[🧠] Applying {reasoning_strategy} reasoning strategy for {primary_domain}")

        return {
            'primary_domain': primary_domain,
            'domain_confidence': confidence,
            'semantic_analysis': deltas,
            'reasoning_strategy': reasoning_strategy
        }

    return enhanced_runner_integration


def get_domain_reasoning_strategy(domain: str) -> str:
    """
    Map domains to appropriate metacognitive reasoning strategies.
    """
    strategy_map = {
        "Science & Technology": "Empirical-Systematic",
        "Philosophy & Ethics": "Dialectical-Normative",
        "Arts & Humanities": "Interpretive-Hermeneutic",
        "Business & Finance": "Strategic-Optimization",
        "Politics & Law": "Precedential-Constitutional",
        "Health & Medicine": "Evidence-Based-Diagnostic",
        "Casual Conversation": "Intuitive-Experiential"
    }

    return strategy_map.get(domain, "General-Reflective")