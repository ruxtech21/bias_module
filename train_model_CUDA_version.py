import pandas as pd
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification, get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from typing import Dict, List, Tuple, Optional
import os
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.nn.functional as F

class BiasDataset(Dataset):
    def __init__(self, csv_file, tokenizer_name='roberta-base', max_length=256):  # Increased max_length
        df = pd.read_csv(csv_file)
        self.texts = df['comment_text'].tolist()
        self.labels = df['label'].tolist()
        self.tokenizer = RobertaTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        
        # Improved text preprocessing
        processed_texts = [self._preprocess_text(text) for text in self.texts]
        
        self.encodings = self.tokenizer(
            processed_texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors=None
        )

    def _preprocess_text(self, text: str) -> str:
        # Basic text cleaning
        text = text.strip()
        text = ' '.join(text.split())  # Normalize whitespace
        return text

    def __getitem__(self, idx):
        item = {
            key: torch.tensor(val[idx], dtype=torch.long)
            for key, val in self.encodings.items()
        }
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item

    def __len__(self):
        return len(self.labels)

class EnhancedBiasDetector(nn.Module):
    def __init__(self, num_labels=1, dropout_rate=0.2):  # Reduced dropout
        super().__init__()
        self.roberta = RobertaForSequenceClassification.from_pretrained(
            'roberta-base',
            num_labels=num_labels,
            hidden_dropout_prob=dropout_rate,
            attention_probs_dropout_prob=dropout_rate
        )
        
        # More complex classification head
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dense1 = nn.Linear(768, 384)
        self.layer_norm1 = nn.LayerNorm(384)
        self.dense2 = nn.Linear(384, 128)
        self.layer_norm2 = nn.LayerNorm(128)
        self.classifier = nn.Linear(128, num_labels)
        
    def forward(self, **inputs):
        labels = inputs.pop('labels', None)
        
        # Get the base model outputs
        outputs = self.roberta.roberta(**inputs)
        sequence_output = outputs.last_hidden_state[:, 0, :]
        
        # More sophisticated feature processing
        x = self.dropout1(sequence_output)
        x = self.dense1(x)
        x = self.layer_norm1(x)
        x = F.gelu(x)
        x = self.dense2(x)
        x = self.layer_norm2(x)
        x = F.gelu(x)
        logits = self.classifier(x)
        logits = logits.squeeze(-1)
        
        if labels is not None:
            # Focal Loss implementation for better handling of class imbalance
            BCE_loss = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')
            pt = torch.exp(-BCE_loss)
            focal_loss = (1 - pt) ** 2 * BCE_loss
            loss = focal_loss.mean()
            return loss, logits
        return logits

def train_model(dataset_path: str, output_model_path: str, epochs: int = 3, batch_size: int = 32):
    """Train the bias detection model with enhanced training process."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Load and preprocess dataset
    df = pd.read_csv(dataset_path)
    
    # Calculate class weights
    pos_weight = (len(df[df['label'] == 0]) / len(df[df['label'] == 1]))
    print(f"Positive class weight: {pos_weight:.2f}")
    
    # Initialize dataset
    full_dataset = BiasDataset(dataset_path, max_length=256)
    
    # Stratified split to maintain class distribution
    train_idx, val_idx = train_test_split(
        range(len(full_dataset)),
        test_size=0.2,
        stratify=df['label'],
        random_state=42
    )
    
    train_dataset = torch.utils.data.Subset(full_dataset, train_idx)
    val_dataset = torch.utils.data.Subset(full_dataset, val_idx)
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True
    )
    
    model = EnhancedBiasDetector().to(device)
    
    # Optimizer with cyclic learning rate
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=2e-5,
        epochs=epochs,
        steps_per_epoch=len(train_dataloader),
        pct_start=0.1
    )
    
    best_val_f1 = 0.0
    best_val_loss = float('inf')
    patience = 2
    patience_counter = 0
    
    print(f"\nStarting training for {epochs} epochs...")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        train_predictions = []
        train_labels = []
        
        for batch_idx, batch in enumerate(train_dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            
            loss, logits = model(**batch)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            total_train_loss += loss.item()
            
            predictions = (torch.sigmoid(logits) > 0.5).int()
            train_predictions.extend(predictions.cpu().numpy())
            train_labels.extend(batch['labels'].cpu().numpy())
            
            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs} | Batch {batch_idx + 1}/{len(train_dataloader)} | Loss: {loss.item():.4f}")
        
        model.eval()
        total_val_loss = 0
        val_predictions = []
        val_labels = []
        
        with torch.no_grad():
            for batch in val_dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                loss, logits = model(**batch)
                total_val_loss += loss.item()
                
                predictions = (torch.sigmoid(logits) > 0.5).int()
                val_predictions.extend(predictions.cpu().numpy())
                val_labels.extend(batch['labels'].cpu().numpy())
        
        train_f1 = calculate_f1(train_predictions, train_labels)
        val_f1 = calculate_f1(val_predictions, val_labels)
        avg_val_loss = total_val_loss / len(val_dataloader)
        
        print(f"\nEpoch {epoch + 1}/{epochs}")
        print(f"Train Loss: {total_train_loss/len(train_dataloader):.4f}, F1: {train_f1:.4f}")
        print(f"Val Loss: {avg_val_loss:.4f}, F1: {val_f1:.4f}")
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), output_model_path)
            print(f"New best model saved with validation F1: {best_val_f1:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break
    
    print(f"\nTraining completed.")
    print(f"Best validation F1: {best_val_f1:.4f}")
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def calculate_f1(predictions, labels):
    predictions = np.array(predictions)
    labels = np.array(labels)
    
    true_positives = np.sum((predictions == 1) & (labels == 1))
    false_positives = np.sum((predictions == 1) & (labels == 0))
    false_negatives = np.sum((predictions == 0) & (labels == 1))
    
    precision = true_positives / (true_positives + false_positives + 1e-10)
    recall = true_positives / (true_positives + false_negatives + 1e-10)
    
    f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
    return f1

class BiasDetector:
    def __init__(self, model_path: Optional[str] = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        self.model = EnhancedBiasDetector()
        if model_path:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        self.bias_threshold = 0.33  # Further lowered threshold

        # Enhanced racial bias patterns
        self.explicit_racial_terms = [
            'race', 'racial', 'racist', 'ethnic', 'minority', 'minorities',
            'black', 'white', 'asian', 'latino', 'hispanic', 'immigrant',
            'foreigner', 'illegal', 'demographic', 'demographics',
            'jews', 'jewish', 'muslims', 'arabs', 'middle eastern',  # Added more explicit groups
            'british', 'mexican', 'japanese', 'korean', 'vietnamese',
            'thai', 'cambodian', 'filipino', 'haitian', 'guatemalan',
            'nicaraguan', 'cuban'  # Added nationalities often subject to bias
        ]
        
        self.coded_racial_terms = [
            'certain neighborhoods', 'urban', 'inner city',
            'those areas', 'that part of town', 'property values',
            'good neighborhood', 'bad neighborhood', 'safe neighborhood',
            'unsafe area', 'dangerous area', 'culture fit', 'work ethic',
            'changing demographics', 'neighborhood character',
            'traditional values', 'community standards', 'declining standards',
            'cultural values', 'preserve character', 'maintain standards',
            'stable neighborhood', 'property maintenance', 'different values',
            'community character', 'cultural differences', 'way of life',
            'welfare', 'taking our jobs', 'stealing our jobs',  # Added economic stereotypes
            'living off the system', 'burden on society',
            'not paying taxes', 'draining resources'
        ]
        
        self.racial_stereotypes = [
            'they all', 'these people', 'those people', 'their kind',
            'you people', 'that background', 'that community',
            'typical of them', 'like them', 'naturally', 'genetics',
            'their culture', 'their values', 'their way of life',
            'cognitive abilities', 'inherent traits', 'genetic factors',
            'biological differences', 'natural abilities', 'aptitudes',
            'intelligence differences', 'genetic predisposition',
            'evolutionary advantages', 'inherited traits',
            'they always', 'they never', 'all of them',  # Added generalizing terms
            'that type', 'that sort'
        ]

        # Pseudo-scientific racism patterns
        self.pseudo_scientific_terms = [
            'studies show', 'research indicates', 'data suggests',
            'statistics prove', 'scientific evidence', 'measured differences',
            'cognitive development', 'genetic basis', 'evolutionary',
            'biological basis', 'hereditary factors', 'innate abilities',
            'natural selection', 'adaptation', 'group differences',
            'population genetics', 'behavioral traits', 'genetic markers',
            'statistical correlation', 'empirical evidence'
        ]

        # Property and economic dog whistles
        self.property_economic_terms = [
            'property values', 'home prices', 'neighborhood quality',
            'community standards', 'investment risk', 'market rates',
            'property maintenance', 'rental standards', 'housing prices',
            'neighborhood stability', 'economic impact', 'tax base',
            'school quality', 'crime rates', 'public services',
            'community investment', 'property rights', 'market value'
        ]

        # Cultural supremacy patterns
        self.cultural_supremacy_terms = [
            'western values', 'civilized society', 'advanced culture',
            'developed nations', 'modern standards', 'cultural norms',
            'social values', 'traditional values', 'moral standards',
            'cultural heritage', 'societal norms', 'cultural identity',
            'cultural preservation', 'cultural integrity', 'social order'
        ]
        
        # Contextual patterns with more sophisticated combinations
        self.contextual_pairs = [
            # Property-related patterns
            ('property', ['values', 'prices', 'neighborhood', 'declining', 'standards', 'maintenance']),
            ('neighborhood', ['character', 'quality', 'changing', 'stable', 'traditional', 'standards']),
            ('community', ['values', 'standards', 'character', 'traditional', 'changing', 'different']),
            
            # Cultural patterns
            ('cultural', ['values', 'differences', 'background', 'norms', 'identity', 'heritage']),
            ('traditional', ['values', 'neighborhood', 'community', 'standards', 'way of life']),
            ('different', ['values', 'culture', 'standards', 'background', 'way of life']),
            
            # Scientific-sounding patterns
            ('studies', ['show', 'indicate', 'suggest', 'prove', 'demonstrate']),
            ('research', ['indicates', 'suggests', 'shows', 'demonstrates']),
            ('data', ['shows', 'indicates', 'suggests', 'demonstrates']),
            ('genetic', ['factors', 'differences', 'predisposition', 'traits']),
            ('cognitive', ['abilities', 'differences', 'traits', 'strengths']),
            
            # Demographic patterns
            ('demographic', ['changes', 'shift', 'trends', 'makeup']),
            ('population', ['changes', 'shift', 'makeup', 'characteristics']),
            
            # Economic patterns
            ('economic', ['impact', 'effects', 'consequences', 'burden']),
            ('property', ['values', 'prices', 'investments', 'standards'])
        ]

        # Add historical context patterns
        self.historical_bias_terms = [
            'civil war', 'world war', 'rebel', 'confederate', 'axis',
            'hate slaves', 'hated slaves', 'hated jews',
            'never met', 'never known', 'sold a young fool',
            'war incorporates', 'getting attention',
            'percentage of', 'truly hated', 'truly hate'
        ]

        # Add analytical bias patterns (for detecting bias in academic/analytical language)
        self.analytical_bias_terms = [
            'economic in nature', 'argument can be made',
            'percentage of', 'what percentage',
            'getting tired of', 'list of', 'pick from the list',
            'if you go to war', 'because you hate',
            'someone you\'ve never', 'rarely just about'
        ]

    def _check_contextual_patterns(self, text: str) -> Tuple[float, List[str]]:
        """Enhanced check for contextual patterns that suggest racial bias."""
        words = text.lower().split()
        score = 0.0
        patterns_found = []
        
        # Check word pairs in context with larger window
        for base_word, context_words in self.contextual_pairs:
            if base_word in words:
                idx = words.index(base_word)
                # Expanded context window
                window = words[max(0, idx-5):min(len(words), idx+6)]
                matches = [w for w in context_words if w in window]
                if matches:
                    score += 0.15 * len(matches)  # Increased score for multiple matches
                    patterns_found.append(f"{base_word}-{matches[0]}")

        # Check for pseudo-scientific language patterns
        pseudo_scientific_count = sum(term in text.lower() for term in self.pseudo_scientific_terms)
        if pseudo_scientific_count > 0:
            score += 0.2 * pseudo_scientific_count
            patterns_found.append("pseudo-scientific language")

        # Check for cultural supremacy patterns
        cultural_supremacy_count = sum(term in text.lower() for term in self.cultural_supremacy_terms)
        if cultural_supremacy_count > 0:
            score += 0.2 * cultural_supremacy_count
            patterns_found.append("cultural supremacy language")

        # Check for property/economic dog whistles
        property_economic_count = sum(term in text.lower() for term in self.property_economic_terms)
        if property_economic_count > 0:
            score += 0.15 * property_economic_count
            patterns_found.append("property/economic dog whistles")

        # Additional check for combinations of different pattern types
        pattern_types = len(set([p.split(':')[0] for p in patterns_found]))
        if pattern_types > 1:
            score *= (1 + (0.1 * pattern_types))  # Boost score for multiple pattern types

        # Check for historical bias patterns
        historical_count = sum(term in text.lower() for term in self.historical_bias_terms)
        if historical_count > 0:
            score += 0.2 * historical_count
            patterns_found.append("historical bias context")

        # Check for analytical bias patterns
        analytical_count = sum(term in text.lower() for term in self.analytical_bias_terms)
        if analytical_count > 0:
            score += 0.15 * analytical_count
            patterns_found.append("analytical bias language")

        # Check for lists of ethnic/racial groups
        ethnic_group_count = sum(term in text.lower() for term in self.explicit_racial_terms)
        if ethnic_group_count >= 3:  # If multiple ethnic groups are mentioned
            score += 0.25  # Add significant score for listing multiple groups
            patterns_found.append("multiple ethnic group references")

        # Check for war-related context combined with ethnic groups
        war_terms = ['war', 'soldier', 'military', 'battle', 'fought', 'fighting']
        if any(term in text.lower() for term in war_terms) and ethnic_group_count > 0:
            score += 0.2
            patterns_found.append("war-ethnic context")

        return min(score, 1.0), patterns_found

    def detect_bias(self, text: str) -> Tuple[bool, float, Dict[str, float]]:
        # Preprocess text
        text = text.strip()
        text = ' '.join(text.split())
        text_lower = text.lower()
        
        # Get model prediction
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=256
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            logits = self.model(**inputs)
            model_score = torch.sigmoid(logits).squeeze().item()
        
        # Enhanced pattern detection with adjusted weights
        explicit_score = sum(term in text_lower for term in self.explicit_racial_terms) * 0.35
        coded_score = sum(term in text_lower for term in self.coded_racial_terms) * 0.30
        stereotype_score = sum(term in text_lower for term in self.racial_stereotypes) * 0.35
        
        # Get contextual patterns score with enhanced checks
        context_score, patterns_found = self._check_contextual_patterns(text_lower)
        
        # Special case: Check for combinations of stereotypes with economic terms
        economic_terms = ['jobs', 'welfare', 'taxes', 'system', 'burden', 'resources']
        stereotype_terms = ['those people', 'these people', 'they', 'them', 'their']
        
        # If we find both economic terms and stereotype terms, boost the score
        if any(term in text_lower for term in economic_terms) and any(term in text_lower for term in stereotype_terms):
            stereotype_score *= 1.5

        # Special case: Check for historical context combined with multiple ethnic groups
        if "historical bias context" in patterns_found and "multiple ethnic group references" in patterns_found:
            context_score *= 1.4  # Significant boost for historical bias with multiple groups
        
        # Calculate sophisticated bias score with adjusted weights
        sophisticated_score = (
            explicit_score +
            coded_score +
            stereotype_score +
            context_score
        )
        
        # Combine scores with emphasis on sophisticated patterns
        pattern_score = min(sophisticated_score, 1.0)
        
        # Adjust the weighting between model and pattern scores
        final_score = (0.35 * model_score) + (0.65 * pattern_score)  # Give even more weight to pattern detection
        
        # Additional boosts for sophisticated bias patterns
        if len(patterns_found) >= 2:
            final_score *= (1 + (0.15 * len(patterns_found)))
            final_score = min(final_score, 1.0)
        
        # Special boost for analytical language combined with bias patterns
        if "analytical bias language" in patterns_found and len(patterns_found) > 1:
            final_score *= 1.2
            final_score = min(final_score, 1.0)
        
        category_scores = {
            'race': final_score,
            'gender': 0.0,
            'religion': 0.0,
            'sexual_orientation': 0.0
        }
        
        return final_score > self.bias_threshold, final_score, category_scores

    def explain_bias(self, text: str) -> str:
        """Generate a detailed explanation for why the text might be biased."""
        text_lower = text.lower()
        is_biased, confidence, category_scores = self.detect_bias(text)
        
        if not is_biased:
            return "No significant bias detected in the text."
        
        # Identify specific patterns found
        patterns_found = []
        
        # Check for explicit terms
        if any(term in text_lower for term in self.explicit_racial_terms):
            patterns_found.append("explicit racial terms")
        
        # Check for coded language
        coded_matches = [term for term in self.coded_racial_terms if term in text_lower]
        if coded_matches:
            patterns_found.append("coded racial language")
        
        # Check for stereotypes
        if any(term in text_lower for term in self.racial_stereotypes):
            patterns_found.append("racial stereotyping")
        
        # Check for pseudo-scientific language
        if any(term in text_lower for term in self.pseudo_scientific_terms):
            patterns_found.append("pseudo-scientific racial bias")
        
        # Check for property/economic dog whistles
        if any(term in text_lower for term in self.property_economic_terms):
            patterns_found.append("economic-based racial bias")
        
        # Check for cultural supremacy language
        if any(term in text_lower for term in self.cultural_supremacy_terms):
            patterns_found.append("cultural supremacy bias")
        
        # Get contextual patterns
        _, context_patterns = self._check_contextual_patterns(text_lower)
        if context_patterns:
            patterns_found.extend(context_patterns)
        
        if confidence > 0.8:
            strength = "highly confident"
        elif confidence > 0.6:
            strength = "moderately confident"
        else:
            strength = "somewhat confident"
        
        explanation = f"The text shows signs of bias (confidence: {confidence:.2f}). "
        explanation += f"The model is {strength} this text contains bias. "
        
        if patterns_found:
            explanation += "Detected patterns include: " + ", ".join(patterns_found) + ". "
        
        if category_scores['race'] > 0:
            explanation += f"Racial bias score: {category_scores['race']:.2f}"
        
        return explanation

    def suggest_alternative(self, text: str) -> str:
        """Suggest a less biased alternative to the given text."""
        is_biased, _, category_scores = self.detect_bias(text)
        if not is_biased:
            return None
            
        # Identify the main bias category
        main_category = max(category_scores.items(), key=lambda x: x[1])[0]
        
        suggestions = {
            'race': "Consider focusing on individual characteristics rather than racial or ethnic groups.",
            'gender': "Try to use gender-neutral language and avoid gender stereotypes.",
            'religion': "Focus on individual beliefs and practices rather than making assumptions about religious groups.",
            'sexual_orientation': "Use inclusive language and avoid making assumptions about sexual orientation."
        }
        
        return suggestions.get(main_category, "Consider rephrasing to be more inclusive and avoid stereotypes.")

    def analyze_output(self, text: str) -> Dict:
        """Complete analysis of text for bias detection."""
        is_biased, confidence, category_scores = self.detect_bias(text)
        explanation = self.explain_bias(text)
        alternative = self.suggest_alternative(text) if is_biased else None
        
        return {
            "is_biased": is_biased,
            "confidence": confidence,
            "category_scores": category_scores,
            "explanation": explanation,
            "alternative": alternative
        }

def load_and_merge_datasets(train_path: str, private_test_path: str, all_data_path: str, toxicity_annotations_path: str) -> pd.DataFrame:
    """Load and merge multiple datasets with proper error handling."""
    print("\nLoading and merging datasets...")
    
    dfs = []
    
    # Load train dataset
    try:
        train_df = pd.read_csv(train_path)
        print(f"Loaded train dataset with {len(train_df)} rows")
        dfs.append(train_df)
    except Exception as e:
        print(f"Error loading train dataset: {str(e)}")
        return None

    # Load private test dataset
    try:
        private_test_df = pd.read_csv(private_test_path)
        print(f"Loaded private test dataset with {len(private_test_df)} rows")
        dfs.append(private_test_df)
    except Exception as e:
        print(f"Error loading private test dataset: {str(e)}")
    
    # Load all_data dataset
    try:
        all_data_df = pd.read_csv(all_data_path)
        print(f"Loaded all_data dataset with {len(all_data_df)} rows")
        dfs.append(all_data_df)
    except Exception as e:
        print(f"Error loading all_data dataset: {str(e)}")
    
    # Load toxicity annotations
    try:
        toxicity_df = pd.read_csv(toxicity_annotations_path)
        print(f"Loaded toxicity annotations with {len(toxicity_df)} rows")
        # Aggregate toxicity scores by comment_id
        toxicity_agg = toxicity_df.groupby('comment_id').agg({
            'toxic': 'mean',
            'severe_toxic': 'mean',
            'obscene': 'mean',
            'threat': 'mean',
            'insult': 'mean',
            'identity_hate': 'mean'
        }).reset_index()
        dfs.append(toxicity_agg)
    except Exception as e:
        print(f"Error loading toxicity annotations: {str(e)}")
    
    # Merge all dataframes
    print("\nMerging datasets...")
    merged_df = pd.concat(dfs, ignore_index=True)
    
    # Remove duplicates based on comment_text
    merged_df = merged_df.drop_duplicates(subset=['comment_text'], keep='first')
    print(f"Final merged dataset size: {len(merged_df)} rows")
    
    return merged_df

if __name__ == "__main__":
    # Dataset path
    train_path = r"C:\Users\ovidi\OneDrive\Desktop\Cursor (licenta Rux)\Jigsaw dataset\train.csv"
    
    # Load dataset
    print("\nLoading dataset...")
    try:
        df = pd.read_csv(train_path)
        print(f"Loaded train dataset with {len(df)} rows")
    except Exception as e:
        print(f"Error loading train dataset: {str(e)}")
        exit(1)
    
    # Print dataset information
    print("\nDataset Information:")
    print(f"Total number of comments: {len(df)}")
    print(f"Number of toxic comments (target > 0.5): {len(df[df['target'] > 0.5])}")
    print(f"Number of non-toxic comments (target <= 0.5): {len(df[df['target'] <= 0.5])}")
    
    # Process the data
    print("\nProcessing data...")
    df.fillna(0, inplace=True)
    
    # Print column names to verify we have the right columns
    print("\nAvailable columns:")
    print(df.columns.tolist())
    
    # Check if required columns exist
    required_columns = ['identity_attack'] + ['black', 'white', 'asian', 'latino', 'other_race_or_ethnicity']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"\nWarning: Missing required columns: {missing_columns}")
        print("Please check if the dataset has the correct format.")
        exit(1)
    
    # Create bias labels
    print("\nCreating bias labels...")
    race_columns = ['black', 'white', 'asian', 'latino', 'other_race_or_ethnicity']
    toxicity_metrics = ['identity_attack', 'severe_toxicity', 'insult']
    df['race_bias'] = (
        (df[toxicity_metrics].max(axis=1) > 0.3) &  # Lower threshold to 0.3
        (df[race_columns].max(axis=1) > 0.3)  # Lower threshold to 0.3
    ).astype(int)
    
    # Keep only needed columns
    df_filtered = df[['comment_text', 'race_bias']]
    df_filtered = df_filtered.rename(columns={'race_bias': 'label'})
    
    # Print bias distribution
    print("\nBias Distribution:")
    biased_count = len(df_filtered[df_filtered['label'] == 1])
    non_biased_count = len(df_filtered[df_filtered['label'] == 0])
    print(f"Number of biased comments: {biased_count}")
    print(f"Number of non-biased comments: {non_biased_count}")
    
    if biased_count == 0 or non_biased_count == 0:
        print("\nError: No examples found for one or both classes!")
        print("Please check your data processing logic.")
        exit(1)
    
    # Sample size for balanced dataset
    max_samples_per_class = 1000
    sample_size = min(min(biased_count, non_biased_count), max_samples_per_class)
    print(f"\nUsing sample size of {sample_size} for each class")
    
    # Create balanced dataset with error handling
    try:
        print("\nCreating balanced dataset...")
        fair = df_filtered[df_filtered['label'] == 0].sample(n=sample_size, random_state=42)
        biased = df_filtered[df_filtered['label'] == 1].sample(n=sample_size, random_state=42)
        
        print(f"Sampled {len(fair)} non-biased examples")
        print(f"Sampled {len(biased)} biased examples")
        
        df_balanced = pd.concat([fair, biased])
        df_balanced = df_balanced.sample(frac=1, random_state=42)
        
        df_balanced.to_csv("bias_dataset.csv", index=False)
        print(f"Successfully created balanced dataset with {len(df_balanced)} samples")
        
    except ValueError as e:
        print(f"\nError during sampling: {e}")
        print("Falling back to using all available data...")
        df_balanced = df_filtered
        df_balanced.to_csv("bias_dataset.csv", index=False)
        print(f"Created dataset with all available data: {len(df_balanced)} samples")
    
    # Train the model
    print("\nTraining the model...")
    train_model("bias_dataset.csv", "bias_detector_model.pt")
    
    # Test the model with some example comments
    print("\nTesting the model with example comments:")
    detector = BiasDetector("bias_detector_model.pt")
    
    # Example 1: A potentially biased comment
    test_text1 = "People from that race are always causing trouble."
    analysis1 = detector.analyze_output(test_text1)
    print("\nExample 1 (Potentially Biased):")
    print(f"Text: {test_text1}")
    print(f"Is Biased: {analysis1['is_biased']}")
    print(f"Confidence: {analysis1['confidence']:.2f}")
    print(f"Category Scores: {analysis1['category_scores']}")
    print(f"Explanation: {analysis1['explanation']}")
    if analysis1['alternative']:
        print(f"Alternative: {analysis1['alternative']}")
    
    # Example 2: A neutral comment
    test_text2 = "The weather is nice today."
    analysis2 = detector.analyze_output(test_text2)
    print("\nExample 2 (Neutral):")
    print(f"Text: {test_text2}")
    print(f"Is Biased: {analysis2['is_biased']}")
    print(f"Confidence: {analysis2['confidence']:.2f}")
    print(f"Category Scores: {analysis2['category_scores']}")
    print(f"Explanation: {analysis2['explanation']}")
    if analysis2['alternative']:
        print(f"Alternative: {analysis2['alternative']}")
    
    # Example 3: Another racially biased comment
    test_text3 = "Those people are taking all our jobs and living off welfare."
    analysis3 = detector.analyze_output(test_text3)
    print("\nExample 3 (Racially Biased):")
    print(f"Text: {test_text3}")
    print(f"Is Biased: {analysis3['is_biased']}")
    print(f"Confidence: {analysis3['confidence']:.2f}")
    print(f"Category Scores: {analysis3['category_scores']}")
    print(f"Explanation: {analysis3['explanation']}")
    if analysis3['alternative']:
        print(f"Alternative: {analysis3['alternative']}")
    
    # Example 4: Another neutral comment
    test_text4 = "I enjoy reading books and watching movies in my free time."
    analysis4 = detector.analyze_output(test_text4)
    print("\nExample 4 (Neutral):")
    print(f"Text: {test_text4}")
    print(f"Is Biased: {analysis4['is_biased']}")
    print(f"Confidence: {analysis4['confidence']:.2f}")
    print(f"Category Scores: {analysis4['category_scores']}")
    print(f"Explanation: {analysis4['explanation']}")
    if analysis4['alternative']:
        print(f"Alternative: {analysis4['alternative']}")

    # Example 5: Subtle racial bias using coded language
    test_text5 = "These immigrants from certain neighborhoods are making our area unsafe and lowering property values."
    analysis5 = detector.analyze_output(test_text5)
    print("\nExample 5 (Subtle Racial Bias):")
    print(f"Text: {test_text5}")
    print(f"Is Biased: {analysis5['is_biased']}")
    print(f"Confidence: {analysis5['confidence']:.2f}")
    print(f"Category Scores: {analysis5['category_scores']}")
    print(f"Explanation: {analysis5['explanation']}")
    if analysis5['alternative']:
        print(f"Alternative: {analysis5['alternative']}")

    # Example 6: Racial bias with stereotyping
    test_text6 = "Minorities from that ethnic background are naturally better at certain jobs because of their genetics."
    analysis6 = detector.analyze_output(test_text6)
    print("\nExample 6 (Racial Stereotyping):")
    print(f"Text: {test_text6}")
    print(f"Is Biased: {analysis6['is_biased']}")
    print(f"Confidence: {analysis6['confidence']:.2f}")
    print(f"Category Scores: {analysis6['category_scores']}")
    print(f"Explanation: {analysis6['explanation']}")
    if analysis6['alternative']:
        print(f"Alternative: {analysis6['alternative']}")