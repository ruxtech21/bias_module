import pandas as pd
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification, get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from typing import Dict, List, Tuple, Optional
import os
from sklearn.model_selection import train_test_split
import torch.nn as nn

# GPU Configuration
def setup_gpu():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Please check your GPU installation.")
    
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)
    
    print("\nGPU Configuration:")
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory Available: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # Enable cuDNN auto-tuner
    torch.backends.cudnn.benchmark = True
    
    return device

# Initialize GPU
DEVICE = setup_gpu()

class BiasDataset(Dataset):
    def __init__(self, csv_file, tokenizer_name='roberta-base', max_length=256):
        df = pd.read_csv(csv_file)
        self.texts = df['comment_text'].tolist()
        self.labels = df['label'].tolist()
        self.tokenizer = RobertaTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.encodings = self.tokenizer(self.texts, truncation=True, padding=True, max_length=max_length)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

class EnhancedBiasDetector(nn.Module):
    def __init__(self, num_labels=2, dropout_rate=0.2):
        super().__init__()
        self.roberta = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=num_labels)
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, **inputs):
        outputs = self.roberta(**inputs)
        outputs.logits = self.dropout(outputs.logits)
        return outputs

def train_model(dataset_path: str, output_model_path: str, epochs: int = 5, batch_size: int = 32):
    """Train the bias detection model with enhanced training process and GPU optimization."""
    print("\nInitializing training process...")
    
    # Initialize dataset and split into train/validation with GPU-optimized batch size
    full_dataset = BiasDataset(dataset_path)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Use num_workers for faster data loading
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=4,
        pin_memory=True  # This speeds up the data transfer to GPU
    )
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True
    )
    
    # Initialize model and move to GPU
    model = EnhancedBiasDetector().to(DEVICE)
    
    # Enable mixed precision training
    scaler = torch.cuda.amp.GradScaler()
    
    # Initialize optimizer with weight decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    
    # Initialize learning rate scheduler
    num_training_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )
    
    # Training loop with early stopping
    best_val_loss = float('inf')
    patience = 3
    patience_counter = 0
    
    print(f"\nStarting training on {torch.cuda.get_device_name(0)}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Batch size: {batch_size}")
    print(f"Number of epochs: {epochs}")
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        total_train_loss = 0
        
        for batch_idx, batch in enumerate(train_dataloader):
            # Move batch to GPU
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            
            # Mixed precision training
            with torch.cuda.amp.autocast():
                outputs = model(**batch)
                loss = outputs.loss
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
            
            total_train_loss += loss.item()
            
            # Print progress
            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs} | Batch {batch_idx + 1}/{len(train_dataloader)} | Loss: {loss.item():.4f}")
        
        avg_train_loss = total_train_loss / len(train_dataloader)
        
        # Validation phase
        model.eval()
        total_val_loss = 0
        correct_predictions = 0
        total_predictions = 0
        
        with torch.no_grad():
            for batch in val_dataloader:
                batch = {k: v.to(DEVICE) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss
                total_val_loss += loss.item()
                
                predictions = torch.argmax(outputs.logits, dim=1)
                correct_predictions += (predictions == batch['labels']).sum().item()
                total_predictions += len(batch['labels'])
        
        avg_val_loss = total_val_loss / len(val_dataloader)
        accuracy = correct_predictions / total_predictions
        
        print(f"\nEpoch {epoch + 1}/{epochs}")
        print(f"Average Training Loss: {avg_train_loss:.4f}")
        print(f"Average Validation Loss: {avg_val_loss:.4f}")
        print(f"Validation Accuracy: {accuracy:.4f}")
        
        # GPU Memory stats
        print(f"GPU Memory Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        print(f"GPU Memory Cached: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
        
        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save the best model
            torch.save(model.state_dict(), output_model_path)
            print(f"New best model saved with validation loss: {best_val_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break
    
    print(f"\nTraining completed. Best model saved to {output_model_path}")
    
    # Clear GPU cache
    torch.cuda.empty_cache()

class BiasDetector:
    def __init__(self, model_path: Optional[str] = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        self.model = EnhancedBiasDetector()
        if model_path:
            self.model.load_state_dict(torch.load(model_path))
        self.model.to(self.device)
        self.model.eval()
        self.bias_threshold = 0.3
        
        # Define bias categories based on Jigsaw dataset
        self.bias_categories = {
            'race': ['black', 'white', 'asian', 'latino', 'other_race_or_ethnicity'],
            'gender': ['male', 'female', 'transgender', 'other_gender'],
            'religion': ['christian', 'jewish', 'muslim', 'hindu', 'buddhist', 'atheist', 'other_religion'],
            'sexual_orientation': ['heterosexual', 'homosexual_gay_or_lesbian', 'bisexual', 'other_sexual_orientation']
        }

    def detect_bias(self, text: str) -> Tuple[bool, float, Dict[str, float]]:
        """Detect if the given text contains bias and return category-specific scores."""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=1)
            bias_prob = probabilities[0][1].item()
            
        # In a real implementation, you would have separate models for each category
        # For now, we'll use the main bias probability for all categories
        category_scores = {
            category: bias_prob for category in self.bias_categories.keys()
        }
            
        return bias_prob > self.bias_threshold, bias_prob, category_scores

    def explain_bias(self, text: str) -> str:
        """Generate a detailed explanation for why the text might be biased."""
        is_biased, confidence, category_scores = self.detect_bias(text)
        if not is_biased:
            return "No significant bias detected in the text."
        
        # Identify the most likely bias categories
        top_categories = sorted(
            category_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:2]  # Top 2 categories
        
        explanation = f"The text shows signs of bias (confidence: {confidence:.2f}). "
        explanation += "The bias appears to be primarily related to: "
        explanation += ", ".join([f"{cat} ({score:.2f})" for cat, score in top_categories])
        
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

def load_dataset(file_path: str) -> pd.DataFrame:
    """Load the dataset with proper error handling."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found at: {file_path}")
    
    try:
        df = pd.read_csv(file_path)
        print(f"Successfully loaded dataset with {len(df)} rows")
        return df
    except Exception as e:
        raise Exception(f"Error loading dataset: {str(e)}")

if __name__ == "__main__":
    # First create the dataset
    print("\nLoading dataset...")
    dataset_path = r"C:\Users\ovidi\OneDrive\Desktop\Cursor (licenta Rux)\Jigsaw dataset\train.csv"
    try:
        df = load_dataset(dataset_path)
    except Exception as e:
        print(f"Error: {str(e)}")
        print("Please make sure the dataset file exists and is accessible.")
        exit(1)
    
    # Print dataset information
    print("\nDataset Information:")
    print(f"Total number of comments: {len(df)}")
    print(f"Number of toxic comments (target > 0.5): {len(df[df['target'] > 0.5])}")
    print(f"Number of non-toxic comments (target <= 0.5): {len(df[df['target'] <= 0.5])}")
    
    # Process the data as before
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
    # Consider multiple toxicity metrics for better bias detection
    toxicity_metrics = ['identity_attack', 'severe_toxicity', 'insult']
    df['race_bias'] = (
        (df[toxicity_metrics].max(axis=1) > 0.3) &  # Lower threshold to 0.3
        (df[race_columns].max(axis=1) > 0.3)  # Lower threshold to 0.3
    ).astype(int)
    
    # Print some statistics about the bias labels
    print("\nBias label statistics:")
    bias_counts = df['race_bias'].value_counts()
    print(bias_counts)
    
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
    
    # Increase sample size to 1000 per class
    max_sample_size = 1000
    sample_size = min(min(biased_count, non_biased_count), max_sample_size)
    print(f"\nUsing sample size of {sample_size} for each class")
    
    # Create balanced dataset with error handling
    try:
        print("\nCreating balanced dataset...")
        # Get the samples
        fair = df_filtered[df_filtered['label'] == 0].sample(n=sample_size, random_state=42)
        biased = df_filtered[df_filtered['label'] == 1].sample(n=sample_size, random_state=42)
        
        # Verify we got the samples
        print(f"Sampled {len(fair)} non-biased examples")
        print(f"Sampled {len(biased)} biased examples")
        
        # Combine and shuffle
        df_balanced = pd.concat([fair, biased])
        df_balanced = df_balanced.sample(frac=1, random_state=42)
        
        # Save the dataset
        df_balanced.to_csv("bias_dataset.csv", index=False)
        print(f"Successfully created balanced dataset with {len(df_balanced)} samples")
        
    except ValueError as e:
        print(f"\nError during sampling: {e}")
        print("Falling back to using all available data...")
        # If sampling fails, use all available data
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