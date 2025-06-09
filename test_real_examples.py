import pandas as pd
import numpy as np
from train_model_CUDA_version import BiasDetector

def clean_text(text):
    """Clean text by removing newlines and excessive spaces"""
    if isinstance(text, str):
        return ' '.join(text.replace('\n', ' ').split())
    return ''

def test_examples(model, examples):
    print("\nTesting Model on Real Examples from Dataset:\n")
    for i, (text, label) in enumerate(examples, 1):
        text = clean_text(text)
        print(f"Example {i}:")
        print(f"Text: {text}")
        print(f"True Label: {'Biased' if label == 1 else 'Not Biased'}")
        
        # Get model prediction
        analysis = model.analyze_output(text)
        print(f"Model Prediction: {'Biased' if analysis['is_biased'] else 'Not Biased'}")
        print(f"Confidence: {analysis['confidence']:.2f}")
        print(f"Explanation: {analysis['explanation']}")
        print("\n" + "="*80 + "\n")

def main():
    # Load dataset
    print("Loading dataset...")
    df = pd.read_csv('bias_dataset.csv')
    
    # Clean the text data
    df['comment_text'] = df['comment_text'].apply(clean_text)
    
    # Remove empty comments
    df = df[df['comment_text'].str.len() > 0]
    
    print(f"\nDataset Statistics:")
    print(f"Total samples: {len(df)}")
    print(f"Biased comments: {len(df[df['label'] == 1])}")
    print(f"Non-biased comments: {len(df[df['label'] == 0])}")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Get random examples of each class, excluding very short comments
    biased_df = df[
        (df['label'] == 1) & 
        (df['comment_text'].str.len() > 50)
    ].sample(n=5)
    
    non_biased_df = df[
        (df['label'] == 0) & 
        (df['comment_text'].str.len() > 50)
    ].sample(n=5)
    
    biased_examples = list(zip(
        biased_df['comment_text'],
        biased_df['label']
    ))
    
    non_biased_examples = list(zip(
        non_biased_df['comment_text'],
        non_biased_df['label']
    ))
    
    # Initialize model
    print("\nInitializing model...")
    model = BiasDetector("bias_detector_model.pt")
    
    # Test biased examples
    print("\nTesting Biased Examples from Dataset:")
    test_examples(model, biased_examples)
    
    # Test non-biased examples
    print("\nTesting Non-Biased Examples from Dataset:")
    test_examples(model, non_biased_examples)

if __name__ == "__main__":
    main() 