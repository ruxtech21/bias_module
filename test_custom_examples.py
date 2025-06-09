from train_model_CUDA_version import BiasDetector

def test_custom_examples(model):
    examples = [
        # Example 1: Subtle bias masked as economic/cultural concern
        """The changing demographics in our traditionally stable neighborhood have coincided with declining property maintenance standards. 
        I'm not saying it's anyone's fault, but there's clearly a different set of cultural values at play. 
        Long-time residents are concerned about their property investments, and you can't blame them for wanting to preserve the character 
        of the community they've spent decades building. It's not about exclusion, it's about maintaining standards.""",
        
        # Example 2: Non-biased criticism of economic policy
        """The proposed tax legislation disproportionately affects middle-class families across all communities. 
        The data shows that households earning between $40,000 and $80,000 annually will see the largest percentage increase in their tax burden, 
        regardless of location or background. We need to focus on creating economic policies that support working families and promote 
        sustainable growth for everyone.""",
        
        # Example 3: Subtle bias wrapped in academic-sounding language
        """Recent demographic studies suggest interesting correlations between cultural background and academic achievement. 
        While some groups consistently outperform others in certain subjects, we must consider whether these patterns reflect 
        inherent aptitudes or societal factors. The data seems to indicate that certain communities have developed particular 
        cognitive strengths over generations, possibly due to historical circumstances and genetic factors. This isn't about 
        superiority, it's simply an objective observation of measurable differences.""",
        
        # Example 4: Non-biased discussion of community change
        """Our city's landscape has evolved significantly over the past decade, bringing both challenges and opportunities. 
        The influx of new businesses has created jobs, while rising housing costs have affected residents across all neighborhoods. 
        Community organizations are working together to address these changes through inclusive planning processes, ensuring that 
        development benefits both long-term residents and newcomers. The key is finding balance between growth and preservation."""
    ]
    
    print("\nTesting Model on Complex Custom Examples:\n")
    for i, text in enumerate(examples, 1):
        print(f"Example {i}:")
        print(f"Text: {text}\n")
        
        # Get model prediction
        analysis = model.analyze_output(text)
        print(f"Model Prediction: {'Biased' if analysis['is_biased'] else 'Not Biased'}")
        print(f"Confidence: {analysis['confidence']:.2f}")
        print(f"Explanation: {analysis['explanation']}")
        print("\n" + "="*80 + "\n")

def main():
    # Initialize model
    print("Initializing model...")
    model = BiasDetector("bias_detector_model.pt")
    
    # Test examples
    test_custom_examples(model)

if __name__ == "__main__":
    main() 