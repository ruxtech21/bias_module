from train_model_CUDA_version import BiasDetector

def print_analysis(text, analysis):
    print(f"\nText: {text}")
    print(f"Is Biased: {analysis['is_biased']}")
    print(f"Confidence: {analysis['confidence']:.2f}")
    print(f"Explanation: {analysis['explanation']}")
    if analysis['alternative']:
        print(f"Alternative: {analysis['alternative']}")
    print("-" * 80)

# Initialize detector
detector = BiasDetector("bias_detector_model.pt")

# Test 1: Pseudo-scientific racism masked as research
test1 = """Recent studies have shown significant cognitive differences between populations from different geographical regions, suggesting evolutionary adaptations may play a role in academic performance."""

# Test 2: Economic dog whistle with property values
test2 = """The changing demographics in our neighborhood have led to declining property values and increased strain on local resources. We need to maintain our community's traditional character."""

# Test 3: Cultural supremacy with coded language
test3 = """Western civilization has developed superior cultural values and advanced societal norms. Other cultures need to adapt to our modern standards of living."""

# Test 4: Subtle stereotype about work ethic
test4 = """While they're hardworking, these immigrants tend to have different values about punctuality and professional conduct. It's just a cultural difference we need to address."""

# Test 5: Complex intersectional bias with multiple patterns
test5 = """The data clearly shows that certain ethnic groups are overrepresented in specific neighborhoods, correlating with lower property maintenance standards and different community values. This is simply an empirical observation about cultural differences."""

# Test 6: Control example (neutral text about demographics)
test6 = """The city's population has become more diverse over the past decade, leading to a rich variety of cultural festivals, restaurants, and community events that benefit everyone."""

# Run tests
print("Testing Challenging Examples for Bias Detection")
print("=" * 80)

examples = [test1, test2, test3, test4, test5, test6]
for i, test in enumerate(examples, 1):
    print(f"\nTest {i}:")
    analysis = detector.analyze_output(test)
    print_analysis(test, analysis) 