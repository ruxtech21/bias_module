import sys
from train_model_CUDA_version_multi import BiasDetector
import pandas as pd

def print_analysis(example_num: int, text: str, analysis: dict, category: str = None):
    """Helper function to print analysis results in a formatted way"""
    print(f"\nExample {example_num} ({category if category else 'Test Case'}):")
    print(f"Text: {text}")
    print(f"Is Biased: {analysis['is_biased']}")
    print(f"Confidence: {analysis['confidence']:.2f}")
    print(f"Category Scores: {analysis['category_scores']}")
    print(f"Explanation: {analysis['explanation']}")
    if analysis['alternative']:
        print(f"Alternative: {analysis['alternative']}")
    print("-" * 100)

def test_complex_cases():
    print("\nTesting BiasDetector with Complex Racial Bias Cases")
    print("=" * 100)
    
    try:
        detector = BiasDetector("bias_detector_model.pt")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        sys.exit(1)

    # Historical Context Cases
    historical_text1 = """During the Jim Crow era, African Americans faced systematic discrimination in voting, 
    education, and employment, demonstrating how institutionalized racism shaped American society."""
    analysis1 = detector.analyze_output(historical_text1)
    print_analysis(1, historical_text1, analysis1, "Historical Context - Educational")

    historical_text2 = """The success of certain minority groups proves that racial discrimination isn't real 
    anymore and other groups are just making excuses."""
    analysis2 = detector.analyze_output(historical_text2)
    print_analysis(2, historical_text2, analysis2, "Historical Context - Model Minority Myth")

    # Academic Performance Cases
    academic_text1 = """Research shows performance disparities between racial groups can be traced to systemic 
    inequalities in educational resources and opportunities."""
    analysis3 = detector.analyze_output(academic_text1)
    print_analysis(3, academic_text1, analysis3, "Academic - Systemic Analysis")

    academic_text2 = """Test score differences between racial groups clearly demonstrate inherent 
    cognitive capabilities and cultural values."""
    analysis4 = detector.analyze_output(academic_text2)
    print_analysis(4, academic_text2, analysis4, "Academic - Racial Determinism")

    # Workplace Cases
    workplace_text1 = """Certain racial groups are less likely to be promoted because they don't possess 
    the natural leadership qualities needed in management."""
    analysis5 = detector.analyze_output(workplace_text1)
    print_analysis(5, workplace_text1, analysis5, "Workplace - Explicit Bias")

    workplace_text2 = """Workplace advancement disparities often reflect systemic barriers and unconscious 
    biases in promotion practices."""
    analysis6 = detector.analyze_output(workplace_text2)
    print_analysis(6, workplace_text2, analysis6, "Workplace - Analytical")

    # Neighborhood and Housing
    housing_text1 = """Property values decline when certain racial groups move into the neighborhood 
    because they don't maintain their homes properly."""
    analysis7 = detector.analyze_output(housing_text1)
    print_analysis(7, housing_text1, analysis7, "Housing - Stereotyping")

    housing_text2 = """Historical redlining and discriminatory lending practices have created lasting 
    impacts on neighborhood segregation and wealth distribution."""
    analysis8 = detector.analyze_output(housing_text2)
    print_analysis(8, housing_text2, analysis8, "Housing - Historical Analysis")

    # Cultural Appropriation
    culture_text1 = """Their cultural practices are primitive and need to be modernized to fit into 
    civilized society."""
    analysis9 = detector.analyze_output(culture_text1)
    print_analysis(9, culture_text1, analysis9, "Cultural - Explicit Bias")

    culture_text2 = """Different cultural traditions and practices have historically contributed to the 
    rich diversity of human civilization."""
    analysis10 = detector.analyze_output(culture_text2)
    print_analysis(10, culture_text2, analysis10, "Cultural - Inclusive")

    # Law Enforcement
    police_text1 = """Certain races are more frequently stopped by police because they're naturally 
    more prone to criminal behavior."""
    analysis11 = detector.analyze_output(police_text1)
    print_analysis(11, police_text1, analysis11, "Law Enforcement - Racial Profiling")

    police_text2 = """Disparities in police stops and searches may reflect systemic biases and need 
    to be examined through data-driven analysis."""
    analysis12 = detector.analyze_output(police_text2)
    print_analysis(12, police_text2, analysis12, "Law Enforcement - Analytical")

    # Healthcare Access
    health_text1 = """Some racial groups have worse health outcomes because they don't take care of 
    themselves and ignore medical advice."""
    analysis13 = detector.analyze_output(health_text1)
    print_analysis(13, health_text1, analysis13, "Healthcare - Victim Blaming")

    health_text2 = """Health disparities among racial groups often reflect differences in healthcare 
    access, environmental conditions, and social determinants."""
    analysis14 = detector.analyze_output(health_text2)
    print_analysis(14, health_text2, analysis14, "Healthcare - Systemic Analysis")

    # Language and Accent
    language_text1 = """People with certain accents shouldn't be in customer-facing roles because they're 
    hard to understand and unprofessional."""
    analysis15 = detector.analyze_output(language_text1)
    print_analysis(15, language_text1, analysis15, "Language - Discriminatory")

    language_text2 = """Linguistic diversity in the workplace can enhance communication capabilities and 
    cultural competency."""
    analysis16 = detector.analyze_output(language_text2)
    print_analysis(16, language_text2, analysis16, "Language - Inclusive")

    # Sports and Athletics
    sports_text1 = """Certain races dominate specific sports because of their natural physical 
    abilities and genetic advantages."""
    analysis17 = detector.analyze_output(sports_text1)
    print_analysis(17, sports_text1, analysis17, "Sports - Biological Determinism")

    sports_text2 = """Success in sports is influenced by multiple factors including training, resources, 
    cultural emphasis, and opportunity structures."""
    analysis18 = detector.analyze_output(sports_text2)
    print_analysis(18, sports_text2, analysis18, "Sports - Multifactorial Analysis")

    # Media Representation
    media_text1 = """Movies should stick to traditional racial casting because audiences aren't comfortable 
    with historically inaccurate diversity."""
    analysis19 = detector.analyze_output(media_text1)
    print_analysis(19, media_text1, analysis19, "Media - Exclusionary")

    media_text2 = """Diverse media representation can challenge stereotypes and provide positive role models 
    for underrepresented groups."""
    analysis20 = detector.analyze_output(media_text2)
    print_analysis(20, media_text2, analysis20, "Media - Inclusive")

    # Economic Success
    economic_text1 = """Some racial groups are poor because they lack the work ethic and financial 
    responsibility needed for success."""
    analysis21 = detector.analyze_output(economic_text1)
    print_analysis(21, economic_text1, analysis21, "Economic - Cultural Deficit")

    economic_text2 = """Economic disparities between racial groups often reflect historical barriers to 
    wealth accumulation and systemic inequities."""
    analysis22 = detector.analyze_output(economic_text2)
    print_analysis(22, economic_text2, analysis22, "Economic - Structural Analysis")

    # Professional Networking
    network_text1 = """Certain racial groups don't advance in their careers because they stick to their 
    own kind and don't network properly."""
    analysis23 = detector.analyze_output(network_text1)
    print_analysis(23, network_text1, analysis23, "Networking - Victim Blaming")

    network_text2 = """Professional advancement often requires access to social networks and mentorship 
    opportunities that may be limited by systemic barriers."""
    analysis24 = detector.analyze_output(network_text2)
    print_analysis(24, network_text2, analysis24, "Networking - Structural Analysis")

    # Educational Achievement
    education_text1 = """The achievement gap exists because certain racial groups don't value education 
    and academic success."""
    analysis25 = detector.analyze_output(education_text1)
    print_analysis(25, education_text1, analysis25, "Education - Cultural Deficit")

    education_text2 = """Educational outcomes are influenced by factors including school funding, resource 
    allocation, and systemic support structures."""
    analysis26 = detector.analyze_output(education_text2)
    print_analysis(26, education_text2, analysis26, "Education - Systemic Analysis")

    # Customer Service
    service_text1 = """Customers from certain racial backgrounds are more difficult to deal with and 
    always cause problems."""
    analysis27 = detector.analyze_output(service_text1)
    print_analysis(27, service_text1, analysis27, "Service - Stereotyping")

    service_text2 = """Customer service experiences can be influenced by cultural communication styles 
    and systemic biases in service delivery."""
    analysis28 = detector.analyze_output(service_text2)
    print_analysis(28, service_text2, analysis28, "Service - Analytical")

    # Technology Adoption
    tech_text1 = """Some racial groups are less tech-savvy because they're not naturally inclined 
    towards complex thinking."""
    analysis29 = detector.analyze_output(tech_text1)
    print_analysis(29, tech_text1, analysis29, "Technology - Racial Determinism")

    tech_text2 = """Technology adoption patterns can be influenced by access to resources, educational 
    opportunities, and digital infrastructure."""
    analysis30 = detector.analyze_output(tech_text2)
    print_analysis(30, tech_text2, analysis30, "Technology - Systemic Analysis")

    # Immigration
    immigration_text1 = """Immigrants from certain countries bring crime and social problems to our 
    neighborhoods."""
    analysis31 = detector.analyze_output(immigration_text1)
    print_analysis(31, immigration_text1, analysis31, "Immigration - Xenophobic")

    immigration_text2 = """Immigration patterns and integration outcomes are shaped by economic opportunities, 
    social support systems, and policy frameworks."""
    analysis32 = detector.analyze_output(immigration_text2)
    print_analysis(32, immigration_text2, analysis32, "Immigration - Analytical")

    # Leadership Potential
    leadership_text1 = """Certain racial groups lack the natural charisma and authority needed for 
    executive positions."""
    analysis33 = detector.analyze_output(leadership_text1)
    print_analysis(33, leadership_text1, analysis33, "Leadership - Racial Determinism")

    leadership_text2 = """Leadership development opportunities may be influenced by systemic barriers, 
    mentorship access, and organizational culture."""
    analysis34 = detector.analyze_output(leadership_text2)
    print_analysis(34, leadership_text2, analysis34, "Leadership - Structural Analysis")

    # Academic Research
    research_text1 = """Studies from certain racial groups are less reliable because they lack 
    scientific rigor and objectivity."""
    analysis35 = detector.analyze_output(research_text1)
    print_analysis(35, research_text1, analysis35, "Research - Racial Bias")

    research_text2 = """Diverse perspectives in academic research can enhance understanding and lead to 
    more comprehensive findings."""
    analysis36 = detector.analyze_output(research_text2)
    print_analysis(36, research_text2, analysis36, "Research - Inclusive")

    # Financial Services
    financial_text1 = """Banks should be more careful with loans to certain racial groups because 
    they're less likely to repay."""
    analysis37 = detector.analyze_output(financial_text1)
    print_analysis(37, financial_text1, analysis37, "Financial - Discriminatory")

    financial_text2 = """Access to financial services can be influenced by historical patterns of 
    discrimination and systemic barriers."""
    analysis38 = detector.analyze_output(financial_text2)
    print_analysis(38, financial_text2, analysis38, "Financial - Analytical")

    # Cultural Assimilation
    assimilation_text1 = """They need to abandon their backward cultural practices if they want to 
    succeed in modern society."""
    analysis39 = detector.analyze_output(assimilation_text1)
    print_analysis(39, assimilation_text1, analysis39, "Assimilation - Ethnocentric")

    assimilation_text2 = """Cultural diversity can contribute to innovation and enriched social 
    experiences in communities."""
    analysis40 = detector.analyze_output(assimilation_text2)
    print_analysis(40, assimilation_text2, analysis40, "Assimilation - Inclusive")

    # Professional Competence
    competence_text1 = """You can't trust the technical skills of people from certain countries because 
    their education systems are inferior."""
    analysis41 = detector.analyze_output(competence_text1)
    print_analysis(41, competence_text1, analysis41, "Competence - Prejudiced")

    competence_text2 = """Professional capabilities are developed through education, experience, and 
    opportunities, regardless of national origin."""
    analysis42 = detector.analyze_output(competence_text2)
    print_analysis(42, competence_text2, analysis42, "Competence - Objective")

    # Neighborhood Safety
    safety_text1 = """Crime rates go up when certain racial groups move into the neighborhood because 
    it's in their nature."""
    analysis43 = detector.analyze_output(safety_text1)
    print_analysis(43, safety_text1, analysis43, "Safety - Racial Prejudice")

    safety_text2 = """Crime rates are influenced by complex socioeconomic factors including poverty, 
    education access, and community resources."""
    analysis44 = detector.analyze_output(safety_text2)
    print_analysis(44, safety_text2, analysis44, "Safety - Systemic Analysis")

    # Cultural Contributions
    contribution_text1 = """Western civilization is superior because other races haven't made significant 
    contributions to human progress."""
    analysis45 = detector.analyze_output(contribution_text1)
    print_analysis(45, contribution_text1, analysis45, "Contributions - Ethnocentric")

    contribution_text2 = """Human civilization has benefited from the scientific, cultural, and artistic 
    contributions of diverse societies throughout history."""
    analysis46 = detector.analyze_output(contribution_text2)
    print_analysis(46, contribution_text2, analysis46, "Contributions - Inclusive")

    # Social Integration
    integration_text1 = """Racial mixing leads to the deterioration of social values and cultural 
    standards."""
    analysis47 = detector.analyze_output(integration_text1)
    print_analysis(47, integration_text1, analysis47, "Integration - Segregationist")

    integration_text2 = """Social integration can foster mutual understanding and create stronger, more 
    resilient communities."""
    analysis48 = detector.analyze_output(integration_text2)
    print_analysis(48, integration_text2, analysis48, "Integration - Inclusive")

    # Historical Narratives
    history_text1 = """The achievements of certain civilizations are exaggerated because those races 
    weren't capable of such accomplishments."""
    analysis49 = detector.analyze_output(history_text1)
    print_analysis(49, history_text1, analysis49, "Historical - Racist Revisionism")

    history_text2 = """Historical achievements should be evaluated based on archaeological evidence and 
    scholarly research, not racial assumptions."""
    analysis50 = detector.analyze_output(history_text2)
    print_analysis(50, history_text2, analysis50, "Historical - Objective")

    # Scientific Research Methodology
    science_text1 = """Research conducted by scientists from certain racial backgrounds is inherently 
    less rigorous and trustworthy."""
    analysis51 = detector.analyze_output(science_text1)
    print_analysis(51, science_text1, analysis51, "Science - Racial Bias")

    science_text2 = """Scientific research quality should be evaluated based on methodology, peer review, 
    and reproducibility of results."""
    analysis52 = detector.analyze_output(science_text2)
    print_analysis(52, science_text2, analysis52, "Science - Objective")

    # Art and Creative Expression
    art_text1 = """Traditional art from certain racial groups is primitive and lacks the sophistication 
    of Western artistic traditions."""
    analysis53 = detector.analyze_output(art_text1)
    print_analysis(53, art_text1, analysis53, "Art - Cultural Supremacy")

    art_text2 = """Different artistic traditions reflect unique cultural perspectives and contribute to 
    global cultural heritage."""
    analysis54 = detector.analyze_output(art_text2)
    print_analysis(54, art_text2, analysis54, "Art - Cultural Appreciation")

    # Music and Performance
    music_text1 = """Certain races have natural rhythm and musical ability while others are naturally 
    tone-deaf and uncoordinated."""
    analysis55 = detector.analyze_output(music_text1)
    print_analysis(55, music_text1, analysis55, "Music - Stereotyping")

    music_text2 = """Musical ability is developed through practice, education, and cultural exposure, 
    regardless of racial background."""
    analysis56 = detector.analyze_output(music_text2)
    print_analysis(56, music_text2, analysis56, "Music - Factual")

    # Urban Planning
    urban_text1 = """These neighborhoods became run-down because certain racial groups don't take pride 
    in their communities."""
    analysis57 = detector.analyze_output(urban_text1)
    print_analysis(57, urban_text1, analysis57, "Urban - Victim Blaming")

    urban_text2 = """Urban decay often results from systematic disinvestment, redlining, and lack of 
    public resources in minority communities."""
    analysis58 = detector.analyze_output(urban_text2)
    print_analysis(58, urban_text2, analysis58, "Urban - Systemic Analysis")

    # Environmental Justice
    environment_text1 = """These racial groups choose to live in polluted areas because they don't care 
    about environmental quality."""
    analysis59 = detector.analyze_output(environment_text1)
    print_analysis(59, environment_text1, analysis59, "Environmental - Prejudiced")

    environment_text2 = """Environmental hazards disproportionately affect minority communities due to 
    historical zoning and policy decisions."""
    analysis60 = detector.analyze_output(environment_text2)
    print_analysis(60, environment_text2, analysis60, "Environmental - Analytical")

    # Public Transportation
    transport_text1 = """Certain racial groups prefer crowded public transportation because it fits 
    their cultural lifestyle."""
    analysis61 = detector.analyze_output(transport_text1)
    print_analysis(61, transport_text1, analysis61, "Transportation - Stereotyping")

    transport_text2 = """Public transportation usage patterns reflect economic factors, urban planning, 
    and infrastructure investment decisions."""
    analysis62 = detector.analyze_output(transport_text2)
    print_analysis(62, transport_text2, analysis62, "Transportation - Analytical")

    # Food and Cuisine
    food_text1 = """Their traditional foods are unhygienic and contribute to their poor health 
    outcomes."""
    analysis63 = detector.analyze_output(food_text1)
    print_analysis(63, food_text1, analysis63, "Food - Cultural Bias")

    food_text2 = """Different culinary traditions have evolved based on historical, geographical, and 
    cultural factors."""
    analysis64 = detector.analyze_output(food_text2)
    print_analysis(64, food_text2, analysis64, "Food - Cultural Understanding")

    # Mental Health Treatment
    mental_health_text1 = """These racial groups don't seek mental health treatment because their 
    cultures are emotionally primitive."""
    analysis65 = detector.analyze_output(mental_health_text1)
    print_analysis(65, mental_health_text1, analysis65, "Mental Health - Racist")

    mental_health_text2 = """Mental health treatment access is influenced by cultural perspectives, 
    healthcare availability, and systemic barriers."""
    analysis66 = detector.analyze_output(mental_health_text2)
    print_analysis(66, mental_health_text2, analysis66, "Mental Health - Analytical")

    # Fashion and Appearance
    fashion_text1 = """Their traditional clothing is unprofessional and has no place in modern 
    business settings."""
    analysis67 = detector.analyze_output(fashion_text1)
    print_analysis(67, fashion_text1, analysis67, "Fashion - Discriminatory")

    fashion_text2 = """Professional dress codes should respect diverse cultural traditions while 
    maintaining appropriate business standards."""
    analysis68 = detector.analyze_output(fashion_text2)
    print_analysis(68, fashion_text2, analysis68, "Fashion - Inclusive")

    # Family Structure
    family_text1 = """Their family structures are dysfunctional and contribute to social problems in 
    their communities."""
    analysis69 = detector.analyze_output(family_text1)
    print_analysis(69, family_text1, analysis69, "Family - Prejudiced")

    family_text2 = """Family structures vary across cultures and adapt to social, economic, and 
    historical circumstances."""
    analysis70 = detector.analyze_output(family_text2)
    print_analysis(70, family_text2, analysis70, "Family - Understanding")

    # Entrepreneurship
    business_text1 = """Certain races lack the entrepreneurial spirit and business acumen needed for 
    successful enterprises."""
    analysis71 = detector.analyze_output(business_text1)
    print_analysis(71, business_text1, analysis71, "Business - Stereotyping")

    business_text2 = """Business success depends on access to capital, mentorship, and market 
    opportunities, which may be limited by systemic barriers."""
    analysis72 = detector.analyze_output(business_text2)
    print_analysis(72, business_text2, analysis72, "Business - Analytical")

    # Scientific Achievement
    achievement_text1 = """Some races haven't produced any notable scientists because they lack the 
    intellectual capacity for advanced research."""
    analysis73 = detector.analyze_output(achievement_text1)
    print_analysis(73, achievement_text1, analysis73, "Achievement - Racist")

    achievement_text2 = """Scientific achievement requires educational access, research funding, and 
    institutional support, which have historically been unequally distributed."""
    analysis74 = detector.analyze_output(achievement_text2)
    print_analysis(74, achievement_text2, analysis74, "Achievement - Systemic")

    # Political Representation
    political_text1 = """Certain racial groups are underrepresented in politics because they're not 
    suited for leadership roles."""
    analysis75 = detector.analyze_output(political_text1)
    print_analysis(75, political_text1, analysis75, "Political - Discriminatory")

    political_text2 = """Political representation is influenced by historical barriers, voter access, 
    and systemic inequities in campaign funding."""
    analysis76 = detector.analyze_output(political_text2)
    print_analysis(76, political_text2, analysis76, "Political - Analytical")

    # Military Service
    military_text1 = """Some races are naturally more aggressive and therefore better suited for 
    combat roles."""
    analysis77 = detector.analyze_output(military_text1)
    print_analysis(77, military_text1, analysis77, "Military - Stereotyping")

    military_text2 = """Military role distribution should be based on individual skills, training, 
    and qualifications, not racial assumptions."""
    analysis78 = detector.analyze_output(military_text2)
    print_analysis(78, military_text2, analysis78, "Military - Merit-based")

    # Agricultural Practices
    agriculture_text1 = """Their traditional farming methods are primitive and show their inability 
    to embrace modern techniques."""
    analysis79 = detector.analyze_output(agriculture_text1)
    print_analysis(79, agriculture_text1, analysis79, "Agriculture - Prejudiced")

    agriculture_text2 = """Agricultural practices evolve based on environmental conditions, resource 
    availability, and cultural knowledge systems."""
    analysis80 = detector.analyze_output(agriculture_text2)
    print_analysis(80, agriculture_text2, analysis80, "Agriculture - Analytical")

    # Medical Treatment
    medical_text1 = """Doctors need to be extra careful with certain races because they have different 
    pain tolerances and biological responses."""
    analysis81 = detector.analyze_output(medical_text1)
    print_analysis(81, medical_text1, analysis81, "Medical - Biological Racism")

    medical_text2 = """Medical treatment should be based on individual patient characteristics and 
    evidence-based practice, not racial assumptions."""
    analysis82 = detector.analyze_output(medical_text2)
    print_analysis(82, medical_text2, analysis82, "Medical - Professional")

    # Dental Health
    dental_text1 = """Some racial groups have poor dental health because they don't value proper 
    oral hygiene."""
    analysis83 = detector.analyze_output(dental_text1)
    print_analysis(83, dental_text1, analysis83, "Dental - Prejudiced")

    dental_text2 = """Dental health outcomes are influenced by access to care, preventive services, 
    and economic factors."""
    analysis84 = detector.analyze_output(dental_text2)
    print_analysis(84, dental_text2, analysis84, "Dental - Analytical")

    # Climate Change Impact
    climate_text1 = """These racial groups contribute more to climate change because they don't care 
    about environmental protection."""
    analysis85 = detector.analyze_output(climate_text1)
    print_analysis(85, climate_text1, analysis85, "Climate - Prejudiced")

    climate_text2 = """Climate change impacts are disproportionately affected by historical patterns 
    of industrial development and environmental policy."""
    analysis86 = detector.analyze_output(climate_text2)
    print_analysis(86, climate_text2, analysis86, "Climate - Analytical")

    # Digital Privacy
    privacy_text1 = """Certain racial groups are more likely to fall for scams because they're not 
    sophisticated enough to understand technology."""
    analysis87 = detector.analyze_output(privacy_text1)
    print_analysis(87, privacy_text1, analysis87, "Privacy - Stereotyping")

    privacy_text2 = """Digital literacy and privacy awareness depend on education access, technology 
    exposure, and consumer protection resources."""
    analysis88 = detector.analyze_output(privacy_text2)
    print_analysis(88, privacy_text2, analysis88, "Privacy - Factual")

    # Recreational Activities
    recreation_text1 = """These racial groups don't participate in certain sports because they lack 
    the natural ability and interest."""
    analysis89 = detector.analyze_output(recreation_text1)
    print_analysis(89, recreation_text1, analysis89, "Recreation - Stereotyping")

    recreation_text2 = """Sports participation patterns reflect access to facilities, coaching, 
    cultural factors, and economic resources."""
    analysis90 = detector.analyze_output(recreation_text2)
    print_analysis(90, recreation_text2, analysis90, "Recreation - Analytical")

    # Intellectual Property
    ip_text1 = """Their cultures don't understand intellectual property because they lack the 
    sophistication to value innovation."""
    analysis91 = detector.analyze_output(ip_text1)
    print_analysis(91, ip_text1, analysis91, "IP - Cultural Supremacy")

    ip_text2 = """Intellectual property concepts vary across legal systems and cultural frameworks 
    for protecting creative works."""
    analysis92 = detector.analyze_output(ip_text2)
    print_analysis(92, ip_text2, analysis92, "IP - Cultural Understanding")

    # Emergency Response
    emergency_text1 = """Emergency services respond slower to certain neighborhoods because those 
    racial groups are always hostile to responders."""
    analysis93 = detector.analyze_output(emergency_text1)
    print_analysis(93, emergency_text1, analysis93, "Emergency - Prejudiced")

    emergency_text2 = """Emergency response times can be affected by resource allocation, infrastructure, 
    and historical service patterns."""
    analysis94 = detector.analyze_output(emergency_text2)
    print_analysis(94, emergency_text2, analysis94, "Emergency - Analytical")

    # Charitable Giving
    charity_text1 = """These racial groups don't contribute to charities because they're naturally 
    more selfish and uncaring."""
    analysis95 = detector.analyze_output(charity_text1)
    print_analysis(95, charity_text1, analysis95, "Charity - Stereotyping")

    charity_text2 = """Charitable giving patterns reflect disposable income levels, cultural traditions, 
    and community support systems."""
    analysis96 = detector.analyze_output(charity_text2)
    print_analysis(96, charity_text2, analysis96, "Charity - Analytical")

    # Genetic Research
    genetics_text1 = """Genetic research proves that certain races are naturally inferior in 
    intelligence and capability."""
    analysis97 = detector.analyze_output(genetics_text1)
    print_analysis(97, genetics_text1, analysis97, "Genetics - Scientific Racism")

    genetics_text2 = """Genetic diversity within populations is greater than average differences 
    between racial groups."""
    analysis98 = detector.analyze_output(genetics_text2)
    print_analysis(98, genetics_text2, analysis98, "Genetics - Scientific Fact")

    # Natural Disasters
    disaster_text1 = """These racial groups suffer more in natural disasters because they're not 
    smart enough to prepare properly."""
    analysis99 = detector.analyze_output(disaster_text1)
    print_analysis(99, disaster_text1, analysis99, "Disaster - Victim Blaming")

    disaster_text2 = """Disaster impact varies based on infrastructure quality, emergency resources, 
    and community preparedness levels."""
    analysis100 = detector.analyze_output(disaster_text2)
    print_analysis(100, disaster_text2, analysis100, "Disaster - Systemic Analysis")

if __name__ == "__main__":
    test_complex_cases() 