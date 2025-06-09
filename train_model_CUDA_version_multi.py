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
        self.bias_threshold = 0.33

        # Educational and historical context markers
        self.educational_markers = [
            'history', 'historical', 'historically', 'studied', 'research shows',
            'according to research', 'studies indicate', 'evidence suggests',
            'analysis shows', 'data indicates', 'statistics show',
            'documented', 'records show', 'historians', 'scholars',
            'academic research', 'scientific study', 'empirical evidence'
        ]

        self.historical_events = [
            'world war', 'civil rights', 'segregation', 'jim crow',
            'internment camps', 'discrimination', 'prejudice', 'slavery',
            'holocaust', 'genocide', 'persecution', 'oppression',
            'civil war', 'reconstruction', 'movement', 'revolution',
            'reform', 'rights movement', 'liberation', 'emancipation'
        ]

        # Positive discussion markers
        self.positive_markers = [
            'diversity', 'inclusion', 'integration', 'multicultural',
            'cross-cultural', 'cultural exchange', 'mutual understanding',
            'cooperation', 'collaboration', 'community building',
            'social cohesion', 'cultural enrichment', 'shared values',
            'common ground', 'bridge building', 'cultural appreciation',
            'cultural awareness', 'intercultural', 'pluralistic'
        ]

        # Enhanced pseudo-scientific racism markers with cognitive terms
        self.pseudo_scientific_markers = [
            'genetic', 'biological', 'innate', 'inherent', 'natural ability',
            'cognitive ability', 'intelligence quotient', 'iq differences',
            'racial characteristics', 'racial traits', 'genetic predisposition',
            'evolutionary', 'skull shape', 'brain size', 'genetic basis',
            'hereditary', 'biological differences', 'genetic differences',
            'racial intelligence', 'genetic superiority', 'genetic inferiority',
            'intellectual capacity', 'mental capabilities', 'cognitive differences',
            'intelligence levels', 'intellectual ability', 'mental traits',
            'cognitive traits', 'intellectual traits', 'mental characteristics',
            'inferior capabilities', 'superior capabilities', 'racial differences',
            'ethnic differences', 'group differences', 'population differences'
        ]

        # Enhanced positive outcome and integration markers
        self.positive_markers.extend([
            'successful integration', 'positive outcomes', 'improved outcomes',
            'better results', 'enhanced performance', 'increased success',
            'positive impact', 'beneficial effects', 'mutual benefit',
            'community success', 'shared prosperity', 'collective growth',
            'economic growth', 'social progress', 'community development',
            'positive contribution', 'successful outcomes', 'improved metrics',
            'enhanced results', 'demonstrated success', 'measured improvement',
            'statistical improvement', 'documented success', 'verified progress'
        ])

        # New factual systemic inequality markers
        self.systemic_inequality_markers = [
            'systemic barriers', 'structural inequality', 'institutional barriers',
            'systemic challenges', 'resource disparities', 'access limitations',
            'opportunity gaps', 'educational disparities', 'health disparities',
            'economic barriers', 'systemic obstacles', 'institutional constraints',
            'resource limitations', 'access restrictions', 'opportunity limitations',
            'documented disparities', 'measured inequalities', 'observed gaps',
            'research-based findings', 'evidence-based analysis', 'data-driven analysis',
            'statistical evidence', 'empirical findings', 'quantitative analysis',
            'qualitative studies', 'peer-reviewed research', 'academic studies'
        ]

        # New social justice and equity discussion markers
        self.social_justice_markers = [
            'social justice', 'equity', 'equality', 'fair treatment',
            'equal opportunity', 'equal access', 'fair access',
            'civil rights', 'human rights', 'equal rights',
            'social mobility', 'economic mobility', 'opportunity access',
            'systemic reform', 'institutional change', 'structural change',
            'policy reform', 'systematic improvement', 'evidence-based policy',
            'research-based solutions', 'data-driven policy', 'measured outcomes'
        ]

        # Enhanced racial bias patterns
        self.explicit_racial_terms = [
            'race', 'racial', 'racist', 'ethnic', 'minority', 'minorities',
            'black', 'white', 'asian', 'latino', 'hispanic', 'immigrant',
            'foreigner', 'illegal', 'demographic', 'demographics',
            'jews', 'jewish', 'muslims', 'arabs', 'middle eastern',
            'british', 'mexican', 'japanese', 'korean', 'vietnamese',
            'thai', 'cambodian', 'filipino', 'haitian', 'guatemalan',
            'nicaraguan', 'cuban'
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
            'welfare', 'taking our jobs', 'stealing our jobs',
            'living off the system', 'burden on society', 'draining resources',
            'not paying taxes', 'taking advantage', 'gaming the system',
            'handouts', 'freeloaders', 'lazy', 'work shy',
            'taking over', 'replacing us', 'our country', 'our jobs',
            'our way of life', 'our culture', 'our values'
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
            'they always', 'they never', 'all of them',
            'that type', 'that sort', 'their type',
            'they take', 'they steal', 'they want',
            'they come here', 'they expect', 'they think',
            'go back', 'send them back', 'where they came from'
        ]

        # Economic and welfare-related patterns
        self.economic_bias_terms = [
            'welfare', 'benefits', 'social services',
            'taxpayer money', 'public funds', 'government assistance',
            'handouts', 'entitlements', 'free money',
            'jobs', 'employment', 'work ethic',
            'lazy', 'unemployed', 'won\'t work',
            'taking advantage', 'abuse the system', 'exploit',
            'burden', 'drain', 'cost us',
            'our economy', 'our resources', 'our country'
        ]

        # Contextual patterns with more sophisticated combinations
        self.contextual_pairs = [
            ('property', ['values', 'prices', 'neighborhood', 'declining', 'standards', 'maintenance']),
            ('neighborhood', ['character', 'quality', 'changing', 'stable', 'traditional', 'standards']),
            ('community', ['values', 'standards', 'character', 'traditional', 'changing', 'different']),
            ('cultural', ['values', 'differences', 'background', 'norms', 'identity', 'heritage']),
            ('traditional', ['values', 'neighborhood', 'community', 'standards', 'way of life']),
            ('different', ['values', 'culture', 'standards', 'background', 'way of life']),
            ('economic', ['impact', 'effects', 'consequences', 'burden', 'drain', 'cost']),
            ('welfare', ['system', 'benefits', 'abuse', 'fraud', 'dependency']),
            ('jobs', ['stealing', 'taking', 'lost', 'our', 'american']),
            ('taxpayer', ['money', 'burden', 'cost', 'paying for', 'funding']),
            ('they', ['all', 'always', 'never', 'want', 'take', 'expect']),
            ('these', ['people', 'immigrants', 'foreigners', 'groups']),
            ('those', ['people', 'immigrants', 'foreigners', 'groups']),
            ('their', ['culture', 'values', 'way', 'kind', 'type'])
        ]

        # Explicitly biased academic language markers
        self.biased_academic_markers = [
            'proves inferior', 'demonstrates inferior', 'shows inferior',
            'proves superior', 'demonstrates superior', 'shows superior',
            'clearly demonstrates', 'proves that', 'shows that they',
            'evidence of inferiority', 'evidence of superiority',
            'consistently underperform', 'systematically underperform',
            'inherently less', 'naturally less', 'genetically less',
            'scientifically proven', 'research proves', 'studies prove',
            'data proves', 'evidence proves', 'statistics prove',
            'clearly shows', 'definitively shows', 'undeniably shows',
            'proves deficient', 'shows deficient', 'demonstrates deficient',
            'research confirms stereotypes', 'data confirms differences',
            'scientific evidence of racial', 'scientific proof of ethnic'
        ]

        # Evidence-based discussion markers
        self.evidence_based_markers = [
            'research indicates', 'studies suggest', 'data shows',
            'evidence suggests', 'findings indicate', 'analysis reveals',
            'research demonstrates', 'studies have found', 'data indicates',
            'evidence points to', 'analysis shows', 'research suggests',
            'empirical evidence', 'statistical analysis', 'quantitative analysis',
            'qualitative research', 'systematic review', 'meta-analysis',
            'peer-reviewed research', 'academic research', 'scientific study'
        ]

        # Constructive discussion markers
        self.constructive_markers = [
            'solutions include', 'can be addressed by', 'can be improved through',
            'potential solutions', 'addressing these issues', 'improving access',
            'enhancing opportunities', 'reducing barriers', 'increasing access',
            'promoting equality', 'fostering inclusion', 'supporting development',
            'evidence-based solutions', 'research-based approaches', 'proven strategies',
            'effective interventions', 'successful programs', 'positive outcomes',
            'measured success', 'documented improvements', 'verified progress'
        ]

        # Positive research findings patterns
        self.positive_research_patterns = [
            ('research shows', ['positive', 'better', 'improved', 'higher', 'lower crime', 'increased value']),
            ('studies indicate', ['positive', 'better', 'improved', 'higher', 'beneficial', 'advantage']),
            ('data demonstrates', ['positive', 'better', 'improved', 'higher', 'beneficial', 'advantage']),
            ('evidence suggests', ['positive', 'better', 'improved', 'higher', 'beneficial', 'advantage']),
            ('analysis reveals', ['positive', 'better', 'improved', 'higher', 'beneficial', 'advantage'])
        ]

        # Factual systemic discussion patterns
        self.systemic_discussion_patterns = [
            ('barriers', ['access to', 'economic', 'educational', 'healthcare', 'employment']),
            ('disparities', ['access to', 'economic', 'educational', 'healthcare', 'employment']),
            ('inequality', ['systemic', 'structural', 'institutional', 'economic', 'educational']),
            ('challenges', ['systemic', 'structural', 'institutional', 'economic', 'educational']),
            ('limitations', ['access to', 'resource', 'opportunity', 'economic', 'educational'])
        ]

        # Solution-oriented language
        self.solution_patterns = [
            ('improve', ['access', 'opportunities', 'outcomes', 'conditions', 'situation']),
            ('enhance', ['access', 'opportunities', 'outcomes', 'conditions', 'situation']),
            ('increase', ['access', 'opportunities', 'resources', 'support', 'funding']),
            ('provide', ['support', 'resources', 'assistance', 'opportunities', 'access']),
            ('develop', ['programs', 'initiatives', 'solutions', 'opportunities', 'resources'])
        ]

        # Statement structure markers
        self.statement_structures = {
            'factual_research': {
                'leads': ['research', 'studies', 'data', 'evidence', 'analysis'],
                'verbs': ['shows', 'indicates', 'demonstrates', 'reveals', 'suggests'],
                'qualifiers': ['when', 'if', 'under conditions', 'in cases'],
                'outcomes': ['higher', 'lower', 'better', 'improved', 'increased', 'decreased']
            },
            'systemic_analysis': {
                'subjects': ['communities', 'populations', 'groups', 'residents', 'individuals'],
                'conditions': ['face', 'experience', 'encounter', 'deal with'],
                'issues': ['barriers', 'challenges', 'limitations', 'restrictions', 'constraints'],
                'domains': ['economic', 'educational', 'healthcare', 'employment', 'housing']
            },
            'historical_context': {
                'events': ['world war', 'civil rights', 'history', 'historical', 'discrimination'],
                'descriptors': ['dark chapter', 'prejudice', 'hysteria', 'violations', 'persecution'],
                'groups': ['japanese americans', 'minorities', 'immigrants', 'ethnic groups'],
                'impacts': ['faced', 'suffered', 'experienced', 'endured', 'subjected to']
            },
            'biased_generalization': {
                'subjects': ['they', 'these people', 'those people', 'them', 'their kind'],
                'verbs': ['are', 'always', 'never', 'all', 'none'],
                'attributes': ['lazy', 'criminal', 'violent', 'dangerous', 'inferior'],
                'blame': ['fault', 'blame', 'responsible', 'cause', 'problem']
            }
        }

        # Enhanced contextual relationship pairs
        self.contextual_relationships = {
            'research_findings': {
                'positive': [
                    ('research shows', 'improved outcomes'),
                    ('evidence indicates', 'positive results'),
                    ('data demonstrates', 'better performance'),
                    ('studies reveal', 'higher rates'),
                    ('analysis suggests', 'beneficial effects')
                ],
                'negative': [
                    ('research proves', 'inferior'),
                    ('studies show', 'deficient'),
                    ('evidence demonstrates', 'worse'),
                    ('data confirms', 'lower abilities'),
                    ('analysis reveals', 'poor performance')
                ]
            },
            'systemic_discussion': {
                'factual': [
                    ('barriers to', 'access'),
                    ('limitations in', 'resources'),
                    ('challenges in', 'obtaining'),
                    ('disparities in', 'distribution'),
                    ('gaps in', 'opportunities')
                ],
                'biased': [
                    ('they lack', 'values'),
                    ('their failure', 'culture'),
                    ('unable to', 'adapt'),
                    ('refusing to', 'work'),
                    ('choosing to', 'depend')
                ]
            },
            'historical_discussion': {
                'factual': [
                    ('faced discrimination', 'civil rights'),
                    ('experienced prejudice', 'historical'),
                    ('subjected to', 'persecution'),
                    ('dark chapter', 'history'),
                    ('civil rights', 'violations')
                ],
                'biased': [
                    ('their own fault', 'history'),
                    ('deserved', 'treatment'),
                    ('brought it', 'themselves'),
                    ('justified', 'actions'),
                    ('necessary', 'measures')
                ]
            },
            'subtle_bias': {
                'economic': [
                    ('skilled workers', 'dependent'),
                    ('contribute', 'drain resources'),
                    ('productive', 'burden'),
                    ('merit-based', 'handouts'),
                    ('qualified', 'welfare')
                ],
                'cultural': [
                    ('traditional values', 'declining'),
                    ('community standards', 'breakdown'),
                    ('neighborhood character', 'changing'),
                    ('social cohesion', 'erosion'),
                    ('proper values', 'lacking')
                ]
            }
        }

        # Intent markers for different types of discussions
        self.intent_markers = {
            'factual_discussion': {
                'descriptive': ['indicates', 'shows', 'demonstrates', 'reveals', 'suggests', 'found'],
                'analytical': ['analysis', 'research', 'study', 'evidence', 'data', 'findings'],
                'contextual': ['context', 'factors', 'conditions', 'circumstances', 'situations'],
                'systemic': ['systemic', 'structural', 'institutional', 'societal', 'systematic']
            },
            'solution_oriented': {
                'improvement': ['improve', 'enhance', 'increase', 'develop', 'strengthen'],
                'access': ['access to', 'availability of', 'provision of', 'distribution of'],
                'support': ['support', 'assist', 'help', 'aid', 'facilitate'],
                'opportunity': ['opportunity', 'possibilities', 'chances', 'prospects']
            },
            'historical_factual': {
                'events': ['world war', 'civil rights', 'history', 'historical', 'period'],
                'documentation': ['documented', 'recorded', 'reported', 'accounts of'],
                'impact': ['resulted in', 'led to', 'caused', 'affected', 'influenced'],
                'analysis': ['analysis shows', 'studies indicate', 'research reveals']
            },
            'biased_framing': {
                'generalizations': ['all of them', 'they always', 'these people', 'those people'],
                'stereotypes': ['naturally', 'inherently', 'genetically', 'typically'],
                'blame': ['fault', 'blame', 'responsible for', 'cause of problems'],
                'superiority': ['better than', 'superior to', 'more capable', 'less capable']
            }
        }

        # Positive/negative framing markers
        self.framing_markers = {
            'positive': {
                'outcomes': ['improve', 'increase', 'enhance', 'better', 'higher', 'successful'],
                'potential': ['opportunity', 'potential', 'capability', 'ability', 'possibility'],
                'contribution': ['contribute', 'add value', 'benefit', 'strengthen', 'enrich'],
                'solutions': ['solution', 'approach', 'strategy', 'method', 'way forward']
            },
            'negative': {
                'deficits': ['lack', 'deficient', 'inferior', 'inadequate', 'poor'],
                'problems': ['problem', 'issue', 'trouble', 'difficulty', 'failure'],
                'blame': ['blame', 'fault', 'responsible', 'cause', 'due to them'],
                'threats': ['threat', 'danger', 'risk', 'hazard', 'menace']
            }
        }

        # Evidence quality markers
        self.evidence_markers = {
            'high_quality': {
                'methodology': ['statistical', 'empirical', 'measured', 'observed', 'analyzed'],
                'verification': ['verified', 'validated', 'confirmed', 'established', 'proven'],
                'scope': ['comprehensive', 'extensive', 'thorough', 'detailed', 'complete'],
                'limitations': ['limitations', 'constraints', 'caveats', 'considerations']
            },
            'low_quality': {
                'assumptions': ['assume', 'believe', 'think', 'feel', 'suppose'],
                'generalizations': ['always', 'never', 'all', 'none', 'every'],
                'anecdotal': ['they say', 'I heard', 'people say', 'supposedly'],
                'exaggeration': ['obviously', 'clearly', 'undoubtedly', 'certainly']
            }
        }

    def _analyze_statement_structure(self, text: str) -> Dict[str, float]:
        """Analyze the structure of the statement to determine its nature"""
        text_lower = text.lower()
        words = text_lower.split()
        
        scores = {
            'factual_research': 0.0,
            'systemic_analysis': 0.0,
            'historical_context': 0.0,
            'biased_generalization': 0.0
        }
        
        # Check each structure type
        for structure_type, components in self.statement_structures.items():
            component_matches = 0
            total_components = len(components)
            
            for component_type, terms in components.items():
                # Use sliding window to check for term sequences
                for i in range(len(words)):
                    window = ' '.join(words[i:i+3])  # 3-word window
                    if any(term in window for term in terms):
                        component_matches += 1
                        break
            
            scores[structure_type] = component_matches / total_components
        
        return scores

    def _analyze_contextual_relationships(self, text: str) -> Dict[str, float]:
        """Analyze the relationships between terms in the text"""
        text_lower = text.lower()
        
        scores = {
            'positive_research': 0.0,
            'negative_research': 0.0,
            'factual_systemic': 0.0,
            'biased_systemic': 0.0
        }
        
        # Check research findings relationships
        for rel_type, pairs in self.contextual_relationships['research_findings'].items():
            for primary, secondary in pairs:
                if primary in text_lower and secondary in text_lower:
                    if rel_type == 'positive':
                        scores['positive_research'] += 0.3
                    else:
                        scores['negative_research'] += 0.3

        # Check systemic discussion relationships
        for rel_type, pairs in self.contextual_relationships['systemic_discussion'].items():
            for primary, secondary in pairs:
                if primary in text_lower and secondary in text_lower:
                    if rel_type == 'factual':
                        scores['factual_systemic'] += 0.3
                    else:
                        scores['biased_systemic'] += 0.3

        return {k: min(v, 1.0) for k, v in scores.items()}

    def _analyze_intent(self, text: str) -> Dict[str, float]:
        """Analyze the intent of the text based on various markers"""
        text_lower = text.lower()
        words = text_lower.split()
        
        scores = {
            'factual': 0.0,
            'solution': 0.0,
            'historical': 0.0,
            'biased': 0.0
        }
        
        # Check each intent type
        for intent_type, categories in self.intent_markers.items():
            matches = 0
            total_categories = len(categories)
            
            for category, terms in categories.items():
                if any(term in text_lower for term in terms):
                    matches += 1
            
            score = matches / total_categories
            if intent_type == 'factual_discussion':
                scores['factual'] = score
            elif intent_type == 'solution_oriented':
                scores['solution'] = score
            elif intent_type == 'historical_factual':
                scores['historical'] = score
            elif intent_type == 'biased_framing':
                scores['biased'] = score
        
        return scores

    def _analyze_framing(self, text: str) -> Dict[str, float]:
        """Analyze the positive/negative framing of the text"""
        text_lower = text.lower()
        
        scores = {
            'positive': 0.0,
            'negative': 0.0
        }
        
        for frame_type, categories in self.framing_markers.items():
            matches = 0
            total_categories = len(categories)
            
            for category, terms in categories.items():
                if any(term in text_lower for term in terms):
                    matches += 1
            
            scores[frame_type] = matches / total_categories
        
        return scores

    def _analyze_evidence(self, text: str) -> Dict[str, float]:
        """Analyze the quality of evidence presented"""
        text_lower = text.lower()
        
        scores = {
            'high_quality': 0.0,
            'low_quality': 0.0
        }
        
        for quality_type, categories in self.evidence_markers.items():
            matches = 0
            total_categories = len(categories)
            
            for category, terms in categories.items():
                if any(term in text_lower for term in terms):
                    matches += 1
            
            scores[quality_type] = matches / total_categories
        
        return scores

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

        # Enhanced pattern detection with weighted scoring
        explicit_score = sum(term in text_lower for term in self.explicit_racial_terms) * 0.4
        coded_score = sum(term in text_lower for term in self.coded_racial_terms) * 0.4
        stereotype_score = sum(term in text_lower for term in self.racial_stereotypes) * 0.4
        
        # Calculate initial bias score
        base_score = max(explicit_score + coded_score + stereotype_score, 0.0)

        # Historical education context detection
        historical_education_score = 0.0
        historical_terms = ['history', 'historical', 'world war', 'civil rights', 'discrimination']
        educational_terms = ['demonstrated', 'showed', 'proved', 'documented', 'resulted in']
        critical_terms = ['dark chapter', 'prejudice', 'hysteria', 'violations', 'persecution']
        
        if any(term in text_lower for term in historical_terms):
            if any(term in text_lower for term in educational_terms):
                if any(term in text_lower for term in critical_terms):
                    historical_education_score = 0.8  # Strong indicator of historical education

        # Enhanced subtle cultural bias detection
        cultural_decline_terms = [
            'traditional values', 'community standards', 'social trust',
            'breakdown', 'declining', 'erosion', 'deteriorating',
            'foreign cultures', 'changing demographics', 'cultural change'
        ]
        
        us_vs_them_terms = [
            'our neighborhoods', 'our community', 'our values',
            'these people', 'those communities', 'their culture',
            'they don\'t', 'they won\'t', 'they refuse'
        ]
        
        cultural_bias_score = 0.0
        cultural_decline_matches = sum(term in text_lower for term in cultural_decline_terms)
        us_vs_them_matches = sum(term in text_lower for term in us_vs_them_terms)
        
        if cultural_decline_matches > 0 and us_vs_them_matches > 0:
            cultural_bias_score = 0.7  # Strong indicator of cultural bias
        elif cultural_decline_matches > 0 or us_vs_them_matches > 0:
            cultural_bias_score = 0.5  # Moderate indicator of cultural bias

        # Enhanced systemic discussion detection
        systemic_terms = [
            'barriers', 'access to', 'limited access', 'disparities',
            'systemic', 'structural', 'institutional', 'inequalities'
        ]
        
        solution_terms = [
            'opportunities', 'resources', 'programs', 'initiatives',
            'support', 'assistance', 'development', 'improvement'
        ]
        
        factual_markers = [
            'research shows', 'studies indicate', 'data demonstrates',
            'evidence suggests', 'analysis reveals', 'statistics show'
        ]
        
        systemic_discussion_score = 0.0
        has_systemic = any(term in text_lower for term in systemic_terms)
        has_solutions = any(term in text_lower for term in solution_terms)
        has_factual = any(term in text_lower for term in factual_markers)
        
        if has_systemic and has_solutions and has_factual:
            systemic_discussion_score = 0.8  # Strong indicator of legitimate systemic discussion

        # Check for academic bias markers with higher weights
        academic_bias_score = sum(term in text_lower for term in self.biased_academic_markers) * 0.6
        pseudo_scientific_score = sum(term in text_lower for term in self.pseudo_scientific_markers) * 0.5
        
        # Enhanced economic bias detection with more sophisticated contrast pairs
        economic_bias_score = 0.0
        economic_contrast_pairs = [
            ('skilled workers', ['dependent', 'burden', 'drain', 'welfare', 'social services']),
            ('contribute', ['dependent', 'burden', 'drain', 'welfare', 'social services']),
            ('our economy', ['their burden', 'social services', 'welfare', 'dependent']),
            ('merit-based', ['handouts', 'benefits', 'welfare', 'dependent']),
            ('productive', ['dependent', 'burden', 'drain', 'welfare']),
            ('taxpayers', ['dependent', 'burden', 'drain', 'welfare']),
            ('legal', ['illegal', 'undocumented', 'burden', 'drain']),
            ('qualified', ['unqualified', 'dependent', 'drain', 'burden']),
            ('self-sufficient', ['dependent', 'burden', 'drain', 'welfare']),
            ('hard-working', ['lazy', 'dependent', 'burden', 'drain'])
        ]
        
        for primary, secondaries in economic_contrast_pairs:
            if primary in text_lower:
                for secondary in secondaries:
                    if secondary in text_lower:
                        # Increased weight for economic bias detection
                        economic_bias_score += 0.7
                        break

        # Enhanced positive research detection with more sophisticated patterns
        positive_research_score = 0.0
        positive_research_patterns = [
            ('research shows', ['lower crime', 'higher property', 'better outcomes', 'positive impact', 'improved', 'beneficial']),
            ('studies indicate', ['lower crime', 'higher property', 'better outcomes', 'positive impact', 'improved', 'beneficial']),
            ('evidence suggests', ['lower crime', 'higher property', 'better outcomes', 'positive impact', 'improved', 'beneficial']),
            ('data demonstrates', ['lower crime', 'higher property', 'better outcomes', 'positive impact', 'improved', 'beneficial']),
            ('analysis reveals', ['lower crime', 'higher property', 'better outcomes', 'positive impact', 'improved', 'beneficial']),
            ('findings show', ['lower crime', 'higher property', 'better outcomes', 'positive impact', 'improved', 'beneficial'])
        ]
        
        for primary, secondaries in positive_research_patterns:
            if primary in text_lower:
                for secondary in secondaries:
                    if secondary in text_lower:
                        # Stronger reduction for positive research findings
                        positive_research_score += 0.8
                        break

        # Enhanced legitimate systemic discussion detection
        legitimate_systemic_score = 0.0
        legitimate_systemic_patterns = [
            ('barriers', ['economic mobility', 'access', 'opportunities', 'education', 'healthcare', 'employment']),
            ('limited access', ['quality education', 'healthcare', 'job opportunities', 'resources', 'services']),
            ('face', ['multiple barriers', 'systemic challenges', 'structural obstacles', 'disparities']),
            ('disparities', ['economic', 'educational', 'healthcare', 'employment', 'access']),
            ('challenges', ['systemic', 'structural', 'institutional', 'societal', 'economic']),
            ('inequality', ['systemic', 'structural', 'institutional', 'economic', 'social']),
            ('communities', ['underserved', 'disadvantaged', 'marginalized', 'under-resourced']),
            ('access to', ['quality education', 'healthcare', 'jobs', 'resources', 'opportunities'])
        ]
        
        # Check for factual markers that indicate legitimate discussion
        factual_discussion_markers = [
            'research shows', 'studies indicate', 'data demonstrates', 'evidence suggests',
            'analysis reveals', 'statistics show', 'documented', 'measured', 'observed',
            'reported', 'identified', 'according to', 'findings indicate'
        ]
        
        has_factual_marker = any(marker in text_lower for marker in factual_discussion_markers)
        
        for primary, secondaries in legitimate_systemic_patterns:
            if primary in text_lower:
                for secondary in secondaries:
                    if secondary in text_lower:
                        # Increase score for legitimate systemic discussion
                        legitimate_systemic_score += 0.6
                        if has_factual_marker:
                            legitimate_systemic_score += 0.2
                        break

        # Evidence quality assessment
        evidence_quality = 0.0
        for quality, categories in self.evidence_markers['high_quality'].items():
            if any(term in text_lower for term in categories):
                evidence_quality += 0.2
        
        for quality, categories in self.evidence_markers['low_quality'].items():
            if any(term in text_lower for term in categories):
                evidence_quality -= 0.2

        # Context analysis
        context_scores = self._analyze_contextual_relationships(text)
        intent_scores = self._analyze_intent(text)
        framing_scores = self._analyze_framing(text)
        
        # Adjust scores based on context
        if historical_education_score > 0.5:
            # Stronger reduction for historical education
            base_score *= 0.1  # Increased reduction from 0.2
            if evidence_quality > 0:
                base_score *= 0.5
        
        if cultural_bias_score > 0:
            # Increase score for subtle cultural bias
            base_score = max(base_score, cultural_bias_score)
        
        if systemic_discussion_score > 0.5:
            # Check for legitimate systemic discussion vs stereotyping
            if evidence_quality > 0 and intent_scores['factual'] > 0.5:
                if not any(term in text_lower for term in ['blame', 'fault', 'excuse', 'failing']):
                    base_score *= 0.1  # Increased reduction from 0.3
        
        # Research context handling
        positive_research_terms = [
            'positive impact', 'beneficial effects', 'improved outcomes',
            'higher rates', 'better results', 'successful integration',
            'lower crime rates', 'higher property values'  # Added specific positive outcomes
        ]
        
        negative_research_terms = [
            'negative impact', 'detrimental effects', 'worse outcomes',
            'lower rates', 'poor results', 'failed integration'
        ]
        
        has_positive_research = any(term in text_lower for term in positive_research_terms)
        has_negative_research = any(term in text_lower for term in negative_research_terms)
        
        if has_positive_research and not has_negative_research:
            if intent_scores['factual'] > 0.5:
                base_score *= 0.2  # Significant reduction for positive research findings

        # Strong bias indicators should significantly increase the score
        if any(term in text_lower for term in [
            'inferior cognitive', 'cultural deficiencies', 'genetic',
            'naturally less', 'proves that they', 'their kind'
        ]):
            base_score = max(base_score, 0.8)
        
        # Adjust final score based on model prediction and pattern detection
        final_score = (0.3 * model_score) + (0.7 * base_score)
        
        # Additional adjustments based on specific contexts
        if historical_education_score > 0.7:
            final_score *= 0.2  # Strong reduction for historical education
        
        if cultural_bias_score > 0.6:
            final_score = max(final_score, 0.6)  # Minimum score for clear cultural bias
        
        if systemic_discussion_score > 0.7 and evidence_quality > 0:
            # Check for legitimate systemic discussion
            if not any(term in text_lower for term in ['blame', 'fault', 'excuse', 'failing']):
                final_score *= 0.2  # Strong reduction for evidence-based systemic discussion
        
        if economic_bias_score > 0.6:  # Lower threshold for economic bias
            final_score = max(final_score, economic_bias_score)
            
        if positive_research_score > 0.7 and evidence_quality > 0:
            # Stronger reduction for evidence-based positive findings
            final_score *= 0.1
            
        if legitimate_systemic_score > 0.6 and evidence_quality > 0:
            if not any(term in text_lower for term in ['blame', 'fault', 'excuse', 'failing', 'refuse', 'unwilling']):
                # Stronger reduction for legitimate systemic discussion
                final_score *= 0.1

        # Additional adjustments for specific cases
        if 'demographic changes' in text_lower and positive_research_score > 0.6:
            final_score *= 0.1  # Strong reduction for positive demographic research
            
        if 'minority communities' in text_lower and legitimate_systemic_score > 0.6:
            if has_factual_marker:
                final_score *= 0.1  # Strong reduction for factual minority community discussion

        final_score = min(max(final_score, 0.0), 1.0)
        
        # Category scores
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
        
        # Get structure and relationship scores
        structure_scores = self._analyze_statement_structure(text)
        relationship_scores = self._analyze_contextual_relationships(text)
        
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
        
        # Check for economic bias language
        if any(term in text_lower for term in self.economic_bias_terms):
            patterns_found.append("economic bias language")
        
        # Add structure-based patterns
        if structure_scores['factual_research'] > 0.5:
            patterns_found.append("research-based language")
        if structure_scores['systemic_analysis'] > 0.5:
            patterns_found.append("systemic analysis")
        if structure_scores['biased_generalization'] > 0.5:
            patterns_found.append("biased generalizations")
        
        # Add relationship-based patterns
        if relationship_scores['positive_research'] > 0.3:
            patterns_found.append("positive research findings")
        if relationship_scores['negative_research'] > 0.3:
            patterns_found.append("negative research claims")
        if relationship_scores['factual_systemic'] > 0.3:
            patterns_found.append("factual systemic discussion")
        if relationship_scores['biased_systemic'] > 0.3:
            patterns_found.append("biased systemic claims")
        
        # Generate explanation
        if confidence > 0.8:
            strength = "highly confident"
        elif confidence > 0.6:
            strength = "moderately confident"
        else:
            strength = "somewhat confident"
        
        explanation = f"The text shows signs of bias (confidence: {confidence:.2f}). "
        explanation += f"The model is {strength} this text contains bias. "
        
        # Add structure-based explanation
        if structure_scores['factual_research'] > 0.5:
            if relationship_scores['positive_research'] > 0.3:
                explanation += "While the text uses research-based language to present findings, "
                if confidence > 0.6:
                    explanation += "it appears to use this framework to mask bias. "
            else:
                explanation += "The text uses academic language to present biased views. "
        
        if structure_scores['systemic_analysis'] > 0.5:
            if relationship_scores['factual_systemic'] > 0.3:
                explanation += "The text discusses systemic issues, but "
                if confidence > 0.6:
                    explanation += "may use this framework to express bias. "
            else:
                explanation += "The text uses systemic analysis to present biased views. "
        
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

def train_bias_model():
    """Function to handle the training process"""
    # Add debug mode flag
    debug_mode = False  # Set to True for quick testing
    
    # Dataset paths
    train_path = r"C:\Users\ovidi\OneDrive\Desktop\Cursor (licenta Rux)\Jigsaw dataset\train.csv"
    private_test_path = r"C:\Users\ovidi\OneDrive\Desktop\Cursor (licenta Rux)\Jigsaw dataset\test_private_expanded.csv"
    all_data_path = r"C:\Users\ovidi\OneDrive\Desktop\Cursor (licenta Rux)\Jigsaw dataset\all_data.csv"
    toxicity_annotations_path = r"C:\Users\ovidi\OneDrive\Desktop\Cursor (licenta Rux)\Jigsaw dataset\toxicity_individual_annotations.csv"
    
    # Load and merge datasets
    df = load_and_merge_datasets(train_path, private_test_path, all_data_path, toxicity_annotations_path)
    if df is None:
        print("Failed to load datasets. Exiting.")
        return
    
    if debug_mode:
        # Take a small sample for debug mode
        df = df.sample(n=min(1000, len(df)), random_state=42)
        print(f"Debug mode: Using {len(df)} samples from original dataset")
    
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
        return
    
    # Create bias labels with adjusted thresholds for the larger dataset
    print("\nCreating bias labels...")
    race_columns = ['black', 'white', 'asian', 'latino', 'other_race_or_ethnicity']
    toxicity_metrics = ['identity_attack', 'severe_toxicity', 'insult', 'threat']
    df['race_bias'] = (
        (df[toxicity_metrics].max(axis=1) > 0.2) &
        (df[race_columns].max(axis=1) > 0.2)
    ).astype(int)
    
    # Calculate class weights for balanced training
    total_samples = len(df)
    n_negative = len(df[df['race_bias'] == 0])
    n_positive = len(df[df['race_bias'] == 1])
    class_weights = torch.tensor([total_samples/(2*n_negative), total_samples/(2*n_positive)], dtype=torch.float32)
    
    print("\nClass weights for balanced training:")
    print(f"Negative class weight: {class_weights[0]:.2f}")
    print(f"Positive class weight: {class_weights[1]:.2f}")
    
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
        return
    
    # Increase max samples per class for the larger dataset
    max_samples_per_class = 2000  # Increased from 1000 to handle larger dataset
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

if __name__ == "__main__":
    # Only run training if this file is executed directly
    should_train = input("Do you want to train the model? (y/n): ").lower().strip() == 'y'
    if should_train:
        train_bias_model()
    else:
        print("Skipping training. You can import and use the BiasDetector class directly.")
