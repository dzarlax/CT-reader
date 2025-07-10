# Configuration file for CT Reader

# Ollama Configuration
OLLAMA_BASE_URL = "http://localhost:11434"
LLAMA_VISION_MODEL = "llama3.2-vision:latest"
GEMMA_MODEL = "gemma3:4b"  # Alternative vision/text model

# Med42 Configuration - Using open 8B model
MED42_MODEL_PATH = "m42-health/Llama3-Med42-8B"  # Open-access clinical LLM
MED42_DEVICE = "auto"  # "auto", "cpu", or specific GPU like "cuda:0"

# Analysis Configuration
MAX_IMAGES_PER_ANALYSIS = 16
TOKEN_LIMIT = 2000
STRATEGIC_SELECTION_COUNT = 12  # Increased for better anatomical coverage

# Enhanced Strategic Selection for Anatomical Coverage
ANATOMICAL_REGIONS = {
    "brain": {"slice_range": (0.85, 1.0), "priority": "high", "min_images": 2},
    "chest_upper": {"slice_range": (0.65, 0.85), "priority": "high", "min_images": 2}, 
    "chest_lower": {"slice_range": (0.45, 0.65), "priority": "high", "min_images": 2},
    "abdomen_upper": {"slice_range": (0.25, 0.45), "priority": "critical", "min_images": 3},  # Liver, spleen, pancreas
    "abdomen_lower": {"slice_range": (0.1, 0.25), "priority": "medium", "min_images": 2},  # Pelvis, kidneys
    "extremities": {"slice_range": (0.0, 0.1), "priority": "low", "min_images": 1}
}

# Spleen-specific detection parameters
SPLEEN_DETECTION = {
    "enabled": True,
    "slice_range": (0.3, 0.5),  # Typical spleen location
    "min_images": 2,  # Ensure spleen coverage
    "keywords": ["spleen", "селезенка", "splenic"]
}

# Image Processing Configuration
TARGET_SIZE = (512, 512)
QUALITY = 85

# Analysis Prompts

MED42_INITIAL_PROMPT = """You are an experienced radiologist analyzing CT images. Please provide a comprehensive medical analysis including:

SPATIAL UNDERSTANDING:
- Anatomical orientation and positioning
- Slice level and anatomical landmarks
- Relationship between structures

ANATOMICAL SEQUENCES:
- Sequential analysis across multiple slices
- 3D spatial relationships
- Continuity assessment

SYSTEM-BASED ANALYSIS:
1. RESPIRATORY SYSTEM:
   - Lung parenchyma assessment
   - Airways evaluation
   - Pleural space analysis

2. CARDIOVASCULAR SYSTEM:
   - Heart size and morphology
   - Great vessels assessment
   - Vascular structures

3. GASTROINTESTINAL SYSTEM:
   - Abdominal organs evaluation
   - Bowel assessment
   - Hepatobiliary system

4. GENITOURINARY SYSTEM:
   - Kidney evaluation
   - Bladder assessment
   - Reproductive organs

5. MUSCULOSKELETAL SYSTEM:
   - Bone structure analysis
   - Joint assessment
   - Soft tissue evaluation

MEDICAL CONCLUSIONS:
- Primary findings summary
- Differential diagnosis considerations
- Clinical significance assessment
- Recommendations for further evaluation

Provide detailed, systematic analysis with medical terminology appropriate for clinical documentation."""

MED42_FOLLOWUP_PROMPT = """Continue the analysis with focus on:

SPATIAL CONTINUITY:
- Integration with previous slice findings
- 3D reconstruction considerations
- Anatomical progression patterns

ADDITIONAL FINDINGS:
- Previously unmentioned structures
- Subtle abnormalities
- Incidental findings

FINAL MEDICAL ASSESSMENT:
- Comprehensive findings summary
- Clinical correlation recommendations
- Priority of findings
- Suggested follow-up"""

LLAMA_VISION_INITIAL_PROMPT = """As an expert radiologist, analyze this CT image in detail:

ANATOMICAL IDENTIFICATION:
- Identify the anatomical region and orientation
- Describe visible anatomical structures
- Note the slice level and anatomical landmarks

PATHOLOGICAL FINDINGS:
- Identify any abnormal findings
- Describe pathological changes
- Assess density, size, and morphology

SYSTEM ANALYSIS:
- Respiratory system assessment
- Cardiovascular structures
- Abdominal/pelvic organs (if visible)
- Musculoskeletal elements

PRELIMINARY CONCLUSIONS:
- Key findings summary
- Clinical significance
- Areas requiring attention

Provide systematic, detailed analysis using appropriate medical terminology."""

LLAMA_VISION_FOLLOWUP_PROMPT = """Continue analysis focusing on:

ADDITIONAL ANATOMICAL DETAILS:
- Previously unmentioned structures
- Subtle anatomical variations
- Comparative assessment

COMPREHENSIVE FINDINGS:
- Integration with previous observations
- Overall anatomical assessment
- Final medical conclusions

Provide complete medical analysis summary."""

# Analysis Templates
ANALYSIS_REPORT_TEMPLATE = """
=== CT ANALYSIS REPORT ===

Patient Study Analysis
Analysis Mode: {mode}
Images Processed: {image_count}
Analysis Date: {timestamp}

{analysis_content}

=== END REPORT ===
""" 