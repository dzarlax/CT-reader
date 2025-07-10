# CT Reader - Advanced Medical Image Analysis System

A comprehensive CT scan analysis system with multiple AI-powered analysis modes and intelligent image processing capabilities.

## Features

### 🔬 Multiple Analysis Modes
- **Med42**: Specialized medical AI model optimized for healthcare applications  
- **Hybrid Mode**: Combines Llama Vision for image analysis with Med42 for medical interpretation

### 🎯 Intelligent Processing
- **Strategic Image Selection**: Automatically selects the most diagnostically relevant images
- **Token Optimization**: Efficient processing to maximize analysis quality within API limits
- **Comprehensive Reporting**: Detailed medical reports with findings and recommendations

### 🏥 Medical Focus
- Anatomical structure identification
- Pathological findings detection
- System-based analysis (respiratory, cardiovascular, GI, genitourinary, musculoskeletal)
- Differential diagnosis suggestions
- Clinical recommendations

## Installation

1. Clone the repository:
```cmd
git clone <repository-url>
cd CT-reader
```

2. Install dependencies:
```cmd
pip install -r requirements.txt
```

3. Configure Ollama settings in `config.py`

## Quick Start

### Basic Analysis
```cmd
python main.py
```

### Analysis Modes Demo
```cmd
python demo.py
```

### Hybrid Mode Demo
```cmd
python demo_hybrid.py
```

## Configuration

Edit `config.py` to configure:
- Ollama settings and model paths
- Med42 model configuration
- Analysis parameters
- Image processing settings
- Output preferences

## File Structure

```
CT-reader/
├── main.py              # Main application entry point
├── ct_analyzer.py       # Core analysis orchestrator
├── config.py           # Configuration and prompts
├── image_processor.py   # DICOM image processing

├── med42_client.py      # Med42 model integration
├── llama_vision_client.py # Llama Vision integration
├── llama_med42_client.py  # Hybrid analysis client
├── demo.py             # Individual mode demonstrations
├── demo_hybrid.py      # Hybrid mode demonstration
├── requirements.txt    # Python dependencies
├── input/              # DICOM input directory
├── output/             # Analysis results
└── temp/               # Temporary processing files
```

## Usage Examples

### Med42 Analysis  
Specialized medical AI analysis optimized for healthcare applications using local models.

### Hybrid Analysis
Combines Llama Vision for detailed image analysis with Med42 for medical interpretation.

## Requirements

- Python 3.8+
- Ollama (for Llama Vision mode)
- Sufficient disk space for DICOM files and models
- GPU recommended for Med42 model

## Support

For issues and questions, please refer to the documentation or create an issue in the repository. 