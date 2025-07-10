# CT Reader - Advanced Medical Image Analysis System

## Overview

CT Reader is an advanced medical image analysis system that uses specialized AI models to analyze CT scans. The system now supports processing **ALL images** in a study with optimized memory management and parallel processing capabilities.

## Key Features

- **Complete Image Analysis**: Process all images in a CT study (not just a subset)
- **Batch Processing**: Intelligent batching to manage memory usage
- **Parallel Processing**: Optimized for performance where possible
- **Memory Management**: Automatic cleanup to prevent memory issues
- **Multiple AI Models**: MedGemma, Med42, and Comprehensive analysis modes
- **Progress Tracking**: Beautiful progress bars and detailed logging
- **Configurable Settings**: Customize batch size and image limits

## Quick Start

### 1. Analyze ALL Images (Recommended)

```bash
python analyze_all.py
```

This script will:
- Process **ALL** DICOM files in the `input/` directory
- Use MedGemma model for medical analysis
- Apply optimized batch processing
- Show detailed progress and results

### 2. Interactive Analysis with Custom Settings

```bash
python main.py
```

This allows you to:
- Choose how many images to process (or all)
- Set batch size for memory management
- Select analysis mode (MedGemma, Med42, Comprehensive)
- Add custom medical context

## Configuration Options

### Image Processing Settings

- **Max Images**: 
  - `None` = Process ALL images (recommended)
  - `Number` = Limit to specific count
  
- **Batch Size**: 
  - `3-5` = Conservative (less memory)
  - `5-10` = Balanced (recommended)
  - `10+` = Aggressive (more memory, faster)

- **Parallel Processing**: 
  - Always enabled for optimal performance

### Analysis Modes

1. **MedGemma** (Recommended)
   - Google's specialized medical AI model
   - Direct medical image analysis
   - Best for diagnostic insights

2. **Med42**
   - Specialized medical AI model
   - Fast analysis
   - Good for general medical assessment

3. **Comprehensive**
   - Full analysis of all images
   - Detailed contextual information
   - Most thorough analysis

## System Requirements

### Memory Requirements

- **Small Studies** (< 100 images): 8GB RAM minimum
- **Medium Studies** (100-500 images): 16GB RAM recommended
- **Large Studies** (500+ images): 32GB RAM recommended

### GPU Support

- **NVIDIA GPU**: CUDA support for faster processing
- **Apple Silicon**: MPS support for M1/M2 Macs
- **CPU Only**: Supported but slower

## Performance Optimization

### For Large Studies (1000+ images)

1. **Adjust Batch Size**:
   ```python
   # Conservative approach
   batch_size = 3
   
   # Balanced approach
   batch_size = 5
   
   # Aggressive approach (if you have enough RAM)
   batch_size = 10
   ```

2. **Monitor Memory Usage**:
   - System automatically cleans up memory after each batch
   - If you encounter memory issues, reduce batch size

3. **Processing Time Estimates**:
   - ~30-60 seconds per image (GPU)
   - ~2-3 minutes per image (CPU)
   - 1000 images ≈ 8-50 hours depending on hardware

## File Structure

```
CT-reader/
├── input/                 # Place DICOM files here
│   ├── D0000
│   ├── D0001
│   └── ...
├── output/               # Analysis results and logs
├── main.py              # Interactive analysis
├── analyze_all.py       # Quick analysis of all images
├── ct_analyzer.py       # Core analysis engine
├── medgemma_analyzer.py # MedGemma integration
└── progress_logger.py   # Progress tracking
```

## Usage Examples

### Example 1: Analyze All Images with Default Settings

```bash
python analyze_all.py
```

### Example 2: Analyze First 100 Images with Custom Batch Size

```bash
python main.py
# Select: Max images = 100, Batch size = 8
```

### Example 3: Full Analysis with Medical Context

```bash
python main.py
# Select: All images, Add context: "65-year-old patient, chest pain, suspected pneumonia"
```

## Output and Logging

### Analysis Results

- **Console Output**: Real-time progress and results
- **Log Files**: Detailed logs in `output/` directory
- **Timestamped Logs**: Each run creates a new log file

### Log File Location

```
output/ct_analysis_YYYY-MM-DD_HH-MM-SS.log
```

## Troubleshooting

### Memory Issues

If you encounter out-of-memory errors:
1. Reduce batch size to 3 or less
2. Close other applications
3. Consider processing in smaller chunks

### Performance Issues

For slow processing:
1. Ensure GPU drivers are updated
2. Check if CUDA/MPS is properly configured
3. Consider using a smaller batch size if system is unstable

### Authentication Issues

For MedGemma access:
1. Ensure `.env` file contains valid `HUGGINGFACE_TOKEN`
2. Check token permissions for gated model access
3. Verify internet connection for model downloads

## Advanced Usage

### Custom Batch Processing

```python
from ct_analyzer import CTAnalyzer

# Custom configuration
analyzer = CTAnalyzer(
    max_images_for_medgemma=None,  # All images
    enable_parallel=True,
    batch_size=8
)

# Analyze with custom settings
result = analyzer.analyze_directory("input", mode="medgemma")
```

### Memory-Optimized Processing

```python
# For very large studies
analyzer = CTAnalyzer(
    max_images_for_medgemma=None,
    enable_parallel=True,
    batch_size=3  # Very conservative
)
```

## Support

For issues or questions:
1. Check the log files in `output/` directory
2. Verify system requirements
3. Ensure all dependencies are installed
4. Check GPU/CUDA configuration if using GPU acceleration

## License

This project is for educational and research purposes. Please ensure compliance with medical data regulations and AI model licenses. 