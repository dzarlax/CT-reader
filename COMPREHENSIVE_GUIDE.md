# Comprehensive CT Analysis Guide

## Overview

The Comprehensive Analyzer processes **ALL** images in a CT study with context preservation and full logging. This mode is designed for thorough analysis without time constraints.

## Key Features

### 🔍 **Complete Analysis**
- Analyzes ALL images in the study (not just selected samples)
- No image limit - processes entire dataset

### 💾 **Context Preservation**
- Maintains context between images
- Each image analysis builds on previous findings
- Context saved to persistent files for history

### 📦 **Batch Processing**
- Processes images in batches of 10
- Saves progress after each batch
- Can resume if interrupted

### 📋 **Full Logging**
- Complete logging of all AI responses (Gemma and Med42)
- Detailed console output showing exact AI responses
- All responses marked with clear headers

### 🗂️ **Session Management**
- Each analysis creates a unique session
- Context files preserved permanently
- Can review previous sessions

## Usage

### Quick Start
```bash
python demo.py
# Select option 5: "🔍 Полный анализ (ВСЕ изображения с контекстом)"
```

### Direct Usage
```python
from ct_analyzer import CTAnalyzer

analyzer = CTAnalyzer()
result = analyzer.analyze_directory("input", mode="comprehensive")
```

### Session Management
```python
from comprehensive_analyzer import ComprehensiveAnalyzer

# List all sessions
analyzer = ComprehensiveAnalyzer()
sessions = analyzer.list_sessions()

# Load specific session
session_data = analyzer.load_session("20250109_143022")
```

## Output Structure

### Context Files
- Location: `context/session_YYYYMMDD_HHMMSS.json`
- Contains: Complete analysis history, progress, findings
- **Never deleted** - permanent record

### Console Logging
```
==================================================
🔍 ПОЛНЫЙ ОТВЕТ GEMMA (Изображение 1):
[Complete AI response here]
==================================================
```

### Final Report
- Comprehensive summary of all findings
- Anatomical regions identified
- Pathological and normal findings
- Clinical assessment

## Analysis Process

1. **Session Initialization**
   - Creates unique session ID
   - Sets up context file
   - Initializes progress tracking

2. **Batch Processing**
   - Groups images into batches of 10
   - Each image analyzed with full context
   - Progress saved after each batch

3. **Context Building**
   - Each analysis incorporates previous findings
   - Context truncated if too long (keeps recent 1500 chars)
   - Cumulative knowledge builds throughout study

4. **Final Report Generation**
   - Synthesizes all findings
   - Creates comprehensive medical report
   - Saves complete session data

## Technical Details

### Models Used
- **Primary**: Gemma 2 27B (medical analysis with vision)
- **Fallback**: Med42 70B (specialized medical model)

### Memory Management
- Context limited to 2000 characters to prevent overflow
- Automatic truncation preserves most recent findings
- Full history preserved in session files

### Error Handling
- Graceful handling of individual image failures
- Progress preservation on interruption
- Detailed error logging

## Comparison with Other Modes

| Feature | Med42 | Hybrid | Gemma | Intelligent | **Comprehensive** |
|---------|-------|--------|-------|-------------|-------------------|
| Images Analyzed | ~8-12 | ~8-12 | ~8-12 | ~15-20 | **ALL** |
| Context Preservation | No | No | No | Limited | **Full** |
| Progress Saving | No | No | No | No | **Yes** |
| Session History | No | No | No | No | **Yes** |
| Full Logging | No | No | Limited | No | **Complete** |

## Best Practices

1. **For Large Studies**: Use comprehensive mode for complete analysis
2. **For Quick Assessment**: Use intelligent or gemma modes
3. **For Research**: Comprehensive mode provides complete dataset
4. **For Clinical Review**: Context preservation helps with continuity

## Troubleshooting

### If Analysis Stops
- Check `context/` directory for session file
- Session data is preserved - analysis can be resumed
- Review console logs for specific errors

### Memory Issues
- Context automatically managed
- Large studies may take significant time
- Monitor disk space for context files

### Model Availability
- Requires Ollama with Gemma 2 27B
- Falls back gracefully if models unavailable
- Check model status with `ollama list` 