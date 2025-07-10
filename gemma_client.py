"""
Gemma 3 Client - Alternative Vision and Text Analysis
Provides image analysis using Gemma 3 model through Ollama
"""

import requests
import json
import base64
from typing import List, Dict, Any, Optional
from PIL import Image
import io
import config

class GemmaClient:
    """Gemma 3 model client for image and text analysis"""
    
    def __init__(self):
        """Initialize Gemma 3 client"""
        print("Инициализация Gemma 3 клиента...")
        self.base_url = config.OLLAMA_BASE_URL
        self.model_name = config.GEMMA_MODEL
        
        # Test connection
        if self._test_connection():
            print("Gemma 3 клиент инициализирован успешно")
        else:
            raise ConnectionError("Не удалось подключиться к Ollama или загрузить Gemma 3")
    
    def _test_connection(self) -> bool:
        """Test connection to Ollama and Gemma 3 model"""
        try:
            # Check if Ollama is running
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            if response.status_code != 200:
                print(f"Ошибка подключения к Ollama: {response.status_code}")
                return False
            
            # Check if Gemma 3 model is available
            models = response.json().get('models', [])
            model_names = [model['name'] for model in models]
            
            if self.model_name not in model_names:
                print(f"Модель {self.model_name} не найдена. Доступные модели: {model_names}")
                return False
            
            # Test simple generation
            test_response = self._generate_text("Test connection", max_tokens=10)
            return bool(test_response and len(test_response) > 0)
            
        except Exception as e:
            print(f"Ошибка тестирования соединения с Gemma 3: {e}")
            return False
    
    def analyze_images(self, images: List[Dict[str, Any]]) -> str:
        """
        Analyze images using Gemma 3 Vision
        Gemma 3 is multimodal and can analyze images directly
        
        Args:
            images: List of processed image data
            
        Returns:
            Analysis text
        """
        import time
        start_time = time.time()
        print(f"Анализ изображений с помощью Gemma 3 Vision...")
        print(f"Обрабатывается {len(images)} изображений...")
        
        try:
            # Select strategic images for analysis (limit to avoid overload)
            selected_images = self._select_strategic_images(images, max_images=8)
            print(f"Выбрано {len(selected_images)} изображений для анализа")
            
            # Analyze images in batches
            analyses = []
            for i, img_data in enumerate(selected_images):
                print(f"Анализ изображения {i+1}/{len(selected_images)}...")
                analysis = self._analyze_single_image_vision(img_data, i)
                if analysis:
                    slice_info = img_data.get('metadata', {}).get('SliceLocation', 'Unknown')
                    analyses.append(f"Slice {slice_info} (Image {i+1}):\n{analysis}")
            
            # Combine analyses with comprehensive summary
            combined_analysis = self._create_comprehensive_summary(analyses, images)
            
            total_time = time.time() - start_time
            print(f"Gemma 3 Vision анализ завершён за {total_time:.1f}с")
            print(f"Размер результата: {len(combined_analysis)} символов")
            return combined_analysis
            
        except Exception as e:
            print(f"Ошибка анализа Gemma 3: {e}")
            return f"Error during Gemma 3 analysis: {str(e)}"
    
    def analyze_single_image(self, image_data: Dict[str, Any], context: str = "") -> str:
        """
        Analyze single image with Gemma 3
        
        Args:
            image_data: Single image data dictionary
            context: Additional context for analysis
            
        Returns:
            Analysis text
        """
        try:
            # Create focused prompt for single image
            prompt = self._create_single_image_prompt(image_data, context)
            
            # Generate analysis
            analysis = self._generate_text(prompt, max_tokens=1000)
            
            return analysis
            
        except Exception as e:
            print(f"Ошибка анализа изображения Gemma 3: {e}")
            return f"Error analyzing image: {str(e)}"
    
    def continue_analysis(self, previous_analysis: str, additional_context: str) -> str:
        """
        Continue analysis with additional context
        
        Args:
            previous_analysis: Previous analysis results
            additional_context: Additional context to consider
            
        Returns:
            Continued analysis
        """
        try:
            prompt = self._create_continuation_prompt(previous_analysis, additional_context)
            
            analysis = self._generate_text(prompt, max_tokens=1500)
            
            return analysis
            
        except Exception as e:
            print(f"Ошибка продолжения анализа Gemma 3: {e}")
            return f"Error continuing analysis: {str(e)}"
    
    def _select_strategic_images(self, images: List[Dict[str, Any]], max_images: int = 8) -> List[Dict[str, Any]]:
        """Select most representative images for analysis"""
        if len(images) <= max_images:
            return images
        
        # Select evenly distributed images across the study
        step = len(images) / max_images
        selected_indices = [int(i * step) for i in range(max_images)]
        
        return [images[i] for i in selected_indices]
    
    def _analyze_single_image_vision(self, image_data: Dict[str, Any], index: int) -> Optional[str]:
        """Analyze a single image using Gemma 3 Vision"""
        try:
            # Create comprehensive medical analysis prompt
            prompt = """As an experienced radiologist, analyze this CT scan image in detail:

ANALYSIS REQUIREMENTS:
1. Identify anatomical structures visible in the image
2. Assess normal anatomical features
3. Detect any pathological changes or abnormalities
4. Evaluate organ morphology, density, and relationships
5. Note any focal lesions, masses, or structural abnormalities
7. Assess vascular structures if visible
8. Provide clinical significance of findings

Provide a detailed medical analysis using appropriate radiological terminology."""
            
            # Prepare request payload
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "images": [image_data['base64_image']],
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "num_predict": 1000
                }
            }
            
            # Send request to Ollama
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=180
            )
            
            if response.status_code == 200:
                result = response.json()
                response_text = result.get('response', '').strip()
                
                # Log the full response
                print("=" * 50)
                print("🔍 ПОЛНЫЙ ОТВЕТ GEMMA (Vision):")
                print(response_text)
                print("=" * 50)
                
                return response_text
            else:
                print(f"Ошибка API Ollama: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"Ошибка анализа изображения {index}: {e}")
            return None
    
    def _create_comprehensive_summary(self, analyses: List[str], all_images: List[Dict[str, Any]]) -> str:
        """Create comprehensive summary from individual image analyses"""
        if not analyses:
            return "No successful image analyses completed."
        
        # Combine individual analyses
        individual_analyses = "\n\n=== INDIVIDUAL IMAGE ANALYSES ===\n\n" + "\n\n".join(analyses)
        
        # Create summary prompt
        summary_prompt = f"""Based on the following CT scan analyses, provide a comprehensive medical report:

{individual_analyses}

STUDY INFORMATION:
- Total images in study: {len(all_images)}
- Images analyzed: {len(analyses)}
- Strategic selection with focus on anatomical coverage

Please provide a COMPREHENSIVE MEDICAL SUMMARY including:

1. OVERALL ANATOMICAL ASSESSMENT:
   - Key anatomical structures identified
   - Spatial relationships and orientation
   
2. SYSTEMATIC ANALYSIS:
   - Respiratory system findings
   - Cardiovascular system findings  
   - Gastrointestinal system findings
   - Genitourinary system findings
   - Musculoskeletal findings

3. PATHOLOGICAL FINDINGS:
   - Summary of abnormal findings across all images
   - Assessment of lesions, masses, or structural abnormalities
   - Clinical significance of findings

4. ORGAN-SPECIFIC ANALYSIS:
- Detailed organ assessment including size, morphology, and density
   - Presence of focal lesions or abnormalities
   - Relationship with adjacent structures
   - Clinical assessment

5. MEDICAL CONCLUSIONS:
   - Primary diagnostic impressions
   - Differential diagnosis considerations
   - Recommendations for further evaluation

Provide a detailed, professional radiological report."""
        
        try:
            # Generate comprehensive summary
            summary = self._generate_text_only(summary_prompt, max_tokens=2000)
            
            return f"""=== CT ANALYSIS REPORT ===

Patient Study Analysis
Analysis Mode: gemma (Vision)
Images Processed: {len(all_images)}
Images Analyzed: {len(analyses)}
Analysis Date: {self._get_current_timestamp()}

{summary}

=== END REPORT ==="""
            
        except Exception as e:
            print(f"Ошибка создания итогового отчета: {e}")
            return individual_analyses
    
    def _get_current_timestamp(self) -> str:
        """Get current timestamp"""
        from datetime import datetime
        return datetime.now().isoformat()

    def _create_analysis_prompt(self, images: List[Dict[str, Any]]) -> str:
        """Create comprehensive analysis prompt"""
        # Extract detailed metadata information
        image_details = []
        
        for i, img_data in enumerate(images):
            metadata = img_data.get('metadata', {})
            
            detail = f"\n--- IMAGE {i+1} ---\n"
            
            # Anatomical information
            if 'anatomical_region' in img_data:
                detail += f"Anatomical region: {img_data['anatomical_region']}\n"
            if 'region_priority' in img_data:
                detail += f"Priority: {img_data['region_priority']}\n"
            
            # DICOM metadata
            if 'SliceLocation' in metadata:
                detail += f"Slice location: {metadata['SliceLocation']}mm\n"
            if 'SliceThickness' in metadata:
                detail += f"Slice thickness: {metadata['SliceThickness']}mm\n"
            if 'SeriesDescription' in metadata:
                detail += f"Series description: {metadata['SeriesDescription']}\n"
            if 'StudyDescription' in metadata:
                detail += f"Study description: {metadata['StudyDescription']}\n"
            if 'BodyPartExamined' in metadata:
                detail += f"Body part examined: {metadata['BodyPartExamined']}\n"
            
            # Technical parameters
            if 'KVP' in metadata:
                detail += f"Voltage: {metadata['KVP']}kV\n"
            if 'ExposureTime' in metadata:
                detail += f"Exposure time: {metadata['ExposureTime']}ms\n"
            
            image_details.append(detail)
        
        # Create comprehensive prompt
        prompt = f"""As an experienced radiologist, analyze this CT study data:

STUDY INFORMATION:
- Total images: {len(images)}
- Strategic selection with enhanced anatomical coverage
- Comprehensive abdominal cavity evaluation

IMAGE DETAILS:
{''.join(image_details)}

COMPREHENSIVE MEDICAL ANALYSIS REQUIRED:

1. ANATOMICAL ORIENTATION:
   - Identification of anatomical regions
   - Spatial relationships between structures
   - Slice levels and anatomical landmarks

2. SYSTEMATIC ANALYSIS:
   - Respiratory system (lungs, pleura, airways)
   - Cardiovascular system (heart, vessels)
   - Gastrointestinal system (liver, pancreas, bowel)
   - Genitourinary system (kidneys, bladder)
   - Musculoskeletal system (bones, joints, soft tissues)

3. DETAILED ORGAN ASSESSMENT:
- Organ size, contours, and density patterns
   - Parenchymal density
   - Presence of focal lesions
   - Splenic vessel status
   - Relationships with adjacent organs

4. PATHOLOGICAL FINDINGS:
   - Detection of abnormal changes
   - Assessment of size, density, morphology
   - Clinical significance of findings

5. MEDICAL CONCLUSIONS:
   - Primary findings summary
   - Differential diagnosis considerations
   - Recommendations for further evaluation

Provide detailed systematic analysis using appropriate medical terminology."""
        
        return prompt
    
    def _create_single_image_prompt(self, image_data: Dict[str, Any], context: str) -> str:
        """Create prompt for single image analysis"""
        metadata = image_data.get('metadata', {})
        
        prompt = f"""Analyze this CT image as an experienced radiologist:

CONTEXT: {context}

IMAGE METADATA:
"""
        
        # Add available metadata
        for key, value in metadata.items():
            if key in ['SliceLocation', 'SliceThickness', 'SeriesDescription', 'BodyPartExamined']:
                prompt += f"- {key}: {value}\n"
        
        if 'anatomical_region' in image_data:
            prompt += f"- Anatomical region: {image_data['anatomical_region']}\n"
        
        prompt += """
ANALYSIS REQUIRED:
1. Identification of anatomical structures
2. Assessment of normal anatomy
3. Detection of pathological changes
4. Evaluation of density and morphology of structures
5. Preliminary conclusions

Provide detailed medical analysis of the image."""
        
        return prompt
    
    def _create_continuation_prompt(self, previous_analysis: str, additional_context: str) -> str:
        """Create prompt for analysis continuation"""
        return f"""Continue medical analysis considering additional information:

PREVIOUS ANALYSIS:
{previous_analysis}

ADDITIONAL CONTEXT:
{additional_context}

CONTINUE ANALYSIS WITH FOCUS ON:
1. Integration of new data with previous findings
2. Refinement of diagnostic considerations
3. Additional anatomical details
4. Final medical conclusions
5. Clinical recommendations

Provide comprehensive completed medical analysis."""
    
    def _generate_text(self, prompt: str, max_tokens: int = 1500) -> str:
        """Generate text using Gemma 3 model"""
        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": 0.7,
                    "top_k": 40,
                    "top_p": 0.9,
                    "repeat_penalty": 1.1
                }
            }
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=180
            )
            
            if response.status_code == 200:
                result = response.json()
                response_text = result.get('response', 'No response generated')
                
                # Log the full response
                print("=" * 50)
                print("🔍 ПОЛНЫЙ ОТВЕТ GEMMA (Text):")
                print(response_text)
                print("=" * 50)
                
                return response_text
            else:
                print(f"Ошибка API Gemma 3: {response.status_code}")
                return f"API Error: {response.status_code}"
                
        except Exception as e:
            print(f"Ошибка генерации текста Gemma 3: {e}")
            return f"Generation error: {str(e)}"
    
    def _generate_text_only(self, prompt: str, max_tokens: int = 1500) -> str:
        """Generate text-only response using Gemma 3 model (no images)"""
        return self._generate_text(prompt, max_tokens) 