"""
Llama Med42 Client - Hybrid Analysis System
Combines Llama Vision for image analysis with Med42 for medical interpretation
"""

import json
from typing import List, Dict, Any, Optional
from llama_vision_client import LlamaVisionClient
from med42_client import Med42Client
import config

class LlamaMed42Client:
    """Hybrid client combining Llama Vision and Med42 capabilities"""
    
    def __init__(self):
        """Initialize hybrid analysis system"""
        print("Инициализация гибридной системы анализа...")
        
        try:
            # Initialize Llama Vision client
            print("Загрузка Llama Vision...")
            self.llama_vision = LlamaVisionClient()
            
            # Initialize Med42 client
            print("Загрузка Med42...")
            self.med42 = Med42Client()
            
            print("Гибридная система инициализирована успешно")
            
        except Exception as e:
            print(f"Ошибка инициализации гибридной системы: {e}")
            raise
    
    def analyze_images(self, images: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Perform hybrid analysis using both Llama Vision and Med42
        
        Args:
            images: List of processed image data
            
        Returns:
            Combined analysis results
        """
        print("Запуск гибридного анализа...")
        
        try:
            # Stage 1: Image analysis with Llama Vision
            print("Этап 1: Анализ изображений с Llama Vision...")
            vision_results = self._perform_vision_analysis(images)
            
            # Stage 2: Medical interpretation with Med42
            print("Этап 2: Медицинская интерпретация с Med42...")
            medical_analysis = self._perform_medical_interpretation(vision_results, images)
            
            # Stage 3: Combine results
            print("Этап 3: Объединение результатов...")
            combined_analysis = self._combine_analyses(vision_results, medical_analysis)
            
            result = {
                'vision_analysis': vision_results,
                'medical_interpretation': medical_analysis,
                'combined_analysis': combined_analysis,
                'tokens_used': self._estimate_tokens_used(vision_results, medical_analysis),
                'analysis_stages': 2
            }
            
            print("Гибридный анализ завершён")
            return result
            
        except Exception as e:
            print(f"Ошибка гибридного анализа: {e}")
            return {
                'error': str(e),
                'combined_analysis': f"Error during hybrid analysis: {str(e)}"
            }
    
    def _perform_vision_analysis(self, images: List[Dict[str, Any]]) -> Dict[str, str]:
        """Perform detailed vision analysis using Llama Vision"""
        try:
            # Get detailed vision analysis prompt
            vision_prompt = self._get_vision_analysis_prompt()
            
            # Analyze strategic selection of images
            from image_processor import ImageProcessor
            processor = ImageProcessor()
            strategic_images = processor.select_strategic_images(images, count=8)
            
            # Analyze initial batch
            initial_analyses = []
            for i, img_data in enumerate(strategic_images[:4]):
                print(f"Анализ изображения {i+1}/4 (первая серия)...")
                analysis = self._analyze_single_image_detailed(img_data, vision_prompt)
                if analysis:
                    initial_analyses.append(f"Slice {img_data.get('index', i)+1}: {analysis}")
            
            # Analyze followup batch
            followup_analyses = []
            for i, img_data in enumerate(strategic_images[4:8]):
                print(f"Анализ изображения {i+1}/4 (вторая серия)...")
                followup_prompt = self._get_vision_followup_prompt()
                analysis = self._analyze_single_image_detailed(img_data, followup_prompt)
                if analysis:
                    followup_analyses.append(f"Slice {img_data.get('index', i+4)+1}: {analysis}")
            
            return {
                'initial_analysis': "\n\n".join(initial_analyses),
                'followup_analysis': "\n\n".join(followup_analyses)
            }
            
        except Exception as e:
            print(f"Ошибка анализа изображений: {e}")
            return {
                'initial_analysis': f"Error in vision analysis: {str(e)}",
                'followup_analysis': ""
            }
    
    def _get_vision_analysis_prompt(self) -> str:
        """Get detailed prompt for vision analysis"""
        return """As an expert radiologist, provide detailed analysis of this CT image:

ANATOMICAL ORIENTATION AND IDENTIFICATION:
- Identify the anatomical region (head, chest, abdomen, pelvis)
- Describe the slice orientation and level
- Identify key anatomical landmarks visible

MORPHOLOGICAL CHARACTERISTICS:
- Describe the shape, size, and contours of visible structures
- Note any asymmetries or anatomical variations
- Assess organ positioning and relationships

DENSITY CHARACTERISTICS:
- Describe tissue densities (hypodense, isodense, hyperdense)
- Identify areas of calcification, fat, fluid, or air
- Note contrast enhancement patterns if applicable

PATHOLOGICAL CHANGES:
- Identify any abnormal findings or lesions
- Describe size, location, and characteristics of abnormalities
- Assess for signs of inflammation, trauma, or neoplasia

TECHNICAL ASPECTS:
- Comment on image quality and artifacts
- Note any limitations in visualization
- Suggest optimal windowing for specific structures

Provide systematic, detailed observations using precise medical terminology."""
    
    def _get_vision_followup_prompt(self) -> str:
        """Get followup prompt for additional vision analysis"""
        return """Continue detailed radiological analysis focusing on:

ADDITIONAL ANATOMICAL DETAILS:
- Previously unmentioned anatomical structures
- Subtle anatomical relationships and spatial orientation
- Fine structural details and tissue interfaces

COMPARATIVE ASSESSMENT:
- Bilateral comparison of structures
- Symmetry assessment and anatomical variations
- Progressive changes across sequential images

COMPREHENSIVE INTEGRATION:
- Integration with findings from previous slices
- Overall anatomical and pathological assessment
- Complete description of the imaging study

Provide thorough supplementary analysis to complete the radiological evaluation."""
    
    def _analyze_single_image_detailed(self, image_data: Dict[str, Any], prompt: str) -> Optional[str]:
        """Analyze single image with detailed prompt"""
        try:
            import requests
            
            payload = {
                "model": config.LLAMA_VISION_MODEL,
                "prompt": prompt,
                "images": [image_data['base64_image']],
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "num_predict": 800
                }
            }
            
            response = requests.post(
                f"{config.OLLAMA_BASE_URL}/api/generate",
                json=payload,
                timeout=180
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', '').strip()
            
            return None
            
        except Exception as e:
            print(f"Ошибка детального анализа изображения: {e}")
            return None
    
    def _perform_medical_interpretation(self, vision_results: Dict[str, str], images: List[Dict[str, Any]]) -> str:
        """Perform medical interpretation using Med42"""
        try:
            # Create comprehensive prompt for Med42
            interpretation_prompt = self._get_med42_interpretation_prompt(vision_results, images)
            
            # Generate medical interpretation
            interpretation = self.med42._generate_analysis(interpretation_prompt)
            
            return interpretation
            
        except Exception as e:
            print(f"Ошибка медицинской интерпретации: {e}")
            return f"Error in medical interpretation: {str(e)}"
    
    def _get_med42_interpretation_prompt(self, vision_results: Dict[str, str], images: List[Dict[str, Any]]) -> str:
        """Create Med42 interpretation prompt based on vision analysis"""
        prompt = f"""As a medical AI specialist, provide comprehensive medical interpretation based on the following radiological image analysis:

IMAGING FINDINGS:
{vision_results.get('initial_analysis', '')}

ADDITIONAL FINDINGS:
{vision_results.get('followup_analysis', '')}

STUDY INFORMATION:
- Total images analyzed: {len(images)}
- Analysis method: Hybrid (Llama Vision + Med42)

Please provide:

FINDINGS INTERPRETATION:
- Clinical significance of identified findings
- Pathophysiological correlations
- Anatomical and functional implications

DIFFERENTIAL DIAGNOSIS:
- Most likely diagnoses based on findings
- Alternative diagnostic considerations
- Additional findings that would support or exclude diagnoses

SYSTEM ASSESSMENT:
- Respiratory system evaluation (if applicable)
- Cardiovascular assessment (if applicable)
- Gastrointestinal/genitourinary evaluation (if applicable)
- Musculoskeletal assessment (if applicable)

CLINICAL RECOMMENDATIONS:
- Suggested follow-up imaging or studies
- Clinical correlation recommendations
- Priority of findings and urgency assessment
- Relevant clinical guidelines or protocols

Provide comprehensive medical analysis with appropriate clinical terminology and evidence-based recommendations."""
        
        return prompt
    
    def _combine_analyses(self, vision_results: Dict[str, str], medical_interpretation: str) -> str:
        """Combine vision and medical analyses into final report"""
        combined = f"""COMPREHENSIVE CT ANALYSIS REPORT

=== IMAGING ANALYSIS ===
{vision_results.get('initial_analysis', '')}

=== ADDITIONAL FINDINGS ===
{vision_results.get('followup_analysis', '')}

=== MEDICAL INTERPRETATION ===
{medical_interpretation}

=== ANALYSIS SUMMARY ===
This comprehensive analysis combines advanced computer vision analysis with specialized medical AI interpretation to provide detailed evaluation of the CT study. The hybrid approach ensures both accurate anatomical identification and clinically relevant medical assessment.

Analysis completed using Llama Vision for detailed image analysis and Med42 for medical interpretation."""
        
        return combined
    
    def _estimate_tokens_used(self, vision_results: Dict[str, str], medical_interpretation: str) -> int:
        """Estimate total tokens used in analysis"""
        total_text = ""
        for result in vision_results.values():
            total_text += result
        total_text += medical_interpretation
        
        # Rough estimate: 1 token ≈ 4 characters
        return len(total_text) // 4 