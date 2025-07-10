"""
Llama Vision Client - Local Vision Model Integration
Provides image analysis using Llama Vision via Ollama
"""

import requests
import json
import base64
from typing import List, Dict, Any, Optional
import config

class LlamaVisionClient:
    """Llama Vision client for image analysis via Ollama"""
    
    def __init__(self):
        """Initialize Llama Vision client"""
        self.base_url = config.OLLAMA_BASE_URL
        self.model = config.LLAMA_VISION_MODEL
        
        # Test connection to Ollama
        if not self._test_connection():
            raise ConnectionError("Cannot connect to Ollama. Please ensure Ollama is running.")
        
        print("Llama Vision клиент инициализирован")
    
    def _test_connection(self) -> bool:
        """Test connection to Ollama server"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def analyze_images(self, images: List[Dict[str, Any]]) -> str:
        """
        Analyze images using Llama Vision
        
        Args:
            images: List of processed image data
            
        Returns:
            Vision analysis text
        """
        print(f"Анализ {len(images)} изображений с помощью Llama Vision...")
        
        try:
            # Select strategic images for analysis
            selected_images = self._select_strategic_images(images)
            
            # Analyze images in batches
            analyses = []
            for i, img_data in enumerate(selected_images):
                print(f"Анализ изображения {i+1}/{len(selected_images)}...")
                analysis = self._analyze_single_image(img_data, i)
                if analysis:
                    analyses.append(f"Image {i+1} Analysis:\n{analysis}")
            
            # Combine analyses
            combined_analysis = "\n\n".join(analyses)
            
            print("Llama Vision анализ завершён")
            return combined_analysis
            
        except Exception as e:
            print(f"Ошибка Llama Vision анализа: {e}")
            return f"Error during Llama Vision analysis: {str(e)}"
    
    def _select_strategic_images(self, images: List[Dict[str, Any]], max_images: int = 8) -> List[Dict[str, Any]]:
        """Select most representative images for analysis"""
        if len(images) <= max_images:
            return images
        
        # Select evenly distributed images
        step = len(images) / max_images
        selected_indices = [int(i * step) for i in range(max_images)]
        
        return [images[i] for i in selected_indices]
    
    def _analyze_single_image(self, image_data: Dict[str, Any], index: int) -> Optional[str]:
        """Analyze a single image using Llama Vision"""
        try:
            # Prepare request
            payload = {
                "model": self.model,
                "prompt": config.LLAMA_VISION_INITIAL_PROMPT,
                "images": [image_data['base64_image']],
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9
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
                return result.get('response', '')
            else:
                print(f"Ошибка API Ollama: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"Ошибка анализа изображения {index}: {e}")
            return None
    
    def analyze_with_followup(self, images: List[Dict[str, Any]]) -> Dict[str, str]:
        """
        Perform initial analysis followed by detailed followup
        
        Returns:
            Dictionary with 'initial' and 'followup' analyses
        """
        print("Выполнение анализа с последующим детальным изучением...")
        
        # Initial analysis
        initial_analysis = self.analyze_images(images)
        
        # Followup analysis with different prompt
        followup_analysis = self._perform_followup_analysis(images)
        
        return {
            'initial': initial_analysis,
            'followup': followup_analysis
        }
    
    def _perform_followup_analysis(self, images: List[Dict[str, Any]]) -> str:
        """Perform followup analysis with detailed prompt"""
        try:
            # Select different images for followup
            selected_images = self._select_strategic_images(images, max_images=4)
            
            analyses = []
            for i, img_data in enumerate(selected_images):
                analysis = self._analyze_with_prompt(img_data, config.LLAMA_VISION_FOLLOWUP_PROMPT, i)
                if analysis:
                    analyses.append(analysis)
            
            return "\n\n".join(analyses)
            
        except Exception as e:
            print(f"Ошибка детального анализа: {e}")
            return f"Error during followup analysis: {str(e)}"
    
    def _analyze_with_prompt(self, image_data: Dict[str, Any], prompt: str, index: int) -> Optional[str]:
        """Analyze image with custom prompt"""
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "images": [image_data['base64_image']],
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9
                }
            }
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=180
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', '')
            else:
                return None
                
        except Exception as e:
            print(f"Ошибка анализа с пользовательским промптом {index}: {e}")
            return None 