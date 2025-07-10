"""
Med42 Client - Specialized Medical AI Integration
Provides medical analysis using the Med42-8B model optimized for healthcare
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Any, Optional
import config

class Med42Client:
    """Med42-8B model client for specialized medical analysis"""
    
    def __init__(self):
        """Initialize Med42-8B model"""
        print("Инициализация Med42-8B модели...")
        try:
            self.device = self._get_device()
            print(f"Используется устройство: {self.device}")
            
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(config.MED42_MODEL_PATH)
            
            # Set pad token if not exists
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
            # Use appropriate dtype based on device
            if self.device == "cuda":
                torch_dtype = torch.bfloat16
            elif self.device == "mps":
                torch_dtype = torch.float16
            else:
                torch_dtype = torch.float32
            
            self.model = AutoModelForCausalLM.from_pretrained(
                config.MED42_MODEL_PATH,
                torch_dtype=torch_dtype,
                device_map="auto" if config.MED42_DEVICE == "auto" else None,
                trust_remote_code=True
            )
            
            # Get model's max length
            self.max_length = getattr(self.model.config, 'max_position_embeddings', 2048)
            print(f"Максимальная длина модели: {self.max_length} токенов")
            
            if config.MED42_DEVICE != "auto":
                self.model = self.model.to(self.device)
            
            print("Med42-8B модель загружена успешно")
            
        except Exception as e:
            print(f"Ошибка загрузки Med42-8B модели: {e}")
            raise
    
    def _get_device(self) -> str:
        """Determine the best available device"""
        if config.MED42_DEVICE == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return config.MED42_DEVICE
    
    def analyze_images(self, images: List[Dict[str, Any]]) -> str:
        """
        Analyze medical data using Med42 model
        Note: Med42 is primarily text-based, so this analyzes image metadata and descriptions
        
        Args:
            images: List of processed image data
            
        Returns:
            Medical analysis text
        """
        print(f"Проведение медицинского анализа с помощью Med42...")
        
        try:
            # Create analysis prompt with image metadata
            prompt = self._create_analysis_prompt(images)
            
            # Generate analysis
            analysis = self._generate_analysis(prompt)
            
            # Log the full response
            print("=" * 50)
            print("🔍 ПОЛНЫЙ ОТВЕТ MED42:")
            print(analysis)
            print("=" * 50)
            
            print("Med42 анализ завершён")
            return analysis
            
        except Exception as e:
            print(f"Ошибка Med42 анализа: {e}")
            return f"Error during Med42 analysis: {str(e)}"
    
    def _create_analysis_prompt(self, images: List[Dict[str, Any]]) -> str:
        """Create analysis prompt with image information"""
        # Extract metadata information
        image_info = []
        for i, img_data in enumerate(images):
            metadata = img_data.get('metadata', {})
            info = f"Image {i+1}: "
            
            if 'SliceLocation' in metadata:
                info += f"Slice Location: {metadata['SliceLocation']}mm, "
            if 'SliceThickness' in metadata:
                info += f"Thickness: {metadata['SliceThickness']}mm, "
            if 'SeriesDescription' in metadata:
                info += f"Series: {metadata['SeriesDescription']}, "
            
            image_info.append(info.rstrip(', '))
        
        # Combine with analysis prompt
        prompt = f"""As a medical AI specialist, analyze this CT study data:

STUDY INFORMATION:
- Total images: {len(images)}
- Enhanced anatomical coverage with strategic selection
- Comprehensive abdominal organ evaluation
- Image details:
{chr(10).join(image_info)}

{config.MED42_INITIAL_PROMPT}

Please provide comprehensive medical analysis of the available data."""
        
        return prompt
    
    def _truncate_prompt(self, prompt: str, max_tokens: int = 1500) -> str:
        """Truncate prompt to fit within token limits"""
        tokens = self.tokenizer.encode(prompt)
        if len(tokens) > max_tokens:
            # Keep the beginning and end of the prompt
            keep_start = max_tokens // 2
            keep_end = max_tokens - keep_start - 50  # Reserve space for template
            
            start_tokens = tokens[:keep_start]
            end_tokens = tokens[-keep_end:]
            
            truncated_tokens = start_tokens + [self.tokenizer.eos_token_id] + end_tokens
            truncated_prompt = self.tokenizer.decode(truncated_tokens, skip_special_tokens=True)
            
            print(f"Промпт обрезан с {len(tokens)} до {len(truncated_tokens)} токенов")
            return truncated_prompt
        
        return prompt

    def _generate_analysis(self, prompt: str) -> str:
        """Generate medical analysis using Med42-8B model with chat template"""
        try:
            # Truncate prompt if too long
            truncated_prompt = self._truncate_prompt(prompt, max_tokens=1500)
            
            # Prepare messages for Med42 chat template
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a helpful, respectful and honest medical assistant. You are Med42 developed by the AI team at M42, UAE. "
                        "Always answer as helpfully as possible, while being safe. "
                        "Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. "
                        "Please ensure that your responses are socially unbiased and positive in nature. "
                        "If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. "
                        "If you don't know the answer to a question, please don't share false information."
                    ),
                },
                {"role": "user", "content": truncated_prompt},
            ]
            
            # Apply chat template
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            # Tokenize input with attention mask
            inputs = self.tokenizer(
                formatted_prompt, 
                return_tensors="pt", 
                padding=True,
                truncation=True,
                max_length=self.max_length - 512  # Reserve space for generation
            ).to(self.device)
            
            # Check input length
            input_length = inputs['input_ids'].shape[1]
            if input_length > self.max_length - 512:
                print(f"Предупреждение: входная последовательность ({input_length}) близка к лимиту модели")
            
            # Define stop tokens
            stop_tokens = [
                self.tokenizer.eos_token_id,
            ]
            
            # Try to get special token for llama
            try:
                eot_token = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
                if eot_token != self.tokenizer.unk_token_id:
                    stop_tokens.append(eot_token)
            except:
                pass
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=min(512, self.max_length - input_length),
                    temperature=0.4,
                    do_sample=True,
                    top_k=150,
                    top_p=0.75,
                    eos_token_id=stop_tokens,
                    pad_token_id=self.tokenizer.pad_token_id,
                    repetition_penalty=1.1,
                    use_cache=True
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract generated part (remove input prompt)
            generated_text = response[len(formatted_prompt):].strip()
            
            return generated_text if generated_text else "Conduct analysis of available data"
            
        except Exception as e:
            print(f"Ошибка генерации Med42-8B: {e}")
            return f"Error generating analysis: {str(e)}" 
    
    def analyze_ct_study(self, images: List[Dict[str, Any]], user_context: str = "") -> Optional[str]:
        """
        Analyze CT study with optional user context
        
        Args:
            images: List of processed image data
            user_context: Additional context from user (symptoms, age, etc.)
            
        Returns:
            Medical analysis text
        """
        print(f"📋 Med42 анализ CT исследования ({len(images)} изображений)")
        
        if user_context:
            print(f"📝 Дополнительный контекст: {user_context}")
        
        try:
            # Create enhanced analysis prompt with user context
            prompt = self._create_enhanced_analysis_prompt(images, user_context)
            
            # Generate analysis
            analysis = self._generate_analysis(prompt)
            
            # Log the full response
            print("=" * 50)
            print("🔍 ПОЛНЫЙ ОТВЕТ MED42:")
            print(analysis)
            print("=" * 50)
            
            print("✅ Med42 анализ завершён")
            return analysis
            
        except Exception as e:
            print(f"❌ Ошибка Med42 анализа: {e}")
            return None
    
    def _create_enhanced_analysis_prompt(self, images: List[Dict[str, Any]], user_context: str = "") -> str:
        """Create enhanced analysis prompt with user context"""
        # Extract metadata information
        image_info = []
        for i, img_data in enumerate(images):
            metadata = img_data.get('metadata', {})
            info = f"Image {i+1}: "
            
            if 'SliceLocation' in metadata:
                info += f"Slice Location: {metadata['SliceLocation']}mm, "
            if 'SliceThickness' in metadata:
                info += f"Thickness: {metadata['SliceThickness']}mm, "
            if 'SeriesDescription' in metadata:
                info += f"Series: {metadata['SeriesDescription']}, "
            
            image_info.append(info.rstrip(', '))
        
        # Build enhanced prompt with user context
        prompt = f"""As a medical AI specialist, analyze this CT study data:

STUDY INFORMATION:
- Total images: {len(images)}
- Enhanced anatomical coverage with strategic selection
- Comprehensive abdominal organ evaluation
- Image details:
{chr(10).join(image_info)}"""

        # Add user context if provided
        if user_context:
            prompt += f"""

ADDITIONAL CLINICAL CONTEXT:
{user_context}

Please incorporate this clinical information into your analysis."""

        prompt += f"""

{config.MED42_INITIAL_PROMPT}

Please provide comprehensive medical analysis considering all available data and context."""
        
        return prompt 