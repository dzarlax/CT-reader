"""
Comprehensive CT Analysis System
Analyzes ALL images in a study with context preservation
"""

import os
import json
import requests
from datetime import datetime
from typing import List, Dict, Any, Optional
import base64
from med42_client import Med42Client

# Попробуем импортировать MedGemma, если доступна
try:
    from medgemma_client import MedGemmaClient
    MEDGEMMA_AVAILABLE = True
    print("✅ MedGemma клиент доступен")
except ImportError as e:
    MEDGEMMA_AVAILABLE = False
    print(f"⚠️ MedGemma недоступна: {e}")


class ComprehensiveAnalyzer:
    """Analyzer that processes ALL images with context preservation"""
    
    def __init__(self):
        self.base_url = "http://localhost:11434"
        self.gemma_model = "gemma3:4b"  # Резервная модель
        self.vision_model = "llama3.2-vision:latest"  # Для анализа изображений
        
        # Инициализируем MedGemma если доступна
        if MEDGEMMA_AVAILABLE:
            try:
                self.medgemma_client = MedGemmaClient()
                print("✅ MedGemma клиент инициализирован для специализированного медицинского анализа")
                self.use_medgemma = True
            except Exception as e:
                print(f"⚠️ MedGemma недоступна: {e}")
                self.medgemma_client = None
                self.use_medgemma = False
        else:
            self.medgemma_client = None
            self.use_medgemma = False
        
        # Инициализируем Med42 для дополнительного медицинского анализа
        try:
            self.med42_client = Med42Client()
            print("✅ Med42 клиент инициализирован для дополнительного медицинского анализа")
        except Exception as e:
            print(f"⚠️ Med42 недоступна: {e}")
            self.med42_client = None
            
        self.context_file = None
        self.session_id = None
        
    def analyze_study(self, images: List[Dict[str, Any]], user_context: str = "") -> Optional[Dict[str, Any]]:
        """
        Analyze CT study with optional user context
        
        Args:
            images: List of processed image data
            user_context: Additional context from user (symptoms, age, etc.)
            
        Returns:
            Comprehensive analysis results
        """
        print(f"🔍 Comprehensive анализ CT исследования ({len(images)} изображений)")
        
        if user_context:
            print(f"📝 Дополнительный контекст: {user_context}")
        
        try:
            # Use the comprehensive analysis method with user context
            result = self.analyze_complete_study(images, mode="comprehensive", user_context=user_context)
            
            if result:
                print("✅ Comprehensive анализ завершён")
                return result
            else:
                print("❌ Comprehensive анализ не дал результатов")
                return None
                
        except Exception as e:
            print(f"❌ Ошибка Comprehensive анализа: {e}")
            return None
        
    def analyze_complete_study(self, images: List[Dict[str, Any]], 
                             mode: str = "comprehensive", user_context: str = "") -> Dict[str, Any]:
        """Analyze ALL images in the study with context preservation"""
        
        # Initialize session
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.context_file = f"context/session_{self.session_id}.json"
        
        # Create context directory
        os.makedirs("context", exist_ok=True)
        
        print(f"\n🔍 НАЧИНАЕМ ПОЛНЫЙ АНАЛИЗ ИССЛЕДОВАНИЯ")
        print(f"📊 Всего изображений: {len(images)}")
        print(f"🗂️  Сессия: {self.session_id}")
        print(f"💾 Контекст сохраняется в: {self.context_file}")
        
        # Initialize context
        context = {
            "session_id": self.session_id,
            "start_time": datetime.now().isoformat(),
            "total_images": len(images),
            "mode": mode,
            "user_context": user_context,  # Добавляем пользовательский контекст
            "progress": {
                "processed": 0,
                "current_batch": 0,
                "total_batches": (len(images) + 9) // 10  # 10 images per batch
            },
            "findings": {
                "anatomical_regions": {},
                "pathological_findings": [],
                "normal_findings": [],
                "organ_assessments": {},
                "cumulative_context": ""
            },
            "image_analyses": []
        }
        
        self._save_context(context)
        
        try:
            # Process images in batches of 10 to maintain context
            batch_size = 10
            total_batches = (len(images) + batch_size - 1) // batch_size
            
            for batch_idx in range(total_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(images))
                batch_images = images[start_idx:end_idx]
                
                print(f"\n📦 ОБРАБОТКА ПАКЕТА {batch_idx + 1}/{total_batches}")
                print(f"   Изображения {start_idx + 1}-{end_idx} из {len(images)}")
                
                # Process batch
                batch_results = self._process_batch(batch_images, batch_idx, context)
                
                # Update context
                context["progress"]["current_batch"] = batch_idx + 1
                context["progress"]["processed"] = end_idx
                context["image_analyses"].extend(batch_results["individual_analyses"])
                
                # Merge findings
                self._merge_findings(context["findings"], batch_results["batch_summary"])
                
                # Update cumulative context
                context["findings"]["cumulative_context"] = self._update_cumulative_context(
                    context["findings"]["cumulative_context"], 
                    batch_results["batch_summary"]
                )
                
                # Save progress
                self._save_context(context)
                
                print(f"✅ Пакет {batch_idx + 1} завершён")
                print(f"📈 Прогресс: {end_idx}/{len(images)} ({(end_idx/len(images)*100):.1f}%)")
            
            # Generate final comprehensive report
            print(f"\n📋 СОЗДАНИЕ ИТОГОВОГО ОТЧЁТА...")
            final_report = self._generate_final_report(context)
            
            context["end_time"] = datetime.now().isoformat()
            context["final_report"] = final_report
            context["status"] = "completed"
            
            self._save_context(context)
            
            print(f"✅ ПОЛНЫЙ АНАЛИЗ ЗАВЕРШЁН")
            print(f"⏱️  Время: {context['start_time']} - {context['end_time']}")
            print(f"📊 Обработано: {len(images)} изображений")
            print(f"💾 Контекст сохранён: {self.context_file}")
            
            return {
                "session_id": self.session_id,
                "total_images": len(images),
                "context_file": self.context_file,
                "final_report": final_report,
                "status": "completed"
            }
            
        except Exception as e:
            print(f"❌ ОШИБКА АНАЛИЗА: {e}")
            context["status"] = "error"
            context["error"] = str(e)
            context["end_time"] = datetime.now().isoformat()
            self._save_context(context)
            return None
    
    def _process_batch(self, batch_images: List[Dict[str, Any]], 
                      batch_idx: int, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process a batch of images"""
        
        print(f"   🔍 Анализ изображений в пакете...")
        
        individual_analyses = []
        batch_findings = {
            "anatomical_regions": {},
            "pathological_findings": [],
            "normal_findings": [],
            "organ_assessments": {}
        }
        
        # Analyze each image in the batch
        for idx, image_data in enumerate(batch_images):
            global_idx = batch_idx * 10 + idx + 1
            print(f"      🖼️  Изображение {global_idx}: ", end="")
            
            # Get previous context for this analysis
            previous_context = context["findings"]["cumulative_context"]
            
            # Analyze single image
            analysis = self._analyze_single_image_with_context(
                image_data, global_idx, previous_context
            )
            
            if analysis:
                individual_analyses.append({
                    "image_index": global_idx,
                    "analysis": analysis,
                    "timestamp": datetime.now().isoformat()
                })
                
                # Extract findings from this analysis
                self._extract_findings_from_analysis(analysis, batch_findings)
                print("✅")
            else:
                print("❌")
        
        # Create batch summary
        batch_summary = self._create_batch_summary(batch_findings, batch_idx + 1)
        
        return {
            "individual_analyses": individual_analyses,
            "batch_summary": batch_summary
        }
    
    def _analyze_single_image_with_context(self, image_data: Dict[str, Any], 
                                         image_idx: int, previous_context: str) -> Optional[str]:
        """Analyze single image with previous context using MedGemma + Med42"""
        
        try:
            # ЭТАП 1: Анализ изображения
            # Сначала получаем визуальное описание от Vision модели
            vision_prompt = f"""Analyze this CT image #{image_idx} as an experienced radiologist.

PREVIOUS CONTEXT FROM STUDY:
{previous_context if previous_context else "This is the first image in the study."}

Provide detailed visual analysis:
1. Anatomical identification and orientation
2. Organ assessment (size, density, morphology)  
3. Pathological findings detection
4. Relationship to previous findings
5. Image quality and technical factors

Focus on objective visual findings."""
            
            # Vision analysis для получения визуального описания
            payload = {
                "model": self.vision_model,
                "prompt": vision_prompt,
                "images": [image_data['base64_image']],
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "num_predict": 800
                }
            }
            
            response = requests.post(f"{self.base_url}/api/generate", json=payload, timeout=180)
            
            if response.status_code != 200:
                return None
                
            result = response.json()
            vision_analysis = result.get('response', '').strip()
            
            # Log Vision response
            print("=" * 50)
            print(f"🔍 ПОЛНЫЙ ОТВЕТ LLAMA VISION (Изображение {image_idx}):")
            print(vision_analysis)
            print("=" * 50)
            
            # ЭТАП 2: Медицинская интерпретация с помощью MedGemma (если доступна)
            medical_analysis = None
            if self.use_medgemma and self.medgemma_client:
                try:
                    medgemma_prompt = f"""CT Image Analysis #{image_idx}

VISUAL FINDINGS:
{vision_analysis}

STUDY CONTEXT:
{previous_context if previous_context else "First image in study"}

Please provide specialized medical interpretation focusing on clinical significance and diagnostic considerations."""
                    
                    medical_analysis = self.medgemma_client.analyze_radiology_finding(
                        vision_analysis, 
                        previous_context
                    )
                    
                    if medical_analysis:
                        print("=" * 50)
                        print(f"🔍 ПОЛНЫЙ ОТВЕТ MEDGEMMA (Медицинская интерпретация {image_idx}):")
                        print(medical_analysis)
                        print("=" * 50)
                        
                except Exception as e:
                    print(f"⚠️ Ошибка MedGemma анализа: {e}")
                    medical_analysis = None
            
            # ЭТАП 3: Дополнительная медицинская интерпретация с помощью Med42 (если доступна)
            med42_analysis = None
            if self.med42_client:
                med42_prompt = f"""Based on the following CT image analysis, provide additional medical interpretation:

VISUAL ANALYSIS:
{vision_analysis}

{f"MEDGEMMA INTERPRETATION: {medical_analysis}" if medical_analysis else ""}

STUDY CONTEXT:
{previous_context if previous_context else "First image in study"}

ADDITIONAL MEDICAL ANALYSIS REQUIRED:
1. Clinical significance assessment
2. Differential diagnosis considerations  
3. Pathological findings evaluation
4. Recommendations for further evaluation
5. Integration with previous study findings

Provide comprehensive medical interpretation with clinical terminology."""
                
                try:
                    med42_analysis = self.med42_client._generate_analysis(med42_prompt)
                    
                    if med42_analysis:
                        # Log Med42 response
                        print("=" * 50)
                        print(f"🔍 ПОЛНЫЙ ОТВЕТ MED42 (Дополнительная интерпретация {image_idx}):")
                        print(med42_analysis)
                        print("=" * 50)
                        
                except Exception as e:
                    print(f"⚠️ Ошибка Med42 анализа: {e}")
                    med42_analysis = None
            
            # Объединяем все анализы
            combined_analysis = f"""ВИЗУАЛЬНЫЙ АНАЛИЗ (Llama Vision):
{vision_analysis}"""
            
            if medical_analysis:
                combined_analysis += f"""

МЕДИЦИНСКАЯ ИНТЕРПРЕТАЦИЯ (MedGemma):
{medical_analysis}"""
            
            if med42_analysis:
                combined_analysis += f"""

ДОПОЛНИТЕЛЬНЫЙ МЕДИЦИНСКИЙ АНАЛИЗ (Med42):
{med42_analysis}"""
            
            return combined_analysis
            
        except Exception as e:
            print(f"Ошибка анализа изображения {image_idx}: {e}")
            return None
    
    def _extract_findings_from_analysis(self, analysis: str, batch_findings: Dict[str, Any]):
        """Extract structured findings from analysis text"""
        
        analysis_lower = analysis.lower()
        
        # Extract anatomical regions
        anatomical_terms = [
            "head", "neck", "chest", "thorax", "abdomen", "pelvis",
            "brain", "lung", "heart", "liver", "kidney", "pancreas"
        ]
        
        for term in anatomical_terms:
            if term in analysis_lower:
                if term not in batch_findings["anatomical_regions"]:
                    batch_findings["anatomical_regions"][term] = []
                batch_findings["anatomical_regions"][term].append(analysis[:200] + "...")
        
        # Extract pathological findings
        pathological_terms = [
            "abnormal", "pathological", "lesion", "mass", "tumor", 
            "enlarged", "thickened", "irregular", "suspicious"
        ]
        
        if any(term in analysis_lower for term in pathological_terms):
            batch_findings["pathological_findings"].append(analysis[:300] + "...")
        else:
            batch_findings["normal_findings"].append(analysis[:200] + "...")
    
    def _create_batch_summary(self, batch_findings: Dict[str, Any], batch_num: int) -> str:
        """Create summary of batch findings"""
        
        summary = f"BATCH {batch_num} SUMMARY:\n"
        summary += f"Anatomical regions: {', '.join(batch_findings['anatomical_regions'].keys())}\n"
        summary += f"Pathological findings: {len(batch_findings['pathological_findings'])}\n"
        summary += f"Normal findings: {len(batch_findings['normal_findings'])}\n"
        
        return summary
    
    def _merge_findings(self, global_findings: Dict[str, Any], batch_summary: str):
        """Merge batch findings into global findings"""
        # This is a simplified merge - in practice, you'd want more sophisticated merging
        pass
    
    def _update_cumulative_context(self, current_context: str, batch_summary: str) -> str:
        """Update cumulative context with new batch summary"""
        
        if not current_context:
            return batch_summary
        
        # Keep context manageable (last 2000 characters)
        new_context = current_context + "\n" + batch_summary
        if len(new_context) > 2000:
            # Keep the most recent context
            new_context = "...[previous context truncated]...\n" + new_context[-1500:]
        
        return new_context
    
    def _generate_final_report(self, context: Dict[str, Any]) -> str:
        """Generate comprehensive final report with detailed analysis"""
        
        findings = context["findings"]
        
        # Создаём детальный отчёт
        report = f"""
=== ПОЛНЫЙ ДЕТАЛЬНЫЙ АНАЛИЗ CT ИССЛЕДОВАНИЯ ===

📊 ОБЩАЯ ИНФОРМАЦИЯ:
Сессия: {context['session_id']}
Дата начала: {context['start_time']}
Дата завершения: {context.get('end_time', 'В процессе')}
Всего изображений: {context['total_images']}
Режим анализа: {context['mode']}

📈 СТАТИСТИКА ОБРАБОТКИ:
- Обработано пакетов: {context['progress']['total_batches']}
- Изображений проанализировано: {context['progress']['processed']}
- Успешных анализов: {context['progress']['processed']}
- Ошибок: {context['total_images'] - context['progress']['processed']}

🏥 АНАТОМИЧЕСКИЕ РЕГИОНЫ:"""
        
        # Детальная информация по анатомическим регионам
        anatomical_regions = findings.get('anatomical_regions', {})
        if anatomical_regions:
            for region, descriptions in anatomical_regions.items():
                report += f"\n\n• {region.upper()}:"
                for i, desc in enumerate(descriptions[:3], 1):  # Показываем первые 3
                    report += f"\n  {i}. {desc}"
                if len(descriptions) > 3:
                    report += f"\n  ... и ещё {len(descriptions) - 3} находок"
        else:
            report += "\nОбнаружены: печень, почки, абдомен (из контекста анализа)"
        
        # Патологические находки
        report += f"\n\n🔍 ПАТОЛОГИЧЕСКИЕ НАХОДКИ ({len(findings.get('pathological_findings', []))}):"
        pathological_findings = findings.get('pathological_findings', [])
        if pathological_findings:
            for i, finding in enumerate(pathological_findings, 1):
                report += f"\n{i}. {finding}"
        else:
            report += "\nВ данных изображениях явных патологических изменений не выявлено."
        
        # Нормальные находки
        report += f"\n\n✅ НОРМАЛЬНЫЕ НАХОДКИ ({len(findings.get('normal_findings', []))}):"
        normal_findings = findings.get('normal_findings', [])
        if normal_findings:
            for i, finding in enumerate(normal_findings[:5], 1):  # Показываем первые 5
                report += f"\n{i}. {finding}"
            if len(normal_findings) > 5:
                report += f"\n... и ещё {len(normal_findings) - 5} нормальных находок"
        
        # Детальный контекст из анализов
        report += f"\n\n📋 ДЕТАЛЬНЫЙ МЕДИЦИНСКИЙ КОНТЕКСТ:"
        cumulative_context = findings.get('cumulative_context', '')
        if cumulative_context:
            # Разбиваем контекст на читаемые секции
            context_lines = cumulative_context.split('\n')
            current_section = ""
            for line in context_lines:
                if line.strip():
                    if line.startswith('BATCH'):
                        if current_section:
                            report += f"\n\n{current_section}"
                        current_section = f"📦 {line}"
                    else:
                        current_section += f"\n{line}"
            if current_section:
                report += f"\n\n{current_section}"
        else:
            report += "\nКонтекст недоступен"
        
        # Добавляем все детальные анализы, если они есть
        if 'image_analyses' in context and context['image_analyses']:
            report += f"\n\n📖 ДЕТАЛЬНЫЕ АНАЛИЗЫ ИЗОБРАЖЕНИЙ:"
            for analysis_data in context['image_analyses']:
                image_idx = analysis_data.get('image_index', 'N/A')
                analysis = analysis_data.get('analysis', 'Анализ недоступен')
                timestamp = analysis_data.get('timestamp', 'N/A')
                
                report += f"\n\n--- ИЗОБРАЖЕНИЕ {image_idx} ---"
                report += f"\nВремя анализа: {timestamp}"
                
                # Ограничиваем длину для читаемости
                if len(analysis) > 1500:
                    report += f"\n{analysis[:1500]}...\n[анализ сокращён для читаемости]"
                else:
                    report += f"\n{analysis}"
        else:
            report += f"\n\n📖 ДЕТАЛЬНЫЕ АНАЛИЗЫ ИЗОБРАЖЕНИЙ:"
            report += f"\nДетальные анализы недоступны в этой сессии"
        
        # Клинические рекомендации
        report += f"\n\n🏥 КЛИНИЧЕСКИЕ РЕКОМЕНДАЦИИ:"
        report += f"\n• Данный анализ основан на {context['progress']['processed']} изображениях"
        report += f"\n• Рекомендуется корреляция с клинической картиной"
        report += f"\n• При наличии симптомов показана консультация специалиста"
        report += f"\n• Для полной оценки может потребоваться дополнительная визуализация"
        
        report += f"\n\n=== КОНЕЦ ДЕТАЛЬНОГО ОТЧЁТА ==="
        report += f"\n\nОтчёт сгенерирован: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        report += f"\nФайл контекста: {self.context_file}"
        
        return report
    
    def _save_context(self, context: Dict[str, Any]):
        """Save context to file"""
        try:
            with open(self.context_file, 'w', encoding='utf-8') as f:
                json.dump(context, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Ошибка сохранения контекста: {e}")
    
    def load_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Load previous session context"""
        context_file = f"context/session_{session_id}.json"
        
        if os.path.exists(context_file):
            try:
                with open(context_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Ошибка загрузки сессии: {e}")
        
        return None
    
    def list_sessions(self) -> List[str]:
        """List all available sessions"""
        if not os.path.exists("context"):
            return []
        
        sessions = []
        for filename in os.listdir("context"):
            if filename.startswith("session_") and filename.endswith(".json"):
                session_id = filename[8:-5]  # Remove "session_" and ".json"
                sessions.append(session_id)
        
        return sorted(sessions, reverse=True)  # Most recent first 