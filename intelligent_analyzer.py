"""
Intelligent CT Analyzer - Multi-Stage Analysis System
Implements three-stage analysis: Subject identification, Anatomical mapping, Medical analysis
"""

import requests
import json
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import config

class IntelligentAnalyzer:
    """Three-stage intelligent CT analysis system"""
    
    def __init__(self):
        """Initialize intelligent analyzer"""
        self.base_url = config.OLLAMA_BASE_URL
        self.vision_model = config.LLAMA_VISION_MODEL
        self.gemma_model = config.GEMMA_MODEL
        print("🧠 Intelligent Analyzer инициализирован")
    
    def analyze_study(self, images: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Perform complete three-stage analysis
        
        Returns:
            Complete analysis results with all stages
        """
        print("\n🔬 ЗАПУСК ТРЁХЭТАПНОГО ИНТЕЛЛЕКТУАЛЬНОГО АНАЛИЗА")
        print("=" * 60)
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'total_images': len(images),
            'stages': {}
        }
        
        try:
            # Stage 1: Subject Identification
            print("\n📋 ЭТАП 1: ОПРЕДЕЛЕНИЕ ТИПА СУБЪЕКТА")
            subject_info = self._stage1_subject_identification(images)
            results['stages']['subject_identification'] = subject_info
            
            # Stage 2: Anatomical Mapping
            print("\n🗺️  ЭТАП 2: КАРТИРОВАНИЕ АНАТОМИИ")
            anatomy_map = self._stage2_anatomical_mapping(images, subject_info)
            results['stages']['anatomical_mapping'] = anatomy_map
            
            # Stage 3: Medical Analysis
            print("\n🏥 ЭТАП 3: ДЕТАЛЬНЫЙ МЕДИЦИНСКИЙ АНАЛИЗ")
            medical_analysis = self._stage3_medical_analysis(images, subject_info, anatomy_map)
            results['stages']['medical_analysis'] = medical_analysis
            
            # Generate comprehensive report
            results['final_report'] = self._generate_comprehensive_report(results)
            
            print("\n✅ ТРЁХЭТАПНЫЙ АНАЛИЗ ЗАВЕРШЁН")
            return results
            
        except Exception as e:
            print(f"❌ Ошибка интеллектуального анализа: {e}")
            results['error'] = str(e)
            return results
    
    def _stage1_subject_identification(self, images: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Stage 1: Identify what type of subject we're analyzing
        """
        print("   🔍 Анализ нескольких случайных снимков для определения субъекта...")
        
        # Select 3-4 random images from different parts of the study
        sample_images = self._select_identification_samples(images)
        print(f"   📸 Выбрано {len(sample_images)} снимков для идентификации")
        
        identifications = []
        for i, img_data in enumerate(sample_images):
            print(f"   🔎 Анализ образца {i+1}/{len(sample_images)}...")
            identification = self._identify_subject_in_image(img_data)
            if identification:
                identifications.append(identification)
        
        # Analyze results and determine subject type
        subject_analysis = self._analyze_subject_identifications(identifications)
        
        print(f"   ✅ Определён субъект: {subject_analysis.get('subject_type', 'неизвестно')}")
        return subject_analysis
    
    def _stage2_anatomical_mapping(self, images: List[Dict[str, Any]], subject_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Stage 2: Map anatomical regions across the study
        """
        print("   🗺️  Создание карты анатомических регионов...")
        
        # Select strategic samples from different parts of the study
        mapping_samples = self._select_mapping_samples(images)
        print(f"   📍 Выбрано {len(mapping_samples)} снимков для картирования")
        
        anatomical_regions = []
        for i, (region_name, img_data) in enumerate(mapping_samples):
            print(f"   🔍 Картирование региона {region_name} ({i+1}/{len(mapping_samples)})...")
            region_analysis = self._map_anatomical_region(img_data, region_name, subject_info)
            if region_analysis:
                anatomical_regions.append({
                    'region': region_name,
                    'analysis': region_analysis,
                    'slice_info': img_data.get('metadata', {}).get('SliceLocation', 'Unknown')
                })
        
        # Create comprehensive anatomical map
        anatomy_map = self._create_anatomical_map(anatomical_regions, subject_info)
        
        print(f"   ✅ Создана карта {len(anatomical_regions)} анатомических регионов")
        return anatomy_map
    
    def _stage3_medical_analysis(self, images: List[Dict[str, Any]], 
                                subject_info: Dict[str, Any], 
                                anatomy_map: Dict[str, Any]) -> Dict[str, Any]:
        """
        Stage 3: Detailed medical analysis with context
        """
        print("   🏥 Проведение детального медицинского анализа...")
        
        # Select images for detailed analysis based on anatomical map
        analysis_images = self._select_medical_analysis_images(images, anatomy_map)
        print(f"   🔬 Выбрано {len(analysis_images)} снимков для медицинского анализа")
        
        medical_findings = []
        for i, (region, img_data) in enumerate(analysis_images):
            print(f"   🩺 Анализ {region} ({i+1}/{len(analysis_images)})...")
            finding = self._analyze_medical_region(img_data, region, subject_info, anatomy_map)
            if finding:
                medical_findings.append({
                    'region': region,
                    'finding': finding,
                    'slice_info': img_data.get('metadata', {}).get('SliceLocation', 'Unknown')
                })
        
        # Synthesize medical conclusions
        medical_synthesis = self._synthesize_medical_findings(medical_findings, subject_info, anatomy_map)
        
        print(f"   ✅ Проанализировано {len(medical_findings)} медицинских регионов")
        return medical_synthesis
    
    def _select_identification_samples(self, images: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Select representative samples for subject identification"""
        if len(images) <= 4:
            return images
        
        # Select from beginning, middle, and end
        indices = [
            0,  # First image
            len(images) // 3,  # First third
            len(images) // 2,  # Middle
            (len(images) * 2) // 3,  # Last third
        ]
        
        return [images[i] for i in indices if i < len(images)]
    
    def _select_mapping_samples(self, images: List[Dict[str, Any]]) -> List[Tuple[str, Dict[str, Any]]]:
        """Select samples for anatomical mapping"""
        total = len(images)
        if total == 0:
            return []
        
        # Define anatomical regions with their approximate positions
        regions = [
            ("head_brain", 0.9),      # Top 10%
            ("chest_upper", 0.75),    # Upper chest
            ("chest_lower", 0.6),     # Lower chest
            ("abdomen_upper", 0.4),   # Upper abdomen (liver, pancreas)
            ("abdomen_middle", 0.25), # Middle abdomen
            ("pelvis", 0.1),          # Pelvis
        ]
        
        samples = []
        for region_name, position in regions:
            index = int(total * position)
            if index >= total:
                index = total - 1
            samples.append((region_name, images[index]))
        
        return samples
    
    def _select_medical_analysis_images(self, images: List[Dict[str, Any]], 
                                      anatomy_map: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
        """Select images for detailed medical analysis based on anatomical map"""
        # Enhanced selection based on anatomy map findings
        total = len(images)
        selections = []
        
        # Focus on regions identified in anatomy map
        important_regions = anatomy_map.get('important_regions', [])
        
        if 'abdomen' in important_regions:
            # Extra focus on abdominal region (30-50% of study)
            abdomen_start = int(total * 0.3)
            abdomen_end = int(total * 0.5)
            abdomen_indices = [abdomen_start, (abdomen_start + abdomen_end) // 2, abdomen_end]
            for i, idx in enumerate(abdomen_indices):
                if idx < total:
                    selections.append((f"abdomen_focus_{i+1}", images[idx]))
        
        # Add other important regions
        if 'chest' in important_regions:
            chest_idx = int(total * 0.7)
            if chest_idx < total:
                selections.append(("chest_detail", images[chest_idx]))
        
        if 'brain' in important_regions:
            brain_idx = int(total * 0.9)
            if brain_idx < total:
                selections.append(("brain_detail", images[brain_idx]))
        
        # Ensure minimum coverage
        if len(selections) < 6:
            step = total // 6
            for i in range(6):
                idx = i * step
                if idx < total and not any(idx == img_data.get('index', -1) for _, img_data in selections):
                    selections.append((f"general_{i+1}", images[idx]))
        
        return selections[:8]  # Limit to 8 images
    
    def _identify_subject_in_image(self, image_data: Dict[str, Any]) -> Optional[str]:
        """Identify subject type in a single image"""
        try:
            prompt = """Analyze this medical image and identify the subject:

IDENTIFICATION REQUIREMENTS:
1. What type of subject is this? (human, animal, other)
2. If human: approximate age group (infant, child, adult, elderly)
3. If animal: what type of animal?
4. Body orientation (axial, sagittal, coronal)
5. Anatomical region visible
6. Image quality assessment

Provide clear, concise identification focusing on SUBJECT TYPE."""
            
            payload = {
                "model": self.vision_model,
                "prompt": prompt,
                "images": [image_data['base64_image']],
                "stream": False,
                "options": {
                    "temperature": 0.3,  # Lower temperature for more consistent identification
                    "num_predict": 300
                }
            }
            
            response = requests.post(f"{self.base_url}/api/generate", json=payload, timeout=180)
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', '').strip()
            
            return None
            
        except Exception as e:
            print(f"      ❌ Ошибка идентификации: {e}")
            return None
    
    def _map_anatomical_region(self, image_data: Dict[str, Any], region_name: str, 
                              subject_info: Dict[str, Any]) -> Optional[str]:
        """Map anatomical structures in a specific region"""
        try:
            subject_type = subject_info.get('subject_type', 'unknown')
            
            prompt = f"""Analyze this {region_name} region CT image for a {subject_type}:

ANATOMICAL MAPPING REQUIREMENTS:
1. Identify all visible anatomical structures
2. Describe organ positions and relationships
3. Note any anatomical landmarks
4. Assess organ sizes and morphology
5. Identify vascular structures
6. Note any obvious abnormalities

Focus on ANATOMICAL IDENTIFICATION and SPATIAL RELATIONSHIPS for region: {region_name}"""
            
            payload = {
                "model": self.vision_model,
                "prompt": prompt,
                "images": [image_data['base64_image']],
                "stream": False,
                "options": {
                    "temperature": 0.5,
                    "num_predict": 500
                }
            }
            
            response = requests.post(f"{self.base_url}/api/generate", json=payload, timeout=90)
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', '').strip()
            
            return None
            
        except Exception as e:
            print(f"      ❌ Ошибка картирования {region_name}: {e}")
            return None
    
    def _analyze_medical_region(self, image_data: Dict[str, Any], region: str,
                               subject_info: Dict[str, Any], anatomy_map: Dict[str, Any]) -> Optional[str]:
        """Perform detailed medical analysis of a specific region"""
        try:
            context = f"Subject: {subject_info.get('subject_type', 'unknown')}\n"
            context += f"Region: {region}\n"
            context += f"Anatomical context: {anatomy_map.get('summary', 'Not available')}"
            
            prompt = f"""Perform detailed medical analysis of this CT image:

CONTEXT:
{context}

MEDICAL ANALYSIS REQUIREMENTS:
1. Pathological findings detection
2. Organ assessment (size, density, morphology)
3. Vascular evaluation
4. Tissue characterization
5. Abnormality identification
6. Clinical significance assessment
7. Comprehensive organ evaluation

Provide detailed MEDICAL ASSESSMENT with clinical terminology."""
            
            payload = {
                "model": self.gemma_model,  # Use Gemma for medical analysis
                "prompt": prompt,
                "images": [image_data['base64_image']],
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "num_predict": 800
                }
            }
            
            response = requests.post(f"{self.base_url}/api/generate", json=payload, timeout=180)
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', '').strip()
            
            return None
            
        except Exception as e:
            print(f"      ❌ Ошибка медицинского анализа {region}: {e}")
            return None
    
    def _analyze_subject_identifications(self, identifications: List[str]) -> Dict[str, Any]:
        """Analyze multiple subject identifications to determine consensus"""
        if not identifications:
            return {'subject_type': 'unknown', 'confidence': 'low'}
        
        # Simple analysis - look for common terms
        text = ' '.join(identifications).lower()
        
        if 'human' in text or 'patient' in text or 'adult' in text:
            subject_type = 'human'
        elif 'animal' in text or 'dog' in text or 'cat' in text:
            subject_type = 'animal'
        else:
            subject_type = 'unknown'
        
        return {
            'subject_type': subject_type,
            'confidence': 'high' if len(identifications) >= 3 else 'medium',
            'raw_identifications': identifications,
            'analysis_summary': f"Identified as {subject_type} based on {len(identifications)} samples"
        }
    
    def _create_anatomical_map(self, regions: List[Dict], subject_info: Dict[str, Any]) -> Dict[str, Any]:
        """Create comprehensive anatomical map from regional analyses"""
        important_regions = []
        organ_locations = {}
        
        for region_data in regions:
            region_name = region_data['region']
            analysis = region_data['analysis']
            
            # Extract important information
            if analysis and len(analysis) > 50:  # Meaningful analysis
                important_regions.append(region_name)
                
                # Look for specific organs
                analysis_lower = analysis.lower()
                if 'liver' in analysis_lower:
                    organ_locations['liver'] = region_name
                if 'heart' in analysis_lower:
                    organ_locations['heart'] = region_name
                if 'brain' in analysis_lower:
                    organ_locations['brain'] = region_name
        
        return {
            'subject_type': subject_info.get('subject_type', 'unknown'),
            'important_regions': important_regions,
            'organ_locations': organ_locations,
            'regional_analyses': regions,
            'summary': f"Mapped {len(important_regions)} important regions with {len(organ_locations)} organs identified"
        }
    
    def _synthesize_medical_findings(self, findings: List[Dict], 
                                   subject_info: Dict[str, Any], 
                                   anatomy_map: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize all medical findings into comprehensive assessment"""
        
        pathological_findings = []
        normal_findings = []
        organ_specific_findings = []
        
        for finding_data in findings:
            finding = finding_data['finding']
            region = finding_data['region']
            
            if finding and len(finding) > 30:
                finding_lower = finding.lower()
                
                # Categorize findings
                if any(term in finding_lower for term in ['abnormal', 'pathological', 'lesion', 'mass']):
                    pathological_findings.append(f"{region}: {finding}")
                else:
                    normal_findings.append(f"{region}: {finding}")
                
                # Collect organ-specific findings
                if any(organ in finding_lower for organ in ['liver', 'kidney', 'pancreas', 'heart']):
                    organ_specific_findings.append(f"{region}: {finding}")
        
        return {
            'subject_type': subject_info.get('subject_type', 'unknown'),
            'total_regions_analyzed': len(findings),
            'pathological_findings': pathological_findings,
            'normal_findings': normal_findings,
            'organ_specific_findings': organ_specific_findings,
            'anatomy_context': anatomy_map.get('summary', ''),
            'clinical_summary': self._generate_clinical_summary(pathological_findings, normal_findings, organ_specific_findings)
        }
    
    def _generate_clinical_summary(self, pathological: List[str], normal: List[str], organ_specific: List[str]) -> str:
        """Generate clinical summary from findings"""
        summary = []
        
        if pathological:
            summary.append(f"PATHOLOGICAL FINDINGS ({len(pathological)}): Notable abnormalities detected requiring clinical attention.")
        
        if normal:
            summary.append(f"NORMAL FINDINGS ({len(normal)}): Multiple regions show normal anatomical appearance.")
        
        if organ_specific:
            summary.append(f"ORGAN-SPECIFIC ASSESSMENT: {len(organ_specific)} detailed organ findings documented.")
        
        if not summary:
            summary.append("ASSESSMENT INCOMPLETE: Limited findings available for clinical interpretation.")
        
        return " ".join(summary)
    
    def _generate_comprehensive_report(self, results: Dict[str, Any]) -> str:
        """Generate final comprehensive report"""
        
        stages = results.get('stages', {})
        subject_info = stages.get('subject_identification', {})
        anatomy_map = stages.get('anatomical_mapping', {})
        medical_analysis = stages.get('medical_analysis', {})
        
        report = f"""
=== INTELLIGENT CT ANALYSIS REPORT ===

Analysis Date: {results.get('timestamp', 'Unknown')}
Total Images: {results.get('total_images', 'Unknown')}
Analysis Method: Three-Stage Intelligent Analysis

STAGE 1 - SUBJECT IDENTIFICATION:
Subject Type: {subject_info.get('subject_type', 'Unknown')}
Confidence: {subject_info.get('confidence', 'Unknown')}
Summary: {subject_info.get('analysis_summary', 'No summary available')}

STAGE 2 - ANATOMICAL MAPPING:
Regions Mapped: {len(anatomy_map.get('important_regions', []))}
Organs Identified: {', '.join(anatomy_map.get('organ_locations', {}).keys()) if anatomy_map.get('organ_locations') else 'None specified'}
Mapping Summary: {anatomy_map.get('summary', 'No mapping summary available')}

STAGE 3 - MEDICAL ANALYSIS:
Regions Analyzed: {medical_analysis.get('total_regions_analyzed', 0)}
Pathological Findings: {len(medical_analysis.get('pathological_findings', []))}
Normal Findings: {len(medical_analysis.get('normal_findings', []))}
Organ-Specific Findings: {len(medical_analysis.get('organ_specific_findings', []))}

CLINICAL SUMMARY:
{medical_analysis.get('clinical_summary', 'No clinical summary available')}

DETAILED FINDINGS:
"""
        
        # Add pathological findings
        if medical_analysis.get('pathological_findings'):
            report += "\nPATHOLOGICAL FINDINGS:\n"
            for finding in medical_analysis['pathological_findings']:
                report += f"- {finding}\n"
        
        # Add organ-specific findings
        if medical_analysis.get('organ_specific_findings'):
            report += "\nORGAN-SPECIFIC FINDINGS:\n"
            for finding in medical_analysis['organ_specific_findings']:
                report += f"- {finding}\n"
        
        report += "\n=== END INTELLIGENT ANALYSIS REPORT ==="
        
        return report 