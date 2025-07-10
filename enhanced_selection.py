"""
Enhanced Strategic Image Selection for CT Analysis
Ensures comprehensive anatomical coverage including spleen and other critical organs
"""

import numpy as np
from typing import List, Dict, Any, Tuple
import config

class EnhancedImageSelector:
    """Enhanced image selection with anatomical region coverage"""
    
    def __init__(self):
        """Initialize enhanced selector"""
        self.regions = config.ANATOMICAL_REGIONS
        self.spleen_config = config.SPLEEN_DETECTION
        
    def select_strategic_images(self, images: List[Dict[str, Any]], 
                              target_count: int = None) -> List[Dict[str, Any]]:
        """
        Select images with enhanced anatomical coverage
        
        Args:
            images: List of processed image data
            target_count: Target number of images to select
            
        Returns:
            Selected images with anatomical coverage
        """
        if not images:
            return []
            
        target_count = target_count or config.STRATEGIC_SELECTION_COUNT
        print(f"🔍 УЛУЧШЕННЫЙ СТРАТЕГИЧЕСКИЙ ВЫБОР:")
        print(f"   Целевое количество: {target_count} изображений из {len(images)}")
        print(f"   Режим: анатомическое покрытие с приоритетом селезенки")
        
        # Sort images by slice location
        sorted_images = self._sort_images_by_location(images)
        
        # Select images by anatomical regions
        selected_images = self._select_by_anatomical_regions(sorted_images, target_count)
        
        # Ensure spleen coverage if enabled
        if self.spleen_config["enabled"]:
            selected_images = self._ensure_spleen_coverage(sorted_images, selected_images)
        
        # Fill remaining slots if needed
        if len(selected_images) < target_count:
            selected_images = self._fill_remaining_slots(sorted_images, selected_images, target_count)
        
        print(f"Выбрано изображений: {len(selected_images)}")
        self._log_selection_summary(selected_images, len(images))
        
        return selected_images[:target_count]
    
    def _sort_images_by_location(self, images: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Sort images by slice location"""
        def get_slice_position(img_data):
            metadata = img_data.get('metadata', {})
            # Try different location fields
            for field in ['SliceLocation', 'ImagePositionPatient', 'InstanceNumber']:
                if field in metadata:
                    value = metadata[field]
                    if isinstance(value, (list, tuple)):
                        return float(value[2]) if len(value) > 2 else float(value[0])
                    return float(value)
            return 0.0
        
        return sorted(images, key=get_slice_position, reverse=True)
    
    def _select_by_anatomical_regions(self, sorted_images: List[Dict[str, Any]], 
                                    target_count: int) -> List[Dict[str, Any]]:
        """Select images based on anatomical regions"""
        selected = []
        total_images = len(sorted_images)
        
        # Calculate region boundaries
        region_selections = {}
        
        for region_name, region_config in self.regions.items():
            start_ratio, end_ratio = region_config["slice_range"]
            min_images = region_config["min_images"]
            
            # Calculate slice indices
            start_idx = int(start_ratio * total_images)
            end_idx = int(end_ratio * total_images)
            
            # Ensure we have enough images in this range
            if end_idx <= start_idx:
                continue
                
            region_images = sorted_images[start_idx:end_idx]
            
            # Select images from this region
            if len(region_images) >= min_images:
                # Distribute selection within region
                selected_indices = np.linspace(0, len(region_images)-1, min_images, dtype=int)
                region_selected = [region_images[i] for i in selected_indices]
                
                # Add region info to metadata
                for img in region_selected:
                    img['anatomical_region'] = region_name
                    img['region_priority'] = region_config["priority"]
                
                region_selections[region_name] = region_selected
                selected.extend(region_selected)
                
                print(f"   📍 {region_name}: {len(region_selected)}/{len(region_images)} изображений (приоритет: {region_config['priority']})")
        
        return selected
    
    def _ensure_spleen_coverage(self, sorted_images: List[Dict[str, Any]], 
                              current_selection: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Ensure adequate spleen coverage"""
        spleen_range = self.spleen_config["slice_range"]
        min_spleen_images = self.spleen_config["min_images"]
        
        # Check if we already have spleen coverage
        spleen_images_count = 0
        for img in current_selection:
            region = img.get('anatomical_region', '')
            if region in ['abdomen_upper', 'chest_lower']:
                spleen_images_count += 1
        
        if spleen_images_count >= min_spleen_images:
            print(f"   🫘 Покрытие селезенки: {spleen_images_count} изображений (достаточно)")
            return current_selection
        
        # Add specific spleen images
        total_images = len(sorted_images)
        start_idx = int(spleen_range[0] * total_images)
        end_idx = int(spleen_range[1] * total_images)
        
        spleen_region_images = sorted_images[start_idx:end_idx]
        
        # Select additional spleen images
        needed_images = min_spleen_images - spleen_images_count
        if len(spleen_region_images) >= needed_images:
            selected_indices = np.linspace(0, len(spleen_region_images)-1, needed_images, dtype=int)
            additional_spleen = [spleen_region_images[i] for i in selected_indices]
            
            # Add spleen-specific metadata
            for img in additional_spleen:
                img['anatomical_region'] = 'spleen_focus'
                img['region_priority'] = 'critical'
                img['spleen_specific'] = True
            
            current_selection.extend(additional_spleen)
            print(f"   🫘 Добавлено {len(additional_spleen)} изображений для покрытия селезенки")
        
        return current_selection
    
    def _fill_remaining_slots(self, sorted_images: List[Dict[str, Any]], 
                            current_selection: List[Dict[str, Any]], 
                            target_count: int) -> List[Dict[str, Any]]:
        """Fill remaining slots with distributed selection"""
        remaining_slots = target_count - len(current_selection)
        if remaining_slots <= 0:
            return current_selection
        
        # Get indices of already selected images
        selected_indices = set()
        for img in current_selection:
            try:
                idx = sorted_images.index(img)
                selected_indices.add(idx)
            except ValueError:
                continue
        
        # Find unselected images
        unselected = [img for i, img in enumerate(sorted_images) if i not in selected_indices]
        
        if len(unselected) >= remaining_slots:
            # Distribute remaining selections
            selected_indices = np.linspace(0, len(unselected)-1, remaining_slots, dtype=int)
            additional = [unselected[i] for i in selected_indices]
            
            # Add metadata
            for img in additional:
                img['anatomical_region'] = 'distributed_fill'
                img['region_priority'] = 'medium'
            
            current_selection.extend(additional)
            print(f"Добавлено {len(additional)} изображений для заполнения")
        
        return current_selection
    
    def _log_selection_summary(self, selected_images: List[Dict[str, Any]], total_images: int):
        """Log selection summary by regions"""
        region_counts = {}
        for img in selected_images:
            region = img.get('anatomical_region', 'unknown')
            region_counts[region] = region_counts.get(region, 0) + 1
        
        print(f"\n📊 ИТОГОВАЯ СВОДКА ВЫБОРА:")
        print(f"   Выбрано: {len(selected_images)} из {total_images} изображений")
        
        for region, count in region_counts.items():
            priority = 'unknown'
            if region in self.regions:
                priority = self.regions[region]['priority']
            elif region == 'spleen_focus':
                priority = 'critical'
            
            # Add emoji based on priority
            emoji = "🔴" if priority == "critical" else "🟡" if priority == "high" else "🟢" if priority == "medium" else "⚪"
            print(f"   {emoji} {region}: {count} изображений ({priority})")
        print("=" * 50)

def select_enhanced_images(images: List[Dict[str, Any]], 
                         target_count: int = None) -> List[Dict[str, Any]]:
    """
    Convenience function for enhanced image selection
    
    Args:
        images: List of processed image data
        target_count: Target number of images to select
        
    Returns:
        Selected images with enhanced anatomical coverage
    """
    selector = EnhancedImageSelector()
    return selector.select_strategic_images(images, target_count) 