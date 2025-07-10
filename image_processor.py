"""
Image Processor - DICOM Image Processing and Preparation
Handles DICOM file loading, image processing, and strategic selection
"""

import os
import pydicom
import numpy as np
from PIL import Image
import io
import base64
from typing import List, Dict, Optional, Tuple, Any
import config

class ImageProcessor:
    """DICOM image processor with strategic selection capabilities"""
    
    def __init__(self):
        """Initialize the image processor"""
        self.target_size = config.TARGET_SIZE
        self.quality = config.QUALITY
        
    def load_dicom_series(self, directory_path: str) -> List[Dict[str, Any]]:
        """
        Load and process a series of DICOM files from directory
        
        Args:
            directory_path: Path to directory containing DICOM files
            
        Returns:
            List of processed image data dictionaries
        """
        dicom_files = []
        processed_images = []
        
        # Find all DICOM files
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                file_path = os.path.join(root, file)
                if self._is_dicom_file(file_path):
                    dicom_files.append(file_path)
        
        # Sort files by slice location or instance number
        dicom_files = self._sort_dicom_files(dicom_files)
        
        print(f"Обработка {len(dicom_files)} DICOM-файлов...")
        
        # Process each DICOM file
        for i, file_path in enumerate(dicom_files):
            try:
                image_data = self._process_dicom_file(file_path, i)
                if image_data:
                    processed_images.append(image_data)
            except Exception as e:
                print(f"Ошибка обработки файла {file_path}: {e}")
                continue
        
        return processed_images
    
    def _is_dicom_file(self, file_path: str) -> bool:
        """Check if file is a valid DICOM file"""
        try:
            pydicom.dcmread(file_path, stop_before_pixels=True)
            return True
        except:
            return False
    
    def _sort_dicom_files(self, file_paths: List[str]) -> List[str]:
        """Sort DICOM files by slice location or instance number"""
        file_data = []
        
        for file_path in file_paths:
            try:
                ds = pydicom.dcmread(file_path, stop_before_pixels=True)
                
                # Try to get slice location or instance number for sorting
                slice_location = getattr(ds, 'SliceLocation', None)
                instance_number = getattr(ds, 'InstanceNumber', None)
                
                sort_key = slice_location if slice_location is not None else instance_number
                if sort_key is None:
                    sort_key = 0  # Default if no sorting info available
                
                file_data.append((sort_key, file_path))
                
            except Exception:
                # If can't read DICOM metadata, use original order
                file_data.append((0, file_path))
        
        # Sort by the key and return file paths
        file_data.sort(key=lambda x: x[0])
        return [file_path for _, file_path in file_data]
    
    def _process_dicom_file(self, file_path: str, index: int) -> Optional[Dict[str, Any]]:
        """
        Process a single DICOM file
        
        Args:
            file_path: Path to DICOM file
            index: File index in series
            
        Returns:
            Processed image data dictionary
        """
        try:
            # Read DICOM file
            ds = pydicom.dcmread(file_path)
            
            # Extract pixel data
            pixel_array = ds.pixel_array
            
            # Handle different pixel data formats
            if len(pixel_array.shape) == 3:
                # Multi-frame or RGB image, take first frame/convert to grayscale
                if pixel_array.shape[2] == 3:  # RGB
                    pixel_array = np.dot(pixel_array[..., :3], [0.2989, 0.5870, 0.1140])
                else:
                    pixel_array = pixel_array[0]  # First frame
            
            # Normalize pixel values to 0-255 range
            pixel_array = self._normalize_pixel_array(pixel_array)
            
            # Convert to PIL Image
            image = Image.fromarray(pixel_array.astype(np.uint8))
            
            # Resize image
            image = image.resize(self.target_size, Image.Resampling.LANCZOS)
            
            # Convert to base64 for API transmission
            buffer = io.BytesIO()
            image.save(buffer, format='JPEG', quality=self.quality)
            base64_image = base64.b64encode(buffer.getvalue()).decode()
            
            # Extract metadata
            metadata = self._extract_metadata(ds)
            
            return {
                'index': index,
                'file_path': file_path,
                'base64_image': base64_image,
                'metadata': metadata,
                'image_size': self.target_size
            }
            
        except Exception as e:
            print(f"Ошибка обработки DICOM файла {file_path}: {e}")
            return None
    
    def _normalize_pixel_array(self, pixel_array: np.ndarray) -> np.ndarray:
        """Normalize pixel array to 0-255 range"""
        # Handle different bit depths and scaling
        min_val = np.min(pixel_array)
        max_val = np.max(pixel_array)
        
        if max_val == min_val:
            return np.zeros_like(pixel_array)
        
        # Normalize to 0-255
        normalized = ((pixel_array - min_val) / (max_val - min_val) * 255)
        return normalized
    
    def _extract_metadata(self, ds: pydicom.Dataset) -> Dict[str, Any]:
        """Extract relevant metadata from DICOM dataset"""
        metadata = {}
        
        # Common DICOM tags
        metadata_fields = [
            'PatientID', 'StudyDate', 'SeriesDescription', 
            'SliceLocation', 'InstanceNumber', 'SliceThickness',
            'WindowCenter', 'WindowWidth', 'RescaleIntercept', 'RescaleSlope'
        ]
        
        for field in metadata_fields:
            try:
                value = getattr(ds, field, None)
                if value is not None:
                    metadata[field] = value
            except:
                continue
        
        return metadata
    
    def select_strategic_images(self, images: List[Dict[str, Any]], count: int = None) -> List[Dict[str, Any]]:
        """
        Select strategically important images for analysis
        
        Args:
            images: List of processed image data
            count: Number of images to select (default from config)
            
        Returns:
            List of selected images
        """
        if count is None:
            count = config.STRATEGIC_SELECTION_COUNT
        
        if len(images) <= count:
            return images
        
        # Strategic selection algorithm
        total_images = len(images)
        
        # Always include first and last images
        selected_indices = [0, total_images - 1]
        
        # Add evenly distributed images in between
        remaining_count = count - 2
        if remaining_count > 0:
            step = (total_images - 1) / (remaining_count + 1)
            for i in range(1, remaining_count + 1):
                index = int(step * i)
                if index not in selected_indices:
                    selected_indices.append(index)
        
        # Sort indices and select images
        selected_indices.sort()
        selected_images = [images[i] for i in selected_indices if i < len(images)]
        
        print(f"Стратегический выбор: {len(selected_images)} изображений из {total_images}")
        return selected_images
    
    def prepare_for_analysis(self, images: List[Dict[str, Any]], max_images: int = None) -> List[Dict[str, Any]]:
        """
        Prepare images for analysis with strategic selection if needed
        
        Args:
            images: List of processed images
            max_images: Maximum number of images for analysis
            
        Returns:
            Prepared images ready for analysis
        """
        if max_images is None:
            max_images = config.MAX_IMAGES_PER_ANALYSIS
        
        if len(images) > max_images:
            print(f"Применение стратегического выбора: {max_images} из {len(images)} изображений")
            return self.select_strategic_images(images, max_images)
        
        return images 