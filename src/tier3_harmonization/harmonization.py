"""
Tier 3: Geospatial Harmonization Layer
Handles CRS normalization, clipping, cloud masking, raster alignment, and vegetation indices
"""

import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.mask import mask
from rasterio.enums import Resampling as ResamplingEnum
import geopandas as gpd
from shapely.geometry import box, mapping
from loguru import logger
import yaml


class CRSNormalizer:
    """Convert all spatial data to consistent coordinate reference system"""
    
    def __init__(self, target_crs: str = "EPSG:4326"):
        self.target_crs = target_crs
    
    def normalize_raster(self, input_path: str, output_path: str) -> str:
        """
        Reproject raster to target CRS
        
        Args:
            input_path: Path to input raster
            output_path: Path for output raster
            
        Returns:
            Path to reprojected raster
        """
        logger.info(f"Normalizing CRS for {input_path}")
        
        with rasterio.open(input_path) as src:
            if src.crs.to_string() == self.target_crs:
                logger.info("Raster already in target CRS")
                return input_path
            
            # Calculate transform
            transform, width, height = calculate_default_transform(
                src.crs, self.target_crs, src.width, src.height, *src.bounds
            )
            
            # Update metadata
            kwargs = src.meta.copy()
            kwargs.update({
                'crs': self.target_crs,
                'transform': transform,
                'width': width,
                'height': height
            })
            
            # Reproject
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with rasterio.open(output_path, 'w', **kwargs) as dst:
                for i in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, i),
                        destination=rasterio.band(dst, i),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=self.target_crs,
                        resampling=Resampling.bilinear
                    )
        
        logger.info(f"CRS normalized: {output_path}")
        return output_path
    
    def normalize_vector(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Reproject vector data to target CRS"""
        if gdf.crs is None or gdf.crs.to_string() == self.target_crs:
            return gdf
        return gdf.to_crs(self.target_crs)


class AOIClipper:
    """Clip raster and vector data to area of interest"""
    
    def clip_raster(
        self,
        raster_path: str,
        aoi_geometry: Dict,
        output_path: str
    ) -> str:
        """
        Clip raster to AOI polygon
        
        Args:
            raster_path: Path to input raster
            aoi_geometry: GeoJSON-like geometry dictionary
            output_path: Path for clipped output
            
        Returns:
            Path to clipped raster
        """
        logger.info(f"Clipping raster {raster_path} to AOI")
        
        with rasterio.open(raster_path) as src:
            # Clip raster
            out_image, out_transform = mask(
                src, [aoi_geometry], crop=True, filled=True, nodata=src.nodata
            )
            
            # Update metadata
            out_meta = src.meta.copy()
            out_meta.update({
                "driver": "GTiff",
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform
            })
            
            # Write clipped raster
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with rasterio.open(output_path, "w", **out_meta) as dest:
                dest.write(out_image)
        
        logger.info(f"Raster clipped: {output_path}")
        return output_path
    
    def clip_vector(
        self,
        gdf: gpd.GeoDataFrame,
        aoi_geometry: Dict
    ) -> gpd.GeoDataFrame:
        """Clip vector data to AOI"""
        from shapely.geometry import shape
        aoi_shape = shape(aoi_geometry)
        return gdf[gdf.intersects(aoi_shape)]


class CloudMasker:
    """Apply cloud masking to satellite imagery"""
    
    def __init__(self):
        # Sentinel-2 Scene Classification Layer (SCL) values
        self.scl_classes = {
            0: 'NO_DATA',
            1: 'SATURATED_DEFECTIVE',
            2: 'DARK_FEATURES',
            3: 'CLOUD_SHADOWS',
            4: 'VEGETATION',
            5: 'NOT_VEGETATED',
            6: 'WATER',
            7: 'UNCLASSIFIED',
            8: 'CLOUD_MEDIUM_PROB',
            9: 'CLOUD_HIGH_PROB',
            10: 'THIN_CIRRUS',
            11: 'SNOW_ICE'
        }
        
        # Values to mask out
        self.mask_values = [0, 1, 3, 8, 9, 10, 11]  # clouds, shadows, defective
    
    def apply_cloud_mask(
        self,
        image_path: str,
        scl_path: str,
        output_path: str
    ) -> str:
        """
        Apply cloud mask using Scene Classification Layer
        
        Args:
            image_path: Path to multispectral image
            scl_path: Path to Scene Classification Layer
            output_path: Path for masked output
            
        Returns:
            Path to masked image
        """
        logger.info(f"Applying cloud mask to {image_path}")
        
        with rasterio.open(image_path) as img_src:
            image_data = img_src.read()
            profile = img_src.profile
            
            with rasterio.open(scl_path) as scl_src:
                scl_data = scl_src.read(1)
                
                # Create mask
                mask = np.isin(scl_data, self.mask_values)
                
                # Apply mask to all bands
                for i in range(image_data.shape[0]):
                    image_data[i][mask] = profile.get('nodata', 0)
                
                # Calculate cloud coverage percentage
                cloud_pct = (mask.sum() / mask.size) * 100
                logger.info(f"Cloud coverage: {cloud_pct:.2f}%")
        
        # Write masked image
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(image_data)
        
        logger.info(f"Cloud mask applied: {output_path}")
        return output_path


class RasterAligner:
    """Align multiple rasters to common grid"""
    
    def __init__(self, target_resolution: float = 10.0):
        self.target_resolution = target_resolution
    
    def align_raster(
        self,
        input_path: str,
        reference_path: str,
        output_path: str,
        resampling_method: str = 'bilinear'
    ) -> str:
        """
        Align raster to reference grid
        
        Args:
            input_path: Path to raster to align
            reference_path: Path to reference raster
            output_path: Path for aligned output
            resampling_method: Resampling method (bilinear, nearest, cubic)
            
        Returns:
            Path to aligned raster
        """
        logger.info(f"Aligning {input_path} to {reference_path}")
        
        # Map resampling method
        resampling_map = {
            'bilinear': Resampling.bilinear,
            'nearest': Resampling.nearest,
            'cubic': Resampling.cubic,
            'average': Resampling.average
        }
        resampling_enum = resampling_map.get(resampling_method, Resampling.bilinear)
        
        with rasterio.open(reference_path) as ref:
            ref_profile = ref.profile.copy()
            ref_transform = ref.transform
            ref_crs = ref.crs
            ref_width = ref.width
            ref_height = ref.height
            
            with rasterio.open(input_path) as src:
                # Update profile
                out_profile = src.profile.copy()
                out_profile.update({
                    'crs': ref_crs,
                    'transform': ref_transform,
                    'width': ref_width,
                    'height': ref_height
                })
                
                # Reproject
                Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                with rasterio.open(output_path, 'w', **out_profile) as dst:
                    for i in range(1, src.count + 1):
                        reproject(
                            source=rasterio.band(src, i),
                            destination=rasterio.band(dst, i),
                            src_transform=src.transform,
                            src_crs=src.crs,
                            dst_transform=ref_transform,
                            dst_crs=ref_crs,
                            resampling=resampling_enum
                        )
        
        logger.info(f"Raster aligned: {output_path}")
        return output_path


class VegetationIndexCalculator:
    """Calculate vegetation indices from multispectral imagery"""
    
    def __init__(self):
        self.indices = {
            'NDVI': self.calculate_ndvi,
            'NDWI': self.calculate_ndwi,
            'EVI': self.calculate_evi
        }
    
    def calculate_ndvi(self, nir: np.ndarray, red: np.ndarray) -> np.ndarray:
        """
        Calculate Normalized Difference Vegetation Index
        NDVI = (NIR - Red) / (NIR + Red)
        """
        with np.errstate(divide='ignore', invalid='ignore'):
            ndvi = (nir - red) / (nir + red)
            ndvi = np.nan_to_num(ndvi, nan=0.0)
        return np.clip(ndvi, -1, 1)
    
    def calculate_ndwi(self, green: np.ndarray, nir: np.ndarray) -> np.ndarray:
        """
        Calculate Normalized Difference Water Index
        NDWI = (Green - NIR) / (Green + NIR)
        """
        with np.errstate(divide='ignore', invalid='ignore'):
            ndwi = (green - nir) / (green + nir)
            ndwi = np.nan_to_num(ndwi, nan=0.0)
        return np.clip(ndwi, -1, 1)
    
    def calculate_evi(
        self,
        nir: np.ndarray,
        red: np.ndarray,
        blue: np.ndarray,
        G: float = 2.5,
        C1: float = 6.0,
        C2: float = 7.5,
        L: float = 1.0
    ) -> np.ndarray:
        """
        Calculate Enhanced Vegetation Index
        EVI = G * ((NIR - Red) / (NIR + C1*Red - C2*Blue + L))
        """
        with np.errstate(divide='ignore', invalid='ignore'):
            evi = G * ((nir - red) / (nir + C1 * red - C2 * blue + L))
            evi = np.nan_to_num(evi, nan=0.0)
        return np.clip(evi, -1, 1)
    
    def compute_indices(
        self,
        image_path: str,
        output_dir: str,
        band_mapping: Dict[str, int] = {'B03': 1, 'B04': 2, 'B08': 3}
    ) -> Dict[str, str]:
        """
        Compute all vegetation indices
        
        Args:
            image_path: Path to multispectral image
            output_dir: Directory for output indices
            band_mapping: Mapping of band names to band numbers
            
        Returns:
            Dictionary mapping index names to file paths
        """
        logger.info(f"Computing vegetation indices for {image_path}")
        
        output_paths = {}
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        with rasterio.open(image_path) as src:
            # Read bands
            green = src.read(band_mapping.get('B03', 1)).astype(float)
            red = src.read(band_mapping.get('B04', 2)).astype(float)
            nir = src.read(band_mapping.get('B08', 3)).astype(float)
            
            # Prepare profile for output
            profile = src.profile.copy()
            profile.update({
                'count': 1,
                'dtype': rasterio.float32,
                'nodata': -9999
            })
            
            # Calculate NDVI
            ndvi = self.calculate_ndvi(nir, red)
            ndvi_path = str(Path(output_dir) / 'NDVI.tif')
            with rasterio.open(ndvi_path, 'w', **profile) as dst:
                dst.write(ndvi.astype(rasterio.float32), 1)
            output_paths['NDVI'] = ndvi_path
            logger.info(f"NDVI calculated: {ndvi_path}")
            
            # Calculate NDWI
            ndwi = self.calculate_ndwi(green, nir)
            ndwi_path = str(Path(output_dir) / 'NDWI.tif')
            with rasterio.open(ndwi_path, 'w', **profile) as dst:
                dst.write(ndwi.astype(rasterio.float32), 1)
            output_paths['NDWI'] = ndwi_path
            logger.info(f"NDWI calculated: {ndwi_path}")
            
            # Calculate EVI (need blue band - using green as approximation if not available)
            evi = self.calculate_evi(nir, red, green)
            evi_path = str(Path(output_dir) / 'EVI.tif')
            with rasterio.open(evi_path, 'w', **profile) as dst:
                dst.write(evi.astype(rasterio.float32), 1)
            output_paths['EVI'] = evi_path
            logger.info(f"EVI calculated: {evi_path}")
        
        return output_paths


class HarmonizationPipeline:
    """Complete harmonization pipeline"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.crs_normalizer = CRSNormalizer(
            self.config['geospatial']['default_crs']
        )
        self.aoi_clipper = AOIClipper()
        self.cloud_masker = CloudMasker()
        self.raster_aligner = RasterAligner(
            self.config['geospatial']['raster_alignment']['target_resolution']
        )
        self.vi_calculator = VegetationIndexCalculator()
    
    def process(
        self,
        image_path: str,
        scl_path: Optional[str],
        aoi_geometry: Dict,
        output_dir: str
    ) -> Dict[str, str]:
        """
        Run complete harmonization pipeline
        
        Args:
            image_path: Path to input image
            scl_path: Path to scene classification layer (optional)
            aoi_geometry: AOI geometry dictionary
            output_dir: Output directory
            
        Returns:
            Dictionary of output file paths
        """
        logger.info("Starting harmonization pipeline")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        outputs = {}
        
        # Step 1: CRS Normalization
        norm_path = str(output_dir / 'normalized.tif')
        self.crs_normalizer.normalize_raster(image_path, norm_path)
        outputs['normalized'] = norm_path
        
        # Step 2: Cloud Masking (if SCL available)
        if scl_path:
            masked_path = str(output_dir / 'masked.tif')
            self.cloud_masker.apply_cloud_mask(norm_path, scl_path, masked_path)
            outputs['masked'] = masked_path
            current_path = masked_path
        else:
            current_path = norm_path
        
        # Step 3: AOI Clipping
        clipped_path = str(output_dir / 'clipped.tif')
        self.aoi_clipper.clip_raster(current_path, aoi_geometry, clipped_path)
        outputs['clipped'] = clipped_path
        
        # Step 4: Calculate Vegetation Indices
        vi_dir = output_dir / 'vegetation_indices'
        vi_paths = self.vi_calculator.compute_indices(clipped_path, str(vi_dir))
        outputs.update(vi_paths)
        
        logger.info("Harmonization pipeline complete")
        return outputs


if __name__ == "__main__":
    # Example usage
    pipeline = HarmonizationPipeline()
    
    # Example AOI (Delhi region)
    aoi = {
        "type": "Polygon",
        "coordinates": [[
            [77.5, 28.4],
            [77.7, 28.4],
            [77.7, 28.6],
            [77.5, 28.6],
            [77.5, 28.4]
        ]]
    }
    
    print("Harmonization pipeline initialized")
