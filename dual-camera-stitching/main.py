import cv2
import numpy as np
import time
from typing import Tuple, Optional, List

class FeatureDetector:
    def __init__(self, max_corners=500, quality_level=0.1, min_distance=10, block_size=7, k=0.04):
        self.max_corners = max_corners
        self.quality_level = quality_level
        self.min_distance = min_distance
        self.block_size = block_size
        self.k = k
    
    def detect_corners(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        
        corners = cv2.cornerHarris(gray.astype(np.float32), self.block_size, 3, self.k)
        corners = cv2.dilate(corners, None)
        
        threshold = self.quality_level * corners.max()
        corner_coords = np.where(corners > threshold)
        corner_points = np.column_stack((corner_coords[1], corner_coords[0]))
        
        if len(corner_points) > 5000:
            responses = [corners[int(corner[1]), int(corner[0])] for corner in corner_points]
            indices = np.argsort(responses)[-5000:]
            corner_points = corner_points[indices]
        
        if len(corner_points) > 0:
            corner_points = self._non_max_suppression_fast(corner_points, corners, self.min_distance)
        
        if len(corner_points) > self.max_corners:
            responses = [corners[int(corner[1]), int(corner[0])] for corner in corner_points]
            indices = np.argsort(responses)[-self.max_corners:]
            corner_points = corner_points[indices]
            
        return corner_points
    
    def _non_max_suppression_fast(self, corners: np.ndarray, response: np.ndarray, min_distance: int) -> np.ndarray:
        if len(corners) == 0:
            return corners
        
        responses = np.array([response[int(corner[1]), int(corner[0])] for corner in corners])
        sorted_indices = np.argsort(responses)[::-1]
        
        selected = []
        grid_size = min_distance
        occupied_cells = set()
        
        for idx in sorted_indices:
            corner = corners[idx]
            grid_x = int(corner[0] // grid_size)
            grid_y = int(corner[1] // grid_size)
            
            is_suppressed = False
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    cell = (grid_x + dx, grid_y + dy)
                    if cell in occupied_cells:
                        for selected_corner in selected:
                            dist = np.sqrt(np.sum((corner - selected_corner) ** 2))
                            if dist < min_distance:
                                is_suppressed = True
                                break
                        if is_suppressed:
                            break
                if is_suppressed:
                    break
            
            if not is_suppressed:
                selected.append(corner)
                occupied_cells.add((grid_x, grid_y))
                
                if len(selected) >= self.max_corners:
                    break
        
        return np.array(selected) if selected else np.array([]).reshape(0, 2)

class DescriptorExtractor:
    def __init__(self, patch_size=24):
        self.patch_size = patch_size
        self.half_size = patch_size // 2
    
    def extract_descriptors(self, image: np.ndarray, keypoints: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if len(keypoints) == 0:
            return np.array([]).reshape(0, self.patch_size * self.patch_size), np.array([]).reshape(0, 2)
            
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        descriptors = []
        valid_keypoints = []
        
        for kp in keypoints:
            x, y = int(kp[0]), int(kp[1])
            
            if (x - self.half_size >= 2 and x + self.half_size < gray.shape[1] - 2 and
                y - self.half_size >= 2 and y + self.half_size < gray.shape[0] - 2):
                
                patch = gray[y - self.half_size:y + self.half_size,
                           x - self.half_size:x + self.half_size]
                
                patch = cv2.GaussianBlur(patch, (3, 3), 0)
                patch = patch.astype(np.float32)
                mean_val = np.mean(patch)
                std_val = np.std(patch)
                
                if std_val > 1e-7:
                    patch = (patch - mean_val) / std_val
                    descriptors.append(patch.flatten())
                    valid_keypoints.append(kp)
        
        return (np.array(descriptors) if descriptors else np.array([]).reshape(0, self.patch_size * self.patch_size),
                np.array(valid_keypoints) if valid_keypoints else np.array([]).reshape(0, 2))

class FeatureMatcher:
    def __init__(self, ratio_threshold=0.9, distance_threshold=0.3):
        self.ratio_threshold = ratio_threshold
        self.distance_threshold = distance_threshold
    
    def match_features(self, desc1: np.ndarray, desc2: np.ndarray) -> List[Tuple[int, int]]:
        if len(desc1) == 0 or len(desc2) == 0:
            return []
        
        matches = []
        
        for i, d1 in enumerate(desc1):
            correlations = []
            for j, d2 in enumerate(desc2):
                try:
                    corr_matrix = np.corrcoef(d1, d2)
                    if corr_matrix.shape == (2, 2):
                        correlation = corr_matrix[0, 1]
                    else:
                        correlation = 0
                    
                    if np.isnan(correlation) or np.isinf(correlation):
                        correlation = 0
                        
                    dot_product = np.dot(d1, d2)
                    norm1 = np.linalg.norm(d1)
                    norm2 = np.linalg.norm(d2)
                    
                    if norm1 > 1e-7 and norm2 > 1e-7:
                        cosine_sim = dot_product / (norm1 * norm2)
                        if abs(cosine_sim) > abs(correlation):
                            correlation = cosine_sim
                    
                except:
                    correlation = 0
                    
                correlations.append((correlation, j))
            
            correlations.sort(reverse=True, key=lambda x: x[0])
            
            if len(correlations) >= 1:
                best_corr, best_idx = correlations[0]
                
                if best_corr > self.distance_threshold:
                    if len(correlations) >= 2:
                        second_corr, _ = correlations[1]
                        if abs(second_corr) < 1e-7:
                            matches.append((i, best_idx))
                        elif best_corr / abs(second_corr) > self.ratio_threshold:
                            matches.append((i, best_idx))
                    else:
                        matches.append((i, best_idx))
        
        if len(matches) < 10:
            matches = []
            relaxed_threshold = max(0.1, self.distance_threshold * 0.5)
            
            for i, d1 in enumerate(desc1):
                correlations = []
                for j, d2 in enumerate(desc2):
                    try:
                        correlation = np.dot(d1, d2) / (np.linalg.norm(d1) * np.linalg.norm(d2) + 1e-7)
                        if np.isnan(correlation):
                            correlation = 0
                    except:
                        correlation = 0
                    correlations.append((correlation, j))
                
                correlations.sort(reverse=True, key=lambda x: x[0])
                if len(correlations) > 0:
                    best_corr, best_idx = correlations[0]
                    if best_corr > relaxed_threshold:
                        matches.append((i, best_idx))
        
        return matches

class RANSACHomography:
    def __init__(self, threshold=8.0, max_iterations=3000, min_inliers=6):
        self.threshold = threshold
        self.max_iterations = max_iterations
        self.min_inliers = min_inliers
    
    def estimate_homography(self, src_pts: np.ndarray, dst_pts: np.ndarray) -> Optional[np.ndarray]:
        if len(src_pts) < 4 or len(dst_pts) < 4:
            return None
        
        best_homography = None
        best_inliers = 0
        
        for iteration in range(self.max_iterations):
            try:
                indices = np.random.choice(len(src_pts), 4, replace=False)
                sample_src = src_pts[indices]
                sample_dst = dst_pts[indices]
                
                H = self._compute_homography_4points(sample_src, sample_dst)
                
                if H is not None:
                    inliers = self._count_inliers(src_pts, dst_pts, H)
                    
                    if inliers > best_inliers:
                        best_inliers = inliers
                        best_homography = H
                        
                        if inliers >= len(src_pts) * 0.6:
                            break
                            
            except Exception as e:
                continue
        
        if best_inliers >= self.min_inliers:
            return best_homography
        else:
            if best_inliers >= max(4, len(src_pts) * 0.2):
                return best_homography
            return None
    
    def _compute_homography_4points(self, src_pts: np.ndarray, dst_pts: np.ndarray) -> Optional[np.ndarray]:
        try:
            if np.any(np.isnan(src_pts)) or np.any(np.isnan(dst_pts)):
                return None
            
            A = []
            for i in range(4):
                x, y = src_pts[i]
                u, v = dst_pts[i]
                
                A.append([x, y, 1, 0, 0, 0, -u*x, -u*y, -u])
                A.append([0, 0, 0, x, y, 1, -v*x, -v*y, -v])
            
            A = np.array(A, dtype=np.float64)
            
            if np.linalg.cond(A) > 1e12:
                return None
            
            _, _, V = np.linalg.svd(A)
            h = V[-1, :]
            H = h.reshape(3, 3)
            
            if abs(H[2, 2]) > 1e-8:
                H = H / H[2, 2]
                if np.any(np.isnan(H)) or np.any(np.isinf(H)):
                    return None
                return H
            else:
                return None
                
        except Exception as e:
            return None
    
    def _count_inliers(self, src_pts: np.ndarray, dst_pts: np.ndarray, H: np.ndarray) -> int:
        try:
            src_homogeneous = np.column_stack([src_pts, np.ones(len(src_pts))])
            transformed = np.dot(H, src_homogeneous.T).T
            transformed = transformed[:, :2] / transformed[:, 2:3]
            distances = np.sqrt(np.sum((transformed - dst_pts) ** 2, axis=1))
            return np.sum(distances < self.threshold)
        except:
            return 0

class ImageStitcher:
    def __init__(self):
        self.feature_detector = FeatureDetector()
        self.descriptor_extractor = DescriptorExtractor()
        self.feature_matcher = FeatureMatcher()
        self.ransac = RANSACHomography()
    
    def stitch_images(self, img1: np.ndarray, img2: np.ndarray, debug=False) -> Optional[np.ndarray]:
        kp1 = self.feature_detector.detect_corners(img1)
        kp2 = self.feature_detector.detect_corners(img2)
        
        if len(kp1) < 4 or len(kp2) < 4:
            return None
        
        desc1, valid_kp1 = self.descriptor_extractor.extract_descriptors(img1, kp1)
        desc2, valid_kp2 = self.descriptor_extractor.extract_descriptors(img2, kp2)
        
        if len(desc1) == 0 or len(desc2) == 0:
            return None
        
        matches = self.feature_matcher.match_features(desc1, desc2)
        
        if len(matches) < 4:
            if debug:
                self._debug_visualize(img1, img2, valid_kp1, valid_kp2, matches)
            return None
        
        src_pts = np.array([valid_kp1[m[0]] for m in matches])
        dst_pts = np.array([valid_kp2[m[1]] for m in matches])
        
        result = self._simple_stitch_with_blend(img1, img2, src_pts, dst_pts, debug)
        
        if debug and result is not None:
            self._debug_visualize(img1, img2, valid_kp1, valid_kp2, matches)
        
        return result
    
    def _simple_stitch_with_blend(self, img1: np.ndarray, img2: np.ndarray, 
                                 src_pts: np.ndarray, dst_pts: np.ndarray, debug=False) -> np.ndarray:
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        
        dx = np.mean(dst_pts[:, 0] - src_pts[:, 0])
        dy = np.mean(dst_pts[:, 1] - src_pts[:, 1])
        
        if abs(dx) > abs(dy):
            if dx > 0:
                return self._horizontal_blend(img1, img2, int(dx), int(dy))
            else:
                return self._horizontal_blend(img2, img1, int(-dx), int(-dy))
        else:
            if dy > 0:
                return self._vertical_blend(img1, img2, int(dx), int(dy))
            else:
                return self._vertical_blend(img2, img1, int(-dx), int(-dy))
    
    def _horizontal_blend(self, left_img: np.ndarray, right_img: np.ndarray, 
                         offset_x: int, offset_y: int) -> np.ndarray:
        """Blend two images horizontally with smooth transition"""
        h1, w1 = left_img.shape[:2]
        h2, w2 = right_img.shape[:2]
        
        # Calculate overlap
        overlap_width = max(0, w1 - offset_x)
        print(f"Horizontal overlap width: {overlap_width} pixels")
        
        if overlap_width <= 0:
            # No overlap, simple concatenation
            print("No overlap detected, using simple concatenation")
            # Make heights equal
            target_h = max(h1, h2)
            left_resized = cv2.resize(left_img, (w1, target_h)) if h1 != target_h else left_img
            right_resized = cv2.resize(right_img, (w2, target_h)) if h2 != target_h else right_img
            
            # Add spacing between images
            spacing = 20
            result = np.zeros((target_h, w1 + w2 + spacing, 3), dtype=np.uint8)
            result[:, :w1] = left_resized
            result[:, w1+spacing:] = right_resized
            return result
        
        # Calculate canvas size
        canvas_width = w1 + w2 - overlap_width
        canvas_height = max(h1, h2 + abs(offset_y))
        
        print(f"Canvas size: {canvas_width}x{canvas_height}")
        
        # Create canvas
        result = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
        
        # Place left image
        left_y_start = max(0, -offset_y)
        left_y_end = left_y_start + h1
        result[left_y_start:left_y_end, :w1] = left_img
        
        # Calculate right image position
        right_x_start = offset_x
        right_y_start = max(0, offset_y)
        right_y_end = right_y_start + h2
        
        # Ensure we don't go out of bounds
        right_y_end = min(right_y_end, canvas_height)
        right_x_end = min(right_x_start + w2, canvas_width)
        
        if right_x_start < canvas_width and right_y_start < canvas_height:
            # Calculate the actual region we can place
            actual_w2 = min(w2, right_x_end - right_x_start)
            actual_h2 = min(h2, right_y_end - right_y_start)
            
            if overlap_width > 20:  # Significant overlap, apply blending
                # Non-overlap region of right image
                non_overlap_start = overlap_width
                if non_overlap_start < actual_w2:
                    result[right_y_start:right_y_start + actual_h2, 
                           right_x_start + non_overlap_start:right_x_end] = \
                        right_img[:actual_h2, non_overlap_start:actual_w2]
                
                # Blend in overlap region
                overlap_left_start = w1 - overlap_width
                overlap_region_width = min(overlap_width, actual_w2)
                
                if overlap_region_width > 0:
                    # Create blending weights
                    weights = np.linspace(1, 0, overlap_region_width)
                    
                    for i in range(overlap_region_width):
                        weight_left = weights[i]
                        weight_right = 1 - weight_left
                        
                        x_pos = overlap_left_start + i
                        if x_pos < canvas_width and i < actual_w2:
                            # Get the overlapping regions
                            left_y_range = slice(left_y_start, min(left_y_end, canvas_height))
                            right_y_range = slice(right_y_start, min(right_y_end, canvas_height))
                            
                            # Find common y range
                            common_y_start = max(left_y_start, right_y_start)
                            common_y_end = min(left_y_end, right_y_end)
                            
                            if common_y_start < common_y_end:
                                left_patch = result[common_y_start:common_y_end, x_pos]
                                right_patch = right_img[common_y_start - right_y_start:common_y_end - right_y_start, i]
                                
                                # Blend
                                blended = (weight_left * left_patch.astype(np.float32) + 
                                         weight_right * right_patch.astype(np.float32))
                                result[common_y_start:common_y_end, x_pos] = blended.astype(np.uint8)
            else:
                # Small or no overlap, just place the right image
                result[right_y_start:right_y_end, right_x_start:right_x_start + actual_w2] = \
                    right_img[:actual_h2, :actual_w2]
        
        return result
    
    def _vertical_blend(self, top_img: np.ndarray, bottom_img: np.ndarray, 
                       offset_x: int, offset_y: int) -> np.ndarray:
        """Blend two images vertically with smooth transition"""
        h1, w1 = top_img.shape[:2]
        h2, w2 = bottom_img.shape[:2]
        
        # Calculate overlap
        overlap_height = max(0, h1 - offset_y)
        print(f"Vertical overlap height: {overlap_height} pixels")
        
        if overlap_height <= 0:
            # No overlap, simple concatenation
            target_w = max(w1, w2)
            top_resized = cv2.resize(top_img, (target_w, h1)) if w1 != target_w else top_img
            bottom_resized = cv2.resize(bottom_img, (target_w, h2)) if w2 != target_w else bottom_img
            
            spacing = 20
            result = np.zeros((h1 + h2 + spacing, target_w, 3), dtype=np.uint8)
            result[:h1, :] = top_resized
            result[h1+spacing:, :] = bottom_resized
            return result
        
        # Calculate canvas size
        canvas_height = h1 + h2 - overlap_height
        canvas_width = max(w1, w2 + abs(offset_x))
        
        result = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
        
        # Place top image
        result[:h1, :w1] = top_img
        
        # Place bottom image with blending
        bottom_y_start = offset_y
        if overlap_height > 20:  # Apply blending
            # Non-overlap region
            result[bottom_y_start + overlap_height:bottom_y_start + h2, :w2] = \
                bottom_img[overlap_height:, :]
            
            # Blend overlap region
            for i in range(overlap_height):
                weight_top = (overlap_height - i) / overlap_height
                weight_bottom = i / overlap_height
                
                y_pos = h1 - overlap_height + i
                common_width = min(w1, w2)
                
                blended = (weight_top * result[y_pos, :common_width].astype(np.float32) + 
                          weight_bottom * bottom_img[i, :common_width].astype(np.float32))
                result[y_pos, :common_width] = blended.astype(np.uint8)
        else:
            # Simple placement
            result[bottom_y_start:bottom_y_start + h2, :w2] = bottom_img
        
        return result
    
    def _debug_visualize(self, img1, img2, kp1, kp2, matches):
        try:
            h1, w1 = img1.shape[:2]
            h2, w2 = img2.shape[:2]
            
            vis_img = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
            vis_img[:h1, :w1] = img1
            vis_img[:h2, w1:w1+w2] = img2
            
            for kp in kp1:
                cv2.circle(vis_img, (int(kp[0]), int(kp[1])), 3, (0, 255, 0), -1)
            
            for kp in kp2:
                cv2.circle(vis_img, (int(kp[0]) + w1, int(kp[1])), 3, (0, 255, 0), -1)
            
            for match in matches:
                pt1 = kp1[match[0]]
                pt2 = kp2[match[1]]
                cv2.line(vis_img, 
                        (int(pt1[0]), int(pt1[1])),
                        (int(pt2[0]) + w1, int(pt2[1])),
                        (255, 0, 0), 1)
            
            cv2.imwrite('debug_matches.jpg', vis_img)
        except Exception as e:
            pass
class DualCameraStitcher:
    def __init__(self, camera1_id=0, camera2_id=1):
        self.camera1_id = camera1_id
        self.camera2_id = camera2_id
        self.stitcher = ImageStitcher()
        self.cap1 = None
        self.cap2 = None
        
    def initialize_cameras(self):
        self.cap1 = cv2.VideoCapture(self.camera1_id)
        self.cap2 = cv2.VideoCapture(self.camera2_id)
        
        if not self.cap1.isOpened() or not self.cap2.isOpened():
            return False
        
        self.cap1.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap2.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        return True
    
    def run_real_time_stitching(self):
        if not self.initialize_cameras():
            return
        
        frame_count = 0
        fps_start_time = time.time()
        
        try:
            while True:
                ret1, frame1 = self.cap1.read()
                ret2, frame2 = self.cap2.read()
                
                if not ret1 or not ret2:
                    break
                
                frame1 = cv2.resize(frame1, (320, 240))
                frame2 = cv2.resize(frame2, (320, 240))
                
                stitched = self.stitcher.stitch_images(frame1, frame2)
                
                cv2.imshow('Camera 1', frame1)
                cv2.imshow('Camera 2', frame2)
                
                if stitched is not None:
                    display_stitched = cv2.resize(stitched, (800, 400))
                    cv2.imshow('Stitched Result', display_stitched)
                else:
                    side_by_side = np.hstack([frame1, frame2])
                    cv2.imshow('Stitched Result', side_by_side)
                
                frame_count += 1
                if frame_count % 30 == 0:
                    fps = 30 / (time.time() - fps_start_time)
                    fps_start_time = time.time()
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s') and stitched is not None:
                    cv2.imwrite('stitched_result.jpg', stitched)
                
        finally:
            self.cleanup()
    
    def process_static_images(self, img1_path: str, img2_path: str, output_path: str = 'stitched_output.jpg', debug: bool = True):
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)
        
        if img1 is None or img2 is None:
            return None
        
        max_dim = 800
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        
        if max(h1, w1) > max_dim:
            scale = max_dim / max(h1, w1)
            new_w1, new_h1 = int(w1 * scale), int(h1 * scale)
            img1 = cv2.resize(img1, (new_w1, new_h1))
            
        if max(h2, w2) > max_dim:
            scale = max_dim / max(h2, w2)
            new_w2, new_h2 = int(w2 * scale), int(h2 * scale)
            img2 = cv2.resize(img2, (new_w2, new_h2))
        
        stitched = self.stitcher.stitch_images(img1, img2, debug=debug)
        
        if stitched is not None:
            cv2.imwrite(output_path, stitched)
        
        return stitched
    
    def cleanup(self):
        if self.cap1:
            self.cap1.release()
        if self.cap2:
            self.cap2.release()
        cv2.destroyAllWindows()

def main():
    stitcher = DualCameraStitcher()
def main():
    stitcher = DualCameraStitcher()
    
    while True:
        print("\n1. Real-time camera stitching")
        print("2. Process static images") 
        print("3. Test with sample images")
        print("4. Exit")
        
        choice = input("Enter choice (1-4): ").strip()
        
        if choice == '1':
            stitcher.run_real_time_stitching()
        elif choice == '2':
            img1_path = input("First image path: ").strip()
            img2_path = input("Second image path: ").strip()
            output_path = input("Output path (default: stitched_output.jpg): ").strip()
            if not output_path:
                output_path = 'stitched_output.jpg'
            stitcher.process_static_images(img1_path, img2_path, output_path)
        elif choice == '3':
            import os
            if os.path.exists('img/left.jpg') and os.path.exists('img/right.jpg'):
                result = stitcher.process_static_images('img/left.jpg', 'img/right.jpg', 'test_stitched_lr.jpg')
                if result is not None:
                    img1 = cv2.imread('img/left.jpg')
                    img2 = cv2.imread('img/right.jpg')
                    
                    max_h = 400
                    h1, w1 = img1.shape[:2]
                    h2, w2 = img2.shape[:2]
                    
                    if h1 > max_h:
                        scale = max_h / h1
                        img1 = cv2.resize(img1, (int(w1 * scale), max_h))
                    if h2 > max_h:
                        scale = max_h / h2
                        img2 = cv2.resize(img2, (int(w2 * scale), max_h))
                    
                    h1, w1 = img1.shape[:2]
                    h2, w2 = img2.shape[:2]
                    target_h = min(h1, h2)
                    
                    img1 = cv2.resize(img1, (int(w1 * target_h / h1), target_h))
                    img2 = cv2.resize(img2, (int(w2 * target_h / h2), target_h))
                    
                    side_by_side = np.hstack([img1, img2])
                    cv2.imwrite('reference_side_by_side.jpg', side_by_side)
        elif choice == '4':
            break

if __name__ == "__main__":
    main()