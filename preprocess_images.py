import os
import cv2
import face_recognition
import numpy as np
from pathlib import Path
import gc
import dlib
from datetime import datetime


class ImagePreprocessor:
    def __init__(self, input_dir="raw_images", output_dir="processed_images"):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self._create_dirs()
        self.batch_size = 1
        self.target_size = (128, 128)
        self.max_retries = 2
        self.current_model = "cnn"  # Start with CNN since you have GPU

    def _create_dirs(self):
        (self.output_dir / "faces").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "metadata").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "failed").mkdir(parents=True, exist_ok=True)

    def _release_resources(self):
        """Aggressive memory cleanup"""
        cv2.destroyAllWindows()
        gc.collect()
        if dlib.DLIB_USE_CUDA:
            try:
                dlib.cuda.set_device(0)
                from ctypes import cdll
                # Attempt loading matching cudart version (try newer ones first)
                for dll_name in ["cudart64_125.dll", "cudart64_122.dll", "cudart64_118.dll"]:
                    try:
                        cudart = cdll.LoadLibrary(dll_name)
                        cudart.cudaDeviceReset()
                        print(f"[CUDA] Successfully reset device using {dll_name}")
                        break
                    except OSError:
                        continue
                else:
                    print("[Non-critical] CUDA cleanup failed: No matching cudart DLL found.")
            except Exception as e:
                print(f"[Non-critical] CUDA cleanup failed: {e}")

    def _safe_face_detection(self, image):
        # Resize to 1/4th size for faster CNN processing
        small_img = cv2.resize(image, (0, 0), fx=0.25, fy=0.25)

        try:
            face_locations_small = face_recognition.face_locations(
                small_img,
                number_of_times_to_upsample=1,
                model="cnn"
            )

            # Scale back coordinates to original size
            face_locations = []
            for top, right, bottom, left in face_locations_small:
                face_locations.append((
                    top * 4,
                    right * 4,
                    bottom * 4,
                    left * 4
                ))

            return face_locations

        except Exception as e:
            print(f"[Error] Face detection failed: {e}")
            return []

    def _detect_and_crop_face(self, image, save_failed_path=None):
        try:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, (640, 480))

            face_locations = self._safe_face_detection(rgb)

            if not face_locations:
                if save_failed_path:
                    cv2.imwrite(str(save_failed_path), image)
                return None

            top, right, bottom, left = face_locations[0]
            margin = int((bottom - top) * 0.1)
            top = max(0, top - margin)
            bottom = min(rgb.shape[0], bottom + margin)
            left = max(0, left - margin)
            right = min(rgb.shape[1], right + margin)

            face_image = rgb[top:bottom, left:right]
            return cv2.resize(face_image, self.target_size)
        except Exception as e:
            print(f"Face processing error: {str(e)}")
            if save_failed_path is not None:
                cv2.imwrite(str(save_failed_path), image)
            return None

    def _preprocess_image(self, image_path):
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"[Warning] Cannot read {image_path.name}")
                return None

            failed_image_path = self.output_dir / "failed" / image_path.name
            cropped_face = self._detect_and_crop_face(image, failed_image_path)
            if cropped_face is None:
                print(f"[Info] No face detected: {image_path.name}")
                return None

            return cv2.cvtColor(cropped_face, cv2.COLOR_RGB2BGR)
        except Exception as e:
            print(f"[Error] Processing {image_path.name}: {e}")
            return None
        finally:
            self._release_resources()

    def process_dataset(self):
        processed_count = 0
        skipped_count = 0

        for person_dir in self.input_dir.iterdir():
            if not person_dir.is_dir():
                continue

            person_name = person_dir.name
            output_person_dir = self.output_dir / "faces" / person_name
            output_person_dir.mkdir(parents=True, exist_ok=True)

            image_paths = [p for p in person_dir.glob("*") if p.suffix.lower() in [".jpg", ".jpeg", ".png"]]

            for image_path in image_paths:
                try:
                    processed_image = self._preprocess_image(image_path)
                    if processed_image is not None:
                        output_path = output_person_dir / image_path.name
                        cv2.imwrite(str(output_path), processed_image)
                        processed_count += 1
                    else:
                        skipped_count += 1
                except Exception as e:
                    print(f"[Failed] {image_path.name}: {str(e)}")
                    skipped_count += 1

        print(f"\n[Processing Complete]\n- Success: {processed_count}\n- Failed: {skipped_count}")


if __name__ == "__main__":
    print(f"[{datetime.now()}] GPU-enabled Face Preprocessing Started")
    preprocessor = ImagePreprocessor(input_dir="raw_images", output_dir="processed_images")
    preprocessor.process_dataset()
