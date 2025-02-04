import cv2 as cv
import numpy as np
import os


class SIFTProcessor:
    """Handles image processing using the SIFT algorithm."""

    def __init__(self):
        """Initialize the SIFT detector."""
        self.sift = cv.SIFT_create()

    def detect_keypoints(self, img):
        """
        Detect keypoints and compute descriptors using SIFT.

        :param img: Grayscale input image
        :return: Keypoints and descriptors
        """
        keypoints, descriptors = self.sift.detectAndCompute(img, None)
        return keypoints, descriptors

    def draw_keypoints(self, img, keypoints, flag=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS):
        """
        Draw keypoints on the image.

        :param img: Input image
        :param keypoints: Keypoints detected using SIFT
        :param flag: Drawing style for keypoints
        :return: Image with keypoints drawn
        """
        return cv.drawKeypoints(img, keypoints, None, flags=flag)

    def match_images(self, img1, img2):
        """
        Match keypoints between two images using BFMatcher.

        :param img1: First grayscale image
        :param img2: Second grayscale image
        :return: Image showing matched keypoints
        """
        keypoints1, descriptors1 = self.detect_keypoints(img1)
        keypoints2, descriptors2 = self.detect_keypoints(img2)

        # Ensure both images have keypoints before matching
        if descriptors1 is None or descriptors2 is None:
            raise ValueError("One or both images have no valid keypoints/descriptors.")

        # Create a Brute-Force matcher with L2 norm
        bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)
        matches = bf.match(descriptors1, descriptors2)

        # Sort matches based on distance (lower distance means better match)
        matches = sorted(matches, key=lambda x: x.distance)

        # Draw the top 50 matches
        matched_img = cv.drawMatches(img1, keypoints1, img2, keypoints2, matches[:50], None,
                                     flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        return matched_img

    def process_image(self, img_path, output_dir, feature_type="keypoints"):
        """
        Process an image based on the selected feature type.

        :param img_path: Path to the input image
        :param output_dir: Directory to save processed images
        :param feature_type: Type of feature extraction ("keypoints" or "descriptors")
        :return: Path to the output image or descriptor file
        """
        img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)  # Load image in grayscale
        if img is None:
            raise FileNotFoundError(f"Error: Unable to read image at {img_path}")

        # Generate output filename based on input filename and feature type
        output_path = os.path.join(output_dir, f"{os.path.basename(img_path).split('.')[0]}_{feature_type}.png")

        if feature_type == "keypoints":
            # Detect keypoints and draw them on the image
            keypoints, _ = self.detect_keypoints(img)
            result_img = self.draw_keypoints(img, keypoints)
            cv.imwrite(output_path, result_img)  # Save keypoints image
        elif feature_type == "descriptors":
            # Extract descriptors and save them in a structured format
            _, descriptors = self.detect_keypoints(img)
            if descriptors is None:
                raise ValueError("No descriptors found in the image.")
            output_path = output_path.replace(".png", ".npy")
            np.save(output_path, descriptors)  # Save descriptors as NumPy array
        else:
            raise ValueError(f"Unsupported feature type: {feature_type}")

        return output_path  # Return path of the processed file
