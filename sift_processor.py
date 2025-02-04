import cv2 as cv
import os


class SIFTProcessor:
    def __init__(self):
        """Initialize the SIFT detector."""
        self.sift = cv.SIFT_create()

    def detect_keypoints(self, img):
        """Detect keypoints and compute descriptors using SIFT."""
        keypoints, descriptors = self.sift.detectAndCompute(img, None)
        return keypoints, descriptors

    def draw_keypoints(self, img, keypoints, flag=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS):
        """Draw keypoints on the image."""
        return cv.drawKeypoints(img, keypoints, None, flags=flag)

    def match_images(self, img1, img2):
        """Match features between two images using BFMatcher."""
        keypoints1, descriptors1 = self.detect_keypoints(img1)
        keypoints2, descriptors2 = self.detect_keypoints(img2)

        bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)
        matches = bf.match(descriptors1, descriptors2)
        matches = sorted(matches, key=lambda x: x.distance)

        matched_img = cv.drawMatches(img1, keypoints1, img2, keypoints2, matches[:50], None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        return matched_img

    def process_image(self, img_path, output_dir, feature_type="keypoints"):
        """Process an image with the selected feature type."""
        img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Error: Unable to read image at {img_path}")

        output_path = os.path.join(output_dir, f"{feature_type}_output.png")

        if feature_type == "keypoints":
            keypoints, _ = self.detect_keypoints(img)
            result_img = self.draw_keypoints(img, keypoints)
        elif feature_type == "descriptors":
            _, descriptors = self.detect_keypoints(img)
            # Save descriptors as a text file
            with open(output_path.replace(".png", ".txt"), "w") as f:
                f.write(str(descriptors))
            return output_path.replace(".png", ".txt")
        else:
            raise ValueError(f"Unsupported feature type: {feature_type}")

        cv.imwrite(output_path, result_img)
        return output_path
