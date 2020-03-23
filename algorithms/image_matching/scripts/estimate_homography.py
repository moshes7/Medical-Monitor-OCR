"""
Based on:
https://www.learnopencv.com/image-alignment-feature-based-using-opencv-c-python/
"""

from __future__ import print_function
import cv2
import numpy as np
import os

MAX_FEATURES = 500
GOOD_MATCH_PERCENT = 0.15


def alignImages(im1, im2):
    # Convert images to grayscale
    im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    # Detect ORB features and compute descriptors.
    orb = cv2.ORB_create(MAX_FEATURES)
    keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)

    # Match features.
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)

    # Sort matches by score
    matches.sort(key=lambda x: x.distance, reverse=False)

    # Remove not so good matches
    numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:numGoodMatches]

    # Draw top matches
    imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
    # cv2.imwrite("matches.jpg", imMatches)

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    # Find homography
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

    # Use homography
    height, width, channels = im2.shape
    im1Reg = cv2.warpPerspective(im1, h, (width, height))

    return im1Reg, h, imMatches


if __name__ == '__main__':


    img1_path = '../images/4/3.tiff'
    img2_path = '../images/4/2.tiff'

    output_dir = 'output'

    output_dir = os.path.join(os.path.dirname(__file__), output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Read reference image
    # refFilename = "form.jpg"
    refFilename = img1_path
    print("Reading reference image : ", refFilename)
    imReference = cv2.imread(refFilename, cv2.IMREAD_COLOR)
    # save image
    save_fig_name = os.path.join(output_dir, '1.jpg')
    cv2.imwrite(save_fig_name, imReference)


    # Read image to be aligned
    # imFilename = "scanned-form.jpg"
    imFilename = img2_path
    print("Reading image to align : ", imFilename);
    im = cv2.imread(imFilename, cv2.IMREAD_COLOR)
    # save image
    save_fig_name = os.path.join(output_dir, '2.jpg')
    cv2.imwrite(save_fig_name, im)

    print("Aligning images ...")
    # Registered image will be resotred in imReg.
    # The estimated homography will be stored in h.
    imReg, h, imMatch = alignImages(im, imReference)
    # save images
    save_fig_name = os.path.join(output_dir, 'matches.jpg')
    cv2.imwrite(save_fig_name, imMatch)
    save_fig_name = os.path.join(output_dir, 'aligned.jpg')
    cv2.imwrite(save_fig_name, imReg)

    # Print estimated homography
    print("Estimated homography : \n", h)
