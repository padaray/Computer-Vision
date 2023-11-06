import cv2

def getKeyPoints(image_path):

    if not image_path.strip():
        print("請選擇圖片")

    # Load the image
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Create a SIFT detector
    sift = cv2.SIFT_create()

    # Detect keypoints and compute descriptors
    kp, des = sift.detectAndCompute(gray, None)

    # Draw the keypoints on the image
    image_with_keypoints = cv2.drawKeypoints(image, kp, None, color=(0, 255, 0))

    # Display the image with keypoints
    cv2.namedWindow("Image with SIFT Keypoints", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Image with SIFT Keypoints", 1200, 800)
    cv2.imshow('Image with SIFT Keypoints', image_with_keypoints)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def getMacheKeypoints(image_path1, image_path2):

    if len(image_path1) == 0:
        print("請選擇圖片1")

    if len(image_path2) == 0:
        print("請選擇圖片2")

    # Load Image 1 (Left.jpg) and Image 2 (Right.jpg)
    image1 = cv2.imread(image_path1, cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread(image_path2, cv2.IMREAD_GRAYSCALE)

    # Create SIFT detector
    sift = cv2.SIFT_create()

    # Find keypoints and descriptors for Image 1 and Image 2
    keypoints1, descriptors1 = sift.detectAndCompute(image1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(image2, None)

    # Create a BFMatcher (Brute-Force Matcher) with default parameters
    bf = cv2.BFMatcher()

    # Match descriptors from Image 1 with descriptors from Image 2
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)

    # Apply ratio test to filter good matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    # Draw the matched keypoints
    result_image = cv2.drawMatches(image1, keypoints1, image2, keypoints2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Display the result
    cv2.namedWindow("Matches", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Matches", 1600, 800)
    cv2.imshow('Matches', result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()