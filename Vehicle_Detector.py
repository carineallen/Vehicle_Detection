import cv2

# The car image
image_file = 'images/car_image.jpg'
video = cv2.VideoCapture('videos/Tesla_Saving_Lives_Compilation_Trim3.mp4')

# pre-trained car classifier
car_classifier_file = 'cars.xml'


# create car classifier
car_detector = cv2.CascadeClassifier(car_classifier_file)

# Infinity loop to read each video frame
while True:

    # Read the current frame
    (read_successful, frame) = video.read()

    #check if the read was successful
    if read_successful:
        # Convert to grayscale
        grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break

    # Detecting car
    ScaleValue = 1.7 #Parameter specifying how much the image size is reduced at each image scale.
    Neighbor = 2 #Parameter specifying how many neighbors each candidate rectangle should have
    cars = car_detector.detectMultiScale(grayscaled_frame,ScaleValue,Neighbor)

    # Draw the rectangles around the detected cars
    for (x,y,w,h) in cars:
        cv2.rectangle(frame, (x,y), (x+w,y+h),(0,0,255), 2)

    cv2.namedWindow('CAS vehicle detector', cv2.WINDOW_NORMAL)

    # Display image with cars detected
    cv2.imshow('CAS vehicle detector', frame)

    # wait for the user to press a key
    key = cv2.waitKey(1)

    # if ENTER is press, exit the loop
    if key==10 or key==13:
        break

# Stop readinf video file
video.release
