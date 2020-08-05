"""

This project uses a pre tained YOLO network on a video file to detect Tennis Players.

This makes use of tiny-YOLO which is specifically  designed for a video file.
This project will work on Wimbledon matches only because in Wimbledon there is a dresscode where players must be dressed in white.

The YOLO detector detects for players, then checks if the number of white pixels on each player is beyond a certain threshold. If it is, a bounding box is drawn around the player.


"""
import cv2
import numpy as np


def main():

	# The following parameters need to be modified according to your local computer.

	path = "/Users/varundhingra/Documents/project/Tennis/"
	video_file = 'match.mp4'


	# Image Preprocessing parameters for input to YOLO

	normalization_scale = 1 / 255
	input_image_size = 320
	output_image_size = 600
	font = cv2.FONT_HERSHEY_PLAIN

	# White color pixels

	lower_white = np.array([0,0,168], dtype=np.uint8)
	upper_white = np.array([172,111,255], dtype=np.uint8)

	
	# Video capture and processing

	cap = cv2.VideoCapture(path + video_file)
	while True:

		_,img = cap.read()
	
		#Setup the YOLO network

		height, width, _ = img.shape
		net = cv2.dnn.readNet('yolov3-tiny.weights', 'yolov3-tiny.cfg')
		classes = []

		with open ("coco.names", "r") as f:
			classes = [line.strip() for line in f.readlines()]

	

	# Preprocess the Image for Input to the YOLO Network
		blob = cv2.dnn.blobFromImage(img, normalization_scale, (input_image_size, input_image_size), (0, 0, 0), swapRB = True, crop = False)
		net.setInput(blob)
		output_layer_names = net.getUnconnectedOutLayersNames()

	# Run the image through the Network
		layerOutputs = net.forward(output_layer_names)



		boxes = []
		confidences = []
		class_ids = []
		colors = np.random.uniform(0, 255, size = (len(classes), 3))

		for out in layerOutputs:
			for detection in out:
				scores = detection[ 5 : ]
				class_id = np.argmax(scores)
				confidence = scores[class_id]
				
				# if an object detected has high chance of being a person, isolate the person's location
				if confidence > 0.5:
					center_x = int(detection[0] * width)
					center_y = int(detection[1] * height)
					w = int(detection[2] * width)
					h = int(detection[3] * height)

					
				# Rectangle coordinates

					x = int(center_x - w / 2)
					y = int(center_y - h / 2)

					boxes.append([x, y, w, h])

					confidences.append(float(confidence))
					class_ids.append(class_id)


		no_of_objects_detected = len(boxes)

		# Since many boxes are dround around the same person, suppress them all using Non Maximal Suppression.

		indexes_post_suppression = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)


		# Separate the player from all the other people detected by the YOLO

		for index in range(no_of_objects_detected):
			if index in indexes_post_suppression:
				x, y, w, h = boxes[index]
				
				label = "player"
				color = colors[0]
				


				crop = img[y : y + h, x : x + w]
				if len(crop) > 0:

					# Get the bit mask of all the white pixels.

					crop_hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
					mask_white = cv2.inRange(crop_hsv, lower_white, upper_white)

					# Bitwise AND with original image
					masked_image = cv2.bitwise_and(crop, crop, mask = mask_white)

					# If number of white Pixels are above a certain threshold, then it means that that person is wearing a white shirt and white shorts. This means he is most probably a player.

					no_of_white_pixels = np.count_nonzero(masked_image)
					if no_of_white_pixels > 3000:
						cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)


						cv2.putText(img, label, (x, y + 30), font, 3, color, 3)
		

		img = cv2.resize(img, (output_image_size, output_image_size))
		cv2.imshow("Office_detections_output", img)
		key = cv2.waitKey(1)
		if key == 27:
			break
	cap.release()
	cv2.destroyAllWindows()


main()