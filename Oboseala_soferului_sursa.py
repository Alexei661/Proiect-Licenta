import cv2
from scipy.spatial import distance
from imutils import face_utils
from pygame import mixer
import imutils
import dlib
import time


#initializam mixer-ul si ii dam calea catre fisierul audio
mixer.init()
mixer.music.load("music.wav")

def eye_aspect_ratio(eye):
	#calculeaza distanta euclideana dintre cele 2 seturi de landmarkuri verticale ale ochilor
	A = distance.euclidean(eye[1], eye[5])
	B = distance.euclidean(eye[2], eye[4])
	#calculeaza distanta euclideana dintre landmarkurile orizontale ale ochilor
	C = distance.euclidean(eye[0], eye[3])
	ear = (A + B) / (2.0 * C)
	return ear
	
def mouth_aspect_ratio(mouth):
	#calculeaza distanta euclideana dintre cele 2 seturi de landmarkuri verticale ale gurii
	A = distance.euclidean(mouth[2], mouth[10])  # 51, 59
	B = distance.euclidean(mouth[4], mouth[8])  # 53, 57

	
	C = distance.euclidean(mouth[0], mouth[6])  # 49, 55

	# compute the mouth aspect ratio
	mar = (A + B) / (2.0 * C)

	# return the mouth aspect ratio
	return mar

eye_thresh = 0.25
frame_check = 20
mouth_thresh = 1.3
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

#extragem indecsii landmarkurilor faciale corespunzatoare ochilor
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["mouth"]
cap = cv2.VideoCapture(0)
counter = 0
yawns = 0
last_yawn=time.time()
yawn_interval=5
eye_not_detected_timer = None  
eye_not_detected_threshold = 1


while True:
	#se ia frame-ul din video, resize si il convertim in gri
	ret,frame = cap.read()
	frame = imutils.resize(frame, width=1024, height=576)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


	subjects = detect(gray, 0)
	alarm_triggered=False
	
	#verificam daca sunt detectati ochii
	if len(subjects) == 0:
			#daca nu sunt detectai ochii atunci se va da drumul la timer
			if eye_not_detected_timer is None:
				eye_not_detected_timer = time.time()
				if not alarm_triggered:
					alarm_triggered = True
			#de indata ce au trecut 5 secunde de cand nu au fost detectati ochii
			elif time.time() - eye_not_detected_timer > eye_not_detected_threshold:
				mixer.music.play()
	else:
		# Dacă ochii sunt detectați, opriți alarma și resetați timestamp-ul
		if alarm_triggered:
			mixer.music.stop()
			alarm_triggered = False
		eye_not_detected_timer = None

	#loop pentru detectia fetelor
	for subject in subjects:
		#determinam landmarkurile faciale, dupa care le convertim intr-un array numpy
		shape = predict(gray, subject)
		shape = face_utils.shape_to_np(shape)

		#partea cu ochii
		#extragem coordonatele pentru ochi, dupa care calculam ear pentru ambii ochi
		leftEye = shape[lStart:lEnd]
		rightEye = shape[rStart:rEnd]
		leftEAR = eye_aspect_ratio(leftEye)
		rightEAR = eye_aspect_ratio(rightEye)

		#facem media celord doi ochi pentru ear
		ear = (leftEAR + rightEAR) / 2.0

		#obtinem formele convexe pentru ochi dupa care le afisam
		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)
		cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

		#daca ear<thresh crestem counterul
		if ear < eye_thresh:
			counter += 1
			print (counter)
			#daca counterul a depasit numarul de frame-uri atunci pornim alarma
			if counter >= frame_check:
				cv2.putText(frame, "TREZIREA!!!", (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
				mixer.music.play()
		#altfel, counterul ramane pe 0		
		else:
			counter = 0
				
		#partea cu gura
		#extragem coordonatele pentru gura
		mouth = shape[mStart:mEnd]

		#dupa care calculam mar-ul
		mar = mouth_aspect_ratio(mouth)

		#obtinem forma convexa pentru gura dupa care o afisam
		mouthHull = cv2.convexHull(mouth)
		cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)
		
		cv2.putText(frame, "cascaturi: {:.0f}".format(yawns), (650, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

		cv2.putText(frame, "MAR: {:.2f}".format(mar), (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

		if mar > mouth_thresh:
			# Verificați dacă a trecut suficient timp de la ultima cascătură
			if time.time() - last_yawn > yawn_interval:
				# Contorizați cascătura
				yawns += 1
				# Actualizați timestamp-ul
				last_yawn = time.time()
			if yawns == 3:
				cv2.putText(frame, "VA RUGAM LUATI O PAUZA!!!", (30, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
				mixer.music.play()

			

	#afisam frameul la noi pe ecran
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
	if key == 27:
		break
cv2.destroyAllWindows()
cap.release() 
