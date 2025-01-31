import cv2


#cap = cv2.VideoCapture(0)
#cap2 = cv2.VideoCapture(2)
cap = cv2.VideoCapture(0)
num = 0
chessboardSize = (9,5)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

itr =0 
while cap.isOpened():

   # succes1, img = cap.read()
   # succes2, img2 = cap2.read()q
    ret, frame = cap.read()

    height, width = frame.shape[:2]
    frame_left = frame[:, :width//2]
    frame_right = frame[:, width//2:]
    grayL = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)
    retL, cornersL = cv2.findChessboardCorners(grayL, chessboardSize, None)
    retR, cornersR = cv2.findChessboardCorners(grayR, chessboardSize, None)

    print(retL)
    print(retR)
    print(itr)
    print()
    itr+=1


    k = cv2.waitKey(5)

    if k == 27:
        break
    elif k == ord('s'):   # wait for 's' key to save and exit  , add #and  retL  == True and retR == True:
        cv2.imwrite('ejad_images/stereoLeft/imageLdepthtest' + str(num) + '.png', frame_left)
        cv2.imwrite('ejad_images/stereoRight/imageRdepthtest' + str(num) + '.png', frame_right)
        print("images saved!: ", num )
        num += 1

    cv2.imshow('frame_left',frame_left)
    cv2.imshow('frame_right',frame_right)
