import cv2 as cv
import graphics.faces as face

#BMO face response to text
def calculate_emotional_response(current_emotion):
    #returns the url to the img of the current emotion
    return face.face_dictionary[current_emotion]

def main():
    # display default face
    path = face.face_dictionary['happy']
    current_face = cv.imread(path)
    cv.imshow('happy', current_face )
    cv.waitKey(1)


    # options are : content, sad, happy, angry, worried, shocked, and 'q' to quit
    k = 0
    while k != ord('q'):
        # get user emotion input
        current_emotion = input("How Do you feel?\n")
        print()

        #change bmo face to that emotion
        path = calculate_emotional_response(current_emotion)
        current_face = cv.imread(path)
        cv.imshow('BMO face', current_face)

        # 'q' to quit loop
        k = cv.waitKey(0)

    cv.destroyAllWindows()



if __name__ == "__main__":
    main()
