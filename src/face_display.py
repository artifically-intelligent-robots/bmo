import cv2 as cv
import graphics.faces as face

#BMO face response to text
def calculate_emotional_response(current_emotion):
    #returns the url to the img of the current emotion
    ## TODO: add error checking for data input
    return face.face_dictionary[current_emotion]

def main():
    # display default face
    initial_face = True
    path = face.face_dictionary['happy']
    current_face = cv.imread(path)
    current_emotion = None


    # options are : content, sad, happy, angry, worried, shocked, and 'q' to quit
    k = 0
    while current_emotion != 'q':
        #display current face
        cv.imshow('BMO face', current_face )
        k = cv.waitKey(1)
        # get next emotion
        current_emotion = input("How Do you feel?\n")
        if current_emotion != 'q':
            path = calculate_emotional_response(current_emotion)
            current_face = cv.imread(path)

    cv.destroyAllWindows()



if __name__ == "__main__":
    main()
