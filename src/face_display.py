import cv2 as cv
import graphics.faces as face

#BMO face response to text
def calculate_emotional_response(current_emotion):
    #returns the url to the img of the current emotion
    return face.face_dictionary[current_emotion]

def main():

    #default face
    path = face.face_dictionary['happy']
    print(path)
    current_face = cv.imread(path)
    cv.imshow('happy', current_face )

    cv.waitKey(0)
    # display default bmo face

    #get user emotion input
    # while True:
    #     # options are : content, happy, angry, worried, shocked, and 'q' to quit
    #     current_emotion = input("How Do you feel?")
    #     if current_emotion == 'q':
    #         break
    #     #change bmo face to that emotion
    #     current_face = calculate_emotional_response(current_emotion)
    #     cv.imshow('BMO face', current_face)
    #     cv.waitKey(0)
    #
    #repeat until program exit

    cv.destroyAllWindows()
