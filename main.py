#to import face rec
import face_recognition
#to open files located in our computer
import os
#to label and draw around faces
import cv2

#the known and unknow faces' folders on our computer
KnownFaces_dir = 'known_faces'
UnknowFaces_dir = 'unknown_faces'
#is a value from 0 to 1 the more you go down the more it will be accurte but -it may not recognize any of the people if it's so low-
TOLERANCE = 0.6
#the thickness of the rectangle the program will draw around the face
FRAME_THICKNESS = 3
#the thickness of the font
FONT_THICKNESS = 2
#cnn >> Convolutional neural network used to analyzing visual data like photos .. hog also used for face rec less accurte and rely on the property of objects -blocks- 
MODEL = 'hog' #cnn


#change color based on the first 3 letter
def name_to_color(name):
    #it takes the order of the 3 first letter -the unicoding of them- makes them as an array which we can use as a RGB color
    color = [(ord(c.lower())-97)*8 for c in name[:3]]
    return color
print('Loading known faces...')
#list of the faces that will will encode and make them to arrayes to compare them to unknown faces
known_faces = []
#list of folders name's
known_names = []
#exploring every folder in known faces
for name in os.listdir(KnownFaces_dir):  
    #exploring every photo in the folder
    for filename in os.listdir(f'{KnownFaces_dir}/{name}'): 
        #upload the image
        image = face_recognition.load_image_file(f'{KnownFaces_dir}/{name}/{filename}')  
       #encode the face >> make it into arrays
        encoding = face_recognition.face_encodings(image)[0]     
        #append the arrays of the face in known_faces list 
        known_faces.append(encoding)
        #append the name of the folder we encoded the face from
        known_names.append(name)

print('Processing unknown faces...')
#exploring unknown_faces folder
for filename in os.listdir(UnknowFaces_dir):   
    #print the photo name , end the print with space without newline
    print(f'Filename {filename}', end='')
    #upload the image
    image = face_recognition.load_image_file(f'{UnknowFaces_dir}/{filename}')
    #locate the face location using model #cnn or #hog
    locations = face_recognition.face_locations(image, model=MODEL)
    #encode the face >> make it into arrays
    encodings = face_recognition.face_encodings(image, locations)
    #convert the image from RGB to BGR since OpenCV uses BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    #print how many faces are in the pic
    print(f', found {len(encodings)} face(s)')
    #put the encoding list and locations list in the same list 
    for face_encoding, face_location in zip(encodings, locations):
        #compare our known faces to the unknown faces
        results = face_recognition.compare_faces(known_faces, face_encoding, TOLERANCE)
        match = None
        #if the unknown faces is in the known faces list ..
        if True in results:  
            #make match ver = to the name of the chrachter 
            match = known_names[results.index(True)]
            print(f' - {match} from {results}')  
            #puting the dimensions of the rectangle around the face
            top_left = (face_location[3], face_location[0])
            bottom_right = (face_location[1], face_location[2])
            #get the color from the name
            color = name_to_color(match) 
            #drawing the rectangle (the face , top and left line,bottom and right line ,the color of the rec,the frame thickness )
            cv2.rectangle(image, top_left, bottom_right, color, FRAME_THICKNESS)  
            #puting the dimensions of the rectangle around the name
            top_left = (face_location[3], face_location[2])
            bottom_right = (face_location[1], face_location[2] + 22)
            #drawing the rectangle (the face , top and left line,bottom and right line ,the color of the rec,the frame thickness )
            cv2.rectangle(image, top_left, bottom_right, color, cv2.FILLED)  
            #typing the name inside the rec (the face , the name , the position of the text , the font , fontScale , the color , font thickness)          
            cv2.putText(image, match, (face_location[3] + 10, face_location[2] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), FONT_THICKNESS)
    # Output img with window name as filename 
    cv2.imshow(filename, image)
    #let the window open untill the user hit any key
    cv2.waitKey(0)
    cv2.destroyWindow(filename)